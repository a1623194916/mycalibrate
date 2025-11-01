#!/usr/bin/env python3
"""
按时间给图片排序并将重命名后的拷贝保存到新目录（不修改源文件）。
特性:
 - 支持使用 EXIF DateTimeOriginal（如果存在）或文件的修改时间/创建时间作为时间依据
 - dry-run 模式（只打印将要做的拷贝，不执行）
 - 支持保留原扩展或强制指定扩展（例如 .png）
 - 支持自定义零填充宽度（默认 3 -> 000）
 - 可选择是否递归子目录（默认为否）

Usage:
  python sort_copy_by_time.py /path/to/images --use-exif --ext png --pad 3 --dry-run
  python .\sort_copy_by_time.py .\Color --use-exif --ext png --pad 3     
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from shutil import copy2
from PIL import Image
from PIL.ExifTags import TAGS


def get_exif_datetime(path: Path):
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag in ("DateTimeOriginal", "DateTime"):
                try:
                    dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    return dt
                except Exception:
                    return None
        return None
    except Exception:
        return None


def file_time(path: Path, prefer='mtime'):
    st = path.stat()
    if prefer == 'ctime':
        return datetime.fromtimestamp(st.st_ctime)
    return datetime.fromtimestamp(st.st_mtime)


def collect_images(folder: Path, patterns, recursive=False):
    if not recursive:
        return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower().lstrip('.') in patterns]
    else:
        return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower().lstrip('.') in patterns]


def main():
    parser = argparse.ArgumentParser(description="按时间顺序复制并重命名图片到新文件夹，从 000 开始（不修改源文件）")
    parser.add_argument("folder", type=str, help="图片所在文件夹路径")
    parser.add_argument("--use-exif", action="store_true", help="优先使用 EXIF DateTimeOriginal（若存在）")
    parser.add_argument("--time-fallback", choices=("mtime","ctime"), default="mtime", help="EXIF 不存在时使用文件时间: 修改时间(mtime) 或 创建时间(ctime)，默认 mtime")
    parser.add_argument("--ext", type=str, default=None, help="强制输出扩展名（例如 png）；默认保留原扩展")
    parser.add_argument("--pad", type=int, default=3, help="零填充宽度，默认 3（000）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的拷贝操作，不真正执行")
    parser.add_argument("--recursive", action="store_true", help="递归子目录")
    parser.add_argument("--out", type=str, default="sorted_out", help="输出目录（相对于输入目录的子目录或绝对路径），默认 sorted_out")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        print("错误：指定路径不是目录或不存在。", folder)
        sys.exit(1)

    patterns = {"jpg","jpeg","png","bmp","tiff","tif","webp"}
    images = collect_images(folder, patterns, recursive=args.recursive)

    if len(images) == 0:
        print("未找到图片文件。")
        return

    items = []
    for p in images:
        dt = None
        if args.use_exif:
            dt = get_exif_datetime(p)
        if dt is None:
            dt = file_time(p, prefer=args.time_fallback)
        items.append((p, dt))

    items.sort(key=lambda x: (x[1], str(x[0])))

    out_root = Path(args.out)
    # 如果指定的是相对路径且输入是文件夹，放到输入目录下
    if not out_root.is_absolute():
        out_root = folder / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    pad = args.pad
    ops = []  # tuples of (src_path, dst_path, dt)
    for idx, (p, dt) in enumerate(items):
        new_base = f"{idx:0{pad}d}"
        out_ext = (args.ext.lower() if args.ext else p.suffix.lstrip('.')).lower()
        out_name = f"{new_base}.{out_ext}"
        dst = out_root / out_name
        ops.append((p, dst, dt))

    # 检查冲突
    dst_names = [str(dst) for _, dst, _ in ops]
    if len(set(dst_names)) != len(dst_names):
        print("错误：目标文件名有冲突，请增加 pad 宽度或检查扩展名设置。")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN: 将执行以下复制（按时间排序）：")
        for src, dst, dt in ops:
            print(f"{src.name}  ({dt})  ->  {dst}")
        return

    # 执行复制（保留原文件不修改）
    for src, dst, dt in ops:
        if dst.exists():
            print(f"目标已存在，跳过: {dst}")
            continue
        print(f"复制: {src} -> {dst}")
        # 使用 copy2 保留元数据
        copy2(src, dst)

    print(f"完成，文件已复制到 {out_root}")


if __name__ == "__main__":
    main()

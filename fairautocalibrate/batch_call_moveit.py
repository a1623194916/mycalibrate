#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys


def read_valid_lines(txt_path: str):
    valid_indices = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 6:
                raise ValueError(f"{txt_path}:{idx} 每行需要 6 个数，实际 {len(parts)} 个: {stripped}")
            valid_indices.append(idx)
    return valid_indices


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_py = os.getenv("MOVEIT_TEST_PY", "/home/u22/lwh/fair_ws/test.py")
    poses_txt = os.path.join(script_dir, "robot_poses.txt")
    auto_enter = os.getenv("MOVEIT_AUTO_ENTER", "0") == "1"
    timeout_s = float(os.getenv("MOVEIT_TIMEOUT_S", "3"))

    if not os.path.exists(test_py):
        print(f"未找到 test.py: {test_py}")
        return 2
    if not os.path.exists(poses_txt):
        print(f"未找到 robot_poses.txt: {poses_txt}")
        return 3

    line_indices = read_valid_lines(poses_txt)
    if not line_indices:
        print("robot_poses.txt 无有效数据")
        return 4

    print(f"将依次执行 {len(line_indices)} 条位姿")

    for line_no in line_indices:
        print(f"\n[调用] 第 {line_no} 行")
        try:
            run_input = "\n" if auto_enter else None
            result = subprocess.run(
                ["python3", test_py, str(line_no), "move_j", "auto"],
                input=run_input,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=script_dir,
            )
        except subprocess.TimeoutExpired:
            if auto_enter:
                print(f"[超时] 第 {line_no} 行执行超时")
            else:
                print(f"[预览结束] 第 {line_no} 行已中止等待输入")
            continue

        if result.returncode != 0:
            print(f"[失败] 第 {line_no} 行返回码: {result.returncode}")
            if result.stderr:
                print(result.stderr)
            continue

        if result.stdout:
            print(result.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http:# www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************

import pyorbbecsdk as ob
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
import cv2
import numpy as np
import os


def pixel_to_cam_point(u, v, depth_mm, intrinsics):
    """将像素+深度转换为相机坐标系 3D 点 (mm)。"""
    fx, fy, cx, cy = intrinsics
    x = (u - cx) * depth_mm / fx
    y = (v - cy) * depth_mm / fy
    z = depth_mm
    return np.array([x, y, z], dtype=np.float64)


def cam_to_end(point_cam_mm, handeye_T):
    """使用手眼标定 4x4 变换矩阵，将相机坐标(mm)变换到末端坐标(mm)。"""
    p = np.array([point_cam_mm[0], point_cam_mm[1],
                 point_cam_mm[2], 1.0], dtype=np.float64)
    out = handeye_T @ p
    return out[:3]


def cam_to_base(point_cam_mm, handeye_T, tcp_pose_mm_deg):
    """相机坐标(mm) -> 基座坐标(mm)，需要当前 TCP 位姿。
    
    tcp_pose_mm_deg: [x, y, z, rx, ry, rz]，单位 mm 和度
    """
    from scipy.spatial.transform import Rotation as R
    
    # 1. 相机 -> 末端
    pt_end = cam_to_end(point_cam_mm, handeye_T)
    
    # 2. 末端 -> 基座
    x1, y1, z1, rx, ry, rz = tcp_pose_mm_deg
    T_base_to_end = np.eye(4, dtype=np.float64)
    T_base_to_end[:3, :3] = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    T_base_to_end[:3, 3] = [x1, y1, z1]
    
    pt_end_homo = np.array([pt_end[0], pt_end[1], pt_end[2], 1.0], dtype=np.float64)
    pt_base_homo = T_base_to_end @ pt_end_homo
    return pt_base_homo[:3]


def compute_point_from_images(rgb_path, depth_path, u, v, intrinsics, depth_scale, handeye_T):
    """
    读取 RGB + 深度图，给定像素(u,v)与相机内参，输出相机坐标和手眼变换后的 3D 点。

    intrinsics: (fx, fy, cx, cy)
    depth_scale: 深度单位 -> mm 的缩放（如深度图为 mm，设为 1.0）
    handeye_T: 4x4 手眼标定矩阵（相机到基座/末端）
    """
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(rgb_path)
    if not os.path.exists(depth_path):
        raise FileNotFoundError(depth_path)

    color = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError("RGB 读取失败")

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("深度图读取失败")

    if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
        raise ValueError("像素坐标超出图像范围")

    depth_raw = float(depth[int(v), int(u)])
    depth_mm = depth_raw * depth_scale
    if depth_mm <= 0:
        raise ValueError("深度为 0 或无效")

    pt_cam = pixel_to_cam_point(u, v, depth_mm, intrinsics)
    pt_end = cam_to_end(pt_cam, handeye_T)
    return pt_cam, pt_end


def select_point_from_images(rgb_path, depth_path, intrinsics, depth_scale, handeye_T):
    """
    弹窗显示 RGB/Depth 叠加图，鼠标左键选点，返回相机坐标和手眼变换后的 3D 点。
    """
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(rgb_path)
    if not os.path.exists(depth_path):
        raise FileNotFoundError(depth_path)

    color = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError("RGB 读取失败")

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("深度图读取失败")

    depth_mm = depth.astype(np.float32) * depth_scale
    depth_norm = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(color, 0.6, depth_color, 0.4, 0)

    result = {"pt_cam": None, "pt_end": None}

    def _on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not (0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]):
            print("Clicked outside image bounds")
            return
        d = float(depth[y, x]) * depth_scale
        if d <= 0:
            print("Depth at clicked pixel is 0 (invalid)")
            return
        pt_cam = pixel_to_cam_point(x, y, d, intrinsics)
        pt_end = cam_to_end(pt_cam, handeye_T)
        result["pt_cam"] = pt_cam
        result["pt_end"] = pt_end
        print(f"Clicked pixel: ({x}, {y}), depth(mm): {d:.2f}")
        print(f"相机坐标(mm): {pt_cam}")
        print(f"末端坐标(mm): {pt_end}")

    win = "Offline RGBD Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, blended)
    cv2.setMouseCallback(win, _on_click)

    while True:
        key = cv2.waitKey(30)
        if result["pt_cam"] is not None:
            break
        if key in [27, ord('q')]:
            break

    cv2.destroyAllWindows()
    return result["pt_cam"], result["pt_end"]


def get_stream_config(pipeline: Pipeline):
    config = Config()
    try:
        profile_list = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR)
        assert profile_list is not None

        for i in range(len(profile_list)):
            color_profile = profile_list[i]
            if color_profile.get_format() != OBFormat.RGB:
                continue

            hw_d2c_profile_list = pipeline.get_d2c_depth_profile_list(
                color_profile, OBAlignMode.HW_MODE)
            if len(hw_d2c_profile_list) == 0:
                continue

            hw_d2c_profile = hw_d2c_profile_list[0]
            print("hw_d2c_profile: ", hw_d2c_profile)

            config.enable_stream(hw_d2c_profile)
            config.enable_stream(color_profile)
            config.set_align_mode(OBAlignMode.HW_MODE)
            return config
    except Exception as e:
        print(e)
        return None
    return None


def on_mouse_click(event, x, y, flags, param):
    """Mouse callback function to get depth value at a clicked pixel and convert to 3D.

    param is a tuple: (depth_data, color_intrinsics, depth_intrinsics, extrinsic)
    - depth_data: ndarray of raw uint16 depth values (units: millimeters)
    - color_intrinsics: OBIntrinsic from color stream profile
    - depth_intrinsics: OBIntrinsic from depth stream profile
    - extrinsic: extrinsic transform from depth to color (depth_profile.get_extrinsic_to(color_profile))
    """
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        try:
            depth_data, color_intrinsics, depth_intrinsics, extrinsic = param
            # print("1111", extrinsic)
        except Exception:
            print("Mouse callback parameters invalid")
            return

        if not (0 <= y < depth_data.shape[0] and 0 <= x < depth_data.shape[1]):
            print("Clicked outside image bounds")
            return

        depth_value = int(depth_data[y, x])  # 原始深度值（单位：毫米）
        print(f"Clicked pixel: ({x}, {y}), Depth value (raw): {depth_value}")

        if depth_value == 0:
            print("Depth at clicked pixel is 0 (invalid)")
            return

        # 使用 SDK 提供的 transformation2dto3d 将 2D+depth 转为 3D 点
        # D2C 模式下的疑问：应该用彩色内参还是深度内参？
        try:
            pt2 = ob.OBPoint2f(float(x), float(y))

            # 方法1：使用彩色相机内参（假设深度已完全对齐到彩色空间）
            pt3_color = ob.transformation2dto3d(
                pt2, depth_value, color_intrinsics, extrinsic)

            test = 0
            if (test):
                # 方法2：使用深度相机内参（深度值来自深度传感器）
                pt3_depth = ob.transformation2dto3d(
                    pt2, depth_value, depth_intrinsics, extrinsic)

                print(f"\n🔍 D2C 模式 - 两种内参的对比:")
                print(
                    f"方法1 (彩色内参): x={pt3_color.x:.2f}, y={pt3_color.y:.2f}, z={pt3_color.z:.2f} mm")
                print(
                    f"方法2 (深度内参): x={pt3_depth.x:.2f}, y={pt3_depth.y:.2f}, z={pt3_depth.z:.2f} mm")
                print(
                    f"差异: ΔX={abs(pt3_color.x-pt3_depth.x):.2f}, ΔY={abs(pt3_color.y-pt3_depth.y):.2f}, ΔZ={abs(pt3_color.z-pt3_depth.z):.2f} mm")

                # 方法3：手动计算（使用彩色内参，假设 D2C 完全对齐）
                fx_c = color_intrinsics.fx
                fy_c = color_intrinsics.fy
                cx_c = color_intrinsics.cx
                cy_c = color_intrinsics.cy

                X_manual = (x - cx_c) * depth_value / fx_c
                Y_manual = (y - cy_c) * depth_value / fy_c
                Z_manual = depth_value

                print(
                    f"方法3 (手动-彩色内参): x={X_manual:.2f}, y={Y_manual:.2f}, z={Z_manual:.2f} mm")
                print(
                    f"与方法1差异: ΔX={abs(pt3_color.x-X_manual):.2f}, ΔY={abs(pt3_color.y-Y_manual):.2f}, ΔZ={abs(pt3_color.z-Z_manual):.2f} mm")

                # 额外验证：打印外参看看是否是单位矩阵
                print(f"\n📐 外参信息:")
                print(f"   旋转矩阵: {extrinsic.rot}")
                print(f"   平移向量: {extrinsic.transform}")

            # 默认使用彩色内参的结果
            pt3 = pt3_color

            # 将结果保存在全局变量，供 d2crun 返回
            try:
                global _d2c_last_3d_point
                _d2c_last_3d_point = (float(pt3.x), float(pt3.y), float(pt3.z))
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to compute 3D point: {e}")


def d2crun(rgb_path, depth_path, intrinsics, depth_scale=1.0):
    """
    从离线 RGB + 深度图获取点击的相机 3D 坐标。
    
    参数:
        rgb_path: RGB 图像路径
        depth_path: 深度图路径
        intrinsics: (fx, fy, cx, cy) 相机内参
        depth_scale: 深度图单位->mm 的缩放（默认1.0，深度已是mm）
    
    返回:
        (x, y, z) 相机坐标系下的 3D 点，单位 mm；如果取消则返回 None
    """
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB 图像不存在: {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"深度图不存在: {depth_path}")

    color_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color_image is None:
        raise RuntimeError(f"无法读取 RGB 图像: {rgb_path}")

    depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_data is None:
        raise RuntimeError(f"无法读取深度图: {depth_path}")

    depth_data_float = depth_data.astype(np.float32) * depth_scale

    min_depth = 20  # mm
    max_depth = 10000  # mm
    depth_data_float = np.clip(depth_data_float, min_depth, max_depth)

    depth_image = cv2.normalize(depth_data_float, None, 0, 255, cv2.NORM_MINMAX)
    depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)

    blended_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)

    result = {"pt_cam": None}

    def _on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not (0 <= y < depth_data.shape[0] and 0 <= x < depth_data.shape[1]):
            print("点击超出图像范围")
            return
        depth_raw = depth_data[y, x]
        # 如果是多通道，取第一个通道
        if isinstance(depth_raw, np.ndarray):
            depth_raw = depth_raw.flat[0]
        depth_value = float(depth_raw) * depth_scale
        if depth_value <= 0:
            print(f"点击像素 ({x}, {y}) 深度无效: {depth_value}")
            return
        
        fx, fy, cx, cy = intrinsics
        X = (x - cx) * depth_value / fx
        Y = (y - cy) * depth_value / fy
        Z = depth_value
        
        result["pt_cam"] = (X, Y, Z)
        print(f"点击像素: ({x}, {y}), 深度: {depth_value:.2f} mm")
        print(f"相机坐标(mm): ({X:.2f}, {Y:.2f}, {Z:.2f})")

    win = "HW D2C Align Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 800)
    cv2.setMouseCallback(win, _on_click)

    while True:
        cv2.imshow(win, blended_image)
        if result["pt_cam"] is not None:
            break
        key = cv2.waitKey(30)
        if key in [27, ord('q')]:
            break

    cv2.destroyAllWindows()
    return result["pt_cam"]


if __name__ == "__main__":
    # 示例：离线图像验证手眼标定
    # 修改为你的相机内参和手眼矩阵（单位：mm）
    intrinsics = (604.25993192, 604.03556638, 643.75378798,
                  363.27535391)  # fx, fy, cx, cy
    handeye_T = np.array(
        [
            [0.98508102, -0.15166231, 0.08132608, -27.58542],
            [0.14750823, 0.98753621, 0.05489591, -91.75181],
            [-0.08863809, -0.04208065, 0.99517461, -221.05245],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # depth_scale: 深度图单位->mm（若深度已是mm，设1.0；若是米，设1000.0）
    depth_scale = 1.0

    # 替换为你的图像路径
    rgb_path = "rgb.png"
    depth_path = "depth.png"

    try:
        pt_cam, pt_end = select_point_from_images(
            rgb_path, depth_path, intrinsics, depth_scale, handeye_T
        )
        if pt_cam is not None:
            print(f"相机坐标(mm): {pt_cam}")
            print(f"末端坐标(mm): {pt_end}")
            
            # 如果需要基座坐标，传入当前 TCP 位姿（示例）
            # tcp_pose = [x1, y1, z1, rx, ry, rz]  # mm 和度
            # pt_base = cam_to_base(pt_cam, handeye_T, tcp_pose)
            # print(f"基座坐标(mm): {pt_base}")
    except Exception as e:
        print(f"验证失败: {e}")

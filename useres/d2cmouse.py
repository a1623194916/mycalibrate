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
from operations import frame_to_bgr_image


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
            
            test=0
            if(test):
                # 方法2：使用深度相机内参（深度值来自深度传感器）
                pt3_depth = ob.transformation2dto3d(
                    pt2, depth_value, depth_intrinsics, extrinsic)
                
                print(f"\n🔍 D2C 模式 - 两种内参的对比:")
                print(f"方法1 (彩色内参): x={pt3_color.x:.2f}, y={pt3_color.y:.2f}, z={pt3_color.z:.2f} mm")
                print(f"方法2 (深度内参): x={pt3_depth.x:.2f}, y={pt3_depth.y:.2f}, z={pt3_depth.z:.2f} mm")
                print(f"差异: ΔX={abs(pt3_color.x-pt3_depth.x):.2f}, ΔY={abs(pt3_color.y-pt3_depth.y):.2f}, ΔZ={abs(pt3_color.z-pt3_depth.z):.2f} mm")
                
                # 方法3：手动计算（使用彩色内参，假设 D2C 完全对齐）
                fx_c = color_intrinsics.fx
                fy_c = color_intrinsics.fy
                cx_c = color_intrinsics.cx
                cy_c = color_intrinsics.cy
                
                X_manual = (x - cx_c) * depth_value / fx_c
                Y_manual = (y - cy_c) * depth_value / fy_c
                Z_manual = depth_value
                
                print(f"方法3 (手动-彩色内参): x={X_manual:.2f}, y={Y_manual:.2f}, z={Z_manual:.2f} mm")
                print(f"与方法1差异: ΔX={abs(pt3_color.x-X_manual):.2f}, ΔY={abs(pt3_color.y-Y_manual):.2f}, ΔZ={abs(pt3_color.z-Z_manual):.2f} mm")
                
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


def d2crun():
    pipeline = Pipeline()
    config = get_stream_config(pipeline)
    if config is None:
        return

    pipeline.start(config)

    min_depth = 20  # Minimum depth value, keep closer depths,mm
    max_depth = 10000  # Maximum depth value, allow far depths to be lost,mm

    # 清除上次点击结果（使用 globals().pop 避免在函数作用域里把名字标记为局部变量）
    globals().pop('_d2c_last_3d_point', None)

    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        depth_format = depth_frame.get_format()
        if depth_format != OBFormat.Y16:
            print("depth format is not Y16")
            continue

        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            print("Failed to convert frame to image")
            continue

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))

        depth_scale = depth_frame.get_depth_scale()
        depth_data_float = depth_data.astype(np.float32) * depth_scale

        # 获取深度/彩色流的配置与内参、外参，用于像素->3D 转换
        try:
            depth_vf = depth_frame.as_video_frame()
            color_vf = color_frame.as_video_frame()
            depth_profile = depth_vf.get_stream_profile()
            color_profile = color_vf.get_stream_profile()
            # depth_intrinsics 用于 transformation2dto3d
            depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic()

            extrinsic = depth_profile.get_extrinsic_to(
                color_profile)  # 深度到彩色的外参
        except Exception as e:
            depth_intrinsics = None
            extrinsic = None
            print(f"Warning: failed to get intrinsics/extrinsic: {e}")

        depth_data_float = np.clip(
            depth_data_float, min_depth, max_depth)  # 限制深度范围，mm

        depth_image = cv2.normalize(
            depth_data_float, None, 0, 255, cv2.NORM_MINMAX)  # 归一化到0-255，单通道，为什么
        depth_image = cv2.applyColorMap(
            depth_image.astype(np.uint8), cv2.COLORMAP_JET)

        blended_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)

        # 创建窗口
        cv2.namedWindow("HW D2C Align Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HW D2C Align Viewer", 1280, 800)  # 设置为图像原始分辨率（1:1 显示）

        # 设置鼠标回调,传入 depth_data、color_intrinsics、depth_intrinsics 与 extrinsic（用于反投影）
        cv2.setMouseCallback("HW D2C Align Viewer", on_mouse_click, param=(
            depth_data, color_intrinsics, depth_intrinsics, extrinsic))

        # 显示图像
        cv2.imshow("HW D2C Align Viewer", blended_image)
        # 如果鼠标回调已经设置了全局3D点，则结束并返回该点
        if '_d2c_last_3d_point' in globals():
            result = _d2c_last_3d_point
            pipeline.stop()
            cv2.destroyAllWindows()
            # print(f"Returning 3D point: {result}")
            return result

        if cv2.waitKey(1) in [ord('q'), 27]:  # 27 is the ESC key
            break

    pipeline.stop()
    cv2.destroyAllWindows()
    return None


if __name__ == "__main__":
    pt = d2crun()
    if pt is not None:
        print(f"Clicked 3D point returned from d2crun(): {pt}")

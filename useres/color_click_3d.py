# ******************************************************************************
#  彩色图像鼠标点击获取 3D 坐标工具
#  参考 coordinate_transform.py 实现
# ******************************************************************************

import cv2
import numpy as np
import pyorbbecsdk as ob
import argparse
import time

# 全局变量存储当前帧数据
clicked_point = None
current_color_frame = None
current_depth_frame = None
current_color_image = None

def get_3d_from_pixel(color_frame, depth_data, x, y):
    """
    color_frame: video frame object (has get_stream_profile -> as_video_stream_profile -> get_intrinsic())
    depth_data: numpy 2D array of raw depth values (uint16), 单位 mm
    x, y: int pixel coordinates in color image (and in D2C mode they index depth_data)
    returns: (X, Y, Z) in mm (floats), 或 None if invalid depth
    """
    # 获取彩色相机内参
    color_intr = color_frame.as_video_frame().get_stream_profile().as_video_stream_profile().get_intrinsic()
    fx = color_intr.fx
    fy = color_intr.fy
    cx = color_intr.cx
    cy = color_intr.cy

    h, w = depth_data.shape
    if not (0 <= x < w and 0 <= y < h):
        return None

    z = float(depth_data[y, x])  # 原始深度值，单位 mm
    if z <= 0:
        return None

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    return (X, Y, Z)

def mouse_callback(event, x, y, flags, param):
    """鼠标点击回调 - 在彩色图像上点击"""
    global clicked_point, _color_last_3d_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        
        if current_depth_frame is None or current_color_frame is None:
            print("❌ 帧数据未就绪")
            return
        
        print(f"\n{'='*60}")
        print(f"🖱️  彩色图像点击位置: ({x}, {y})")
        
        try:
            # 获取相机参数（参考 coordinate_transform.py）
            color_frame_video = current_color_frame.as_video_frame()
            depth_frame_video = current_depth_frame.as_video_frame()
            
            depth_width = depth_frame_video.get_width()
            depth_height = depth_frame_video.get_height()
            
            color_profile = color_frame_video.get_stream_profile()
            depth_profile = depth_frame_video.get_stream_profile()
            
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic() # 获取彩色相机内参
            color_distortion = color_profile.as_video_stream_profile().get_distortion() # 获取彩色相机畸变参数
            depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
            depth_distortion = depth_profile.as_video_stream_profile().get_distortion()
            extrinsic_deptocolor = depth_profile.get_extrinsic_to(color_profile) # 深度相机到彩色相机的外参
            extrinsic_colortodep = color_profile.get_extrinsic_to(depth_profile) # 彩色相机到深度相机的外参

            # 获取深度数据
            depth_data = np.frombuffer(
                depth_frame_video.get_data(), 
                dtype=np.uint16
            ).reshape(depth_height, depth_width)
            
            # 步骤1：将彩色图像的2D点映射到深度图像的2D点
            # 使用 SDK 的 transformation2dto2d
            depth_2d_point = ob.transformation2dto2d(
                ob.OBPoint2f(float(x), float(y)),
                1000,  # 假设深度为1000mm（后续会用实际深度）
                color_intrinsics,
                color_distortion,
                depth_intrinsics,
                depth_distortion,
                extrinsic_colortodep
            )
            
            # 获取深度图中对应点的深度值
            dx = int(depth_2d_point.x)
            dy = int(depth_2d_point.y)
            
            if 0 <= dx < depth_width and 0 <= dy < depth_height:
                depth_value = depth_data[dy, dx]
                print(f"📏 深度图对应位置: ({dx}, {dy})")
                print(f"📏 深度值: {depth_value} mm ({depth_value/1000.0:.3f} m)")
                
                if depth_value > 0:
                    # 方法1：使用 SDK 的 transformation2dto3d
                    # 直接从深度图2D点 + 外参 转换到彩色相机坐标系的3D点
                    point_3d_color_sdk = ob.transformation2dto3d(
                        ob.OBPoint2f(float(dx), float(dy)),  # 深度图的2D点
                        depth_value,
                        depth_intrinsics,     # 深度相机内参
                        extrinsic_deptocolor  # 深度→彩色外参（SDK内部自动完成坐标系转换）
                    )
                    
                    print(f"\n📍 3D 坐标 (彩色相机坐标系 - SDK 方法):")
                    print(f"   X = {point_3d_color_sdk.x:.2f} mm ({point_3d_color_sdk.x/1000.0:.4f} m)")
                    print(f"   Y = {point_3d_color_sdk.y:.2f} mm ({point_3d_color_sdk.y/1000.0:.4f} m)")
                    print(f"   Z = {point_3d_color_sdk.z:.2f} mm ({point_3d_color_sdk.z/1000.0:.4f} m)")
                    
                    # 方法2：手动复现 SDK 的计算（验证理解是否正确）
                    # 步骤1: 深度图 2D → 深度相机 3D
                    manual=0
                    if(manual):
                        fx_d = depth_intrinsics.fx
                        fy_d = depth_intrinsics.fy
                        cx_d = depth_intrinsics.cx
                        cy_d = depth_intrinsics.cy
                        
                        X_depth = (dx - cx_d) * depth_value / fx_d
                        Y_depth = (dy - cy_d) * depth_value / fy_d
                        Z_depth = depth_value
                        
                        print(f"\n🔧 手动计算步骤:")
                        print(f"   步骤1 - 深度相机 3D 坐标 (使用深度图坐标 {dx}, {dy}):")
                        print(f"   X = {X_depth:.2f} mm, Y = {Y_depth:.2f} mm, Z = {Z_depth:.2f} mm")
                    
                        # 步骤2: 通过外参转换到彩色相机坐标系
                        # 外参包含旋转矩阵 R (3x3) 和平移向量 t (3x1)
                        # 转换公式: P_color = R * P_depth + t
                        
                        # 获取外参的旋转和平移
                        rot = np.array(extrinsic_deptocolor.rot).reshape(3, 3)
                        trans = np.array(extrinsic_deptocolor.transform)  # 注意：属性名是 transform，不是 trans
                        
                        point_depth = np.array([X_depth, Y_depth, Z_depth])
                        point_color_manual = rot @ point_depth + trans
                        
                        print(f"   步骤2 - 转换到彩色相机坐标系 (应用外参):")
                        print(f"   X = {point_color_manual[0]:.2f} mm ({point_color_manual[0]/1000.0:.4f} m)")
                        print(f"   Y = {point_color_manual[1]:.2f} mm ({point_color_manual[1]/1000.0:.4f} m)")
                        print(f"   Z = {point_color_manual[2]:.2f} mm ({point_color_manual[2]/1000.0:.4f} m)")
                        
                        # 计算与 SDK 的差异
                        diff_x = abs(point_3d_color_sdk.x - point_color_manual[0])
                        diff_y = abs(point_3d_color_sdk.y - point_color_manual[1])
                        diff_z = abs(point_3d_color_sdk.z - point_color_manual[2])
                        
                        print(f"\n📊 手动计算与 SDK 的差异:")
                        print(f"   ΔX = {diff_x:.2f} mm")
                        print(f"   ΔY = {diff_y:.2f} mm")
                        print(f"   ΔZ = {diff_z:.2f} mm")
                        
                        if diff_x < 1 and diff_y < 1 and diff_z < 1:
                            print(f"   ✅ 差异 < 1mm，手动计算正确！")
                        
                        print(f"\n💡 关键理解:")
                        print(f"   - 彩色图坐标 ({x}, {y}) 和深度图坐标 ({dx}, {dy}) 是不同的")
                        print(f"   - SDK 使用深度图坐标 + 深度相机内参计算深度相机 3D 点")
                        print(f"   - 然后通过外参转换到彩色相机坐标系")
                        print(f"   - 不能直接用彩色图坐标 + 彩色内参计算！")
                    # 保存 SDK 计算的 3D 点，供外部函数读取
                    try:
                        _color_last_3d_point = (float(point_3d_color_sdk.x),
                                                float(point_3d_color_sdk.y),
                                                float(point_3d_color_sdk.z))
                    except Exception:
                        pass
                else:
                    print("❌ 该点深度值为 0（无效深度）")
            else:
                print(f"❌ 映射到深度图的坐标超出范围: ({dx}, {dy})")
                
        except Exception as e:
            print(f"❌ 转换失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*60}\n")


def process_pixel(color_frame, depth_frame, depth_data, x, y):
    """
    处理指定像素（与鼠标点击逻辑相同），打印深度与 3D 坐标信息。
    color_frame: color frame object
    depth_frame: depth frame object
    depth_data: numpy 2D array of depth (uint16, mm)
    x,y: pixel coordinates in color image
    """
    try:
        color_frame_video = color_frame.as_video_frame()
        depth_frame_video = depth_frame.as_video_frame()

        depth_width = depth_frame_video.get_width()
        depth_height = depth_frame_video.get_height()

        color_profile = color_frame_video.get_stream_profile()
        depth_profile = depth_frame_video.get_stream_profile()

        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic()
        color_distortion = color_profile.as_video_stream_profile().get_distortion()
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
        depth_distortion = depth_profile.as_video_stream_profile().get_distortion()
        extrinsic_deptocolor = depth_profile.get_extrinsic_to(color_profile)
        extrinsic_colortodep = color_profile.get_extrinsic_to(depth_profile)

        print(f"\n{'='*60}")
        print(f"🔎 指定像素: ({x}, {y})")

        # 使用 SDK 的 transformation2dto2d 将彩色像素映射到深度图坐标
        depth_2d_point = ob.transformation2dto2d(
            ob.OBPoint2f(float(x), float(y)),
            1000,
            color_intrinsics,
            color_distortion,
            depth_intrinsics,
            depth_distortion,
            extrinsic_colortodep
        )

        dx = int(depth_2d_point.x)
        dy = int(depth_2d_point.y)

        print(f"映射到深度图坐标: ({dx}, {dy})")

        if 0 <= dx < depth_width and 0 <= dy < depth_height:
            depth_value = depth_data[dy, dx]
            print(f"深度值: {depth_value} mm ({depth_value/1000.0:.3f} m)")

            if depth_value > 0:
                # SDK 2D->3D（将深度图点转换到彩色相机坐标系）
                point_3d_color_sdk = ob.transformation2dto3d(
                    ob.OBPoint2f(float(dx), float(dy)),
                    depth_value,
                    depth_intrinsics,
                    extrinsic_deptocolor
                )

                print(f"3D (SDK) X={point_3d_color_sdk.x:.2f} Y={point_3d_color_sdk.y:.2f} Z={point_3d_color_sdk.z:.2f} mm")

                # 手动计算（正确方法：使用 SDK 的 Z 值）
                fx = color_intrinsics.fx
                fy = color_intrinsics.fy
                cx = color_intrinsics.cx
                cy = color_intrinsics.cy

                Z_from_sdk = point_3d_color_sdk.z
                X_manual = (x - cx) * Z_from_sdk / fx
                Y_manual = (y - cy) * Z_from_sdk / fy
                Z_manual = Z_from_sdk

                print(f"3D (manual) X={X_manual:.2f} Y={Y_manual:.2f} Z={Z_manual:.2f} mm")
            else:
                print("该点深度为 0（无效）")
        else:
            print("映射坐标超出深度图范围")

        print(f"{'='*60}\n")
    except Exception as e:
        print(f"process_pixel 失败: {e}")
        import traceback
        traceback.print_exc()


def frame_to_bgr(frame):
    """转换彩色帧为 BGR 图像"""
    try:
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        
        if color_format == ob.OBFormat.RGB:
            image = data.reshape((height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == ob.OBFormat.BGR:
            return data.reshape((height, width, 3))
        elif color_format == ob.OBFormat.YUYV:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        elif color_format == ob.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            print(f"不支持的格式: {color_format}")
            return None
    except Exception as e:
        print(f"转换失败: {e}")
        return None


def main():
    global clicked_point, current_color_frame, current_depth_frame, current_color_image

    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, help="指定彩色图像像素 x 坐标")
    parser.add_argument("--y", type=int, help="指定彩色图像像素 y 坐标")
    parser.add_argument("--once", action="store_true", help="如果指定坐标，则获取一次深度并打印后退出")
    args = parser.parse_args()
    
    print("="*60)
    print("🎯 彩色图像点击获取 3D 坐标")
    print("="*60)
    print("📌 使用说明:")
    print("   1. 点击彩色图像中的任意点")
    print("   2. 控制台显示彩色相机坐标系的 3D 坐标")
    print("   3. 按 'q' 或 ESC 退出")
    print("="*60 + "\n")
    
    # 初始化（参考 coordinate_transform.py）
    config = ob.Config()
    pipeline = ob.Pipeline()
    
    try:
        # 启用深度传感器（默认配置）
        depth_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        assert depth_profile_list is not None
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        print(f"✅ ob.OBSensorType.DEPTH_SENSOR 配置: {depth_profile}")
        config.enable_stream(depth_profile)
        
        # 获取深度分辨率
        depth_width = depth_profile.get_width()
        depth_height = depth_profile.get_height()
        
        # 启用彩色传感器 - 选择与深度相同分辨率的配置
        color_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        assert color_profile_list is not None
        
        # 尝试找到与深度分辨率匹配的彩色配置
        color_profile = None
        for i in range(len(color_profile_list)):
            profile = color_profile_list[i]
            if profile.get_width() == depth_width and profile.get_height() == depth_height:
                color_profile = profile
                print(f"✅ 找到匹配分辨率的彩色配置: {profile}")
                break
        
        # 如果没找到匹配的，使用默认配置
        if color_profile is None:
            color_profile = color_profile_list.get_default_video_stream_profile()
            print(f"⚠️  使用默认彩色配置: {color_profile}")
            print(f"   注意：彩色和深度分辨率不一致！")
        
        config.enable_stream(color_profile)
        
    except Exception as e:
        print(f"❌ 配置失败: {e}")
        return
    
    print("\n🚀 启动相机...")
    pipeline.start(config)
    print("✅ 相机已启动\n")
    
    # 创建窗口
    window = "Color Image - Click for 3D coordinates"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 800)  # 设置为图像原始分辨率（1:1 显示）
    cv2.setMouseCallback(window, mouse_callback)
    
    try:
        while True:
            # 等待帧（参考 coordinate_transform.py）
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            
            # 获取深度和彩色帧
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame is None or color_frame is None:
                continue
            
            # 检查深度帧数据完整性（参考 coordinate_transform.py）
            depth_width = depth_frame.get_width()
            depth_height = depth_frame.get_height()
            depth_data_size = depth_frame.get_data_size()
            if depth_data_size != depth_width * depth_height * 2:
                continue
            
            # 保存当前帧供鼠标回调使用
            current_color_frame = color_frame
            current_depth_frame = depth_frame
            
            # 转换彩色图像
            current_color_image = frame_to_bgr(color_frame)
            if current_color_image is None:
                continue
            
            # 在图像上标记点击点
            display = current_color_image.copy()
            
            # 可选：叠加深度热力图（半透明）
            # 这样可以看到哪些区域有有效深度
            try:
                depth_frame_video = current_depth_frame.as_video_frame()
                dw = depth_frame_video.get_width()
                dh = depth_frame_video.get_height()
                depth_data = np.frombuffer(
                    depth_frame_video.get_data(), 
                    dtype=np.uint16
                ).reshape(dh, dw)
                # 如果通过命令行指定坐标，则在获取到第一帧深度后处理并（可选）退出
                if args.x is not None and args.y is not None:
                    process_pixel(color_frame, depth_frame, depth_data, args.x, args.y)
                    if args.once:
                        # 退出主循环，资源将在 finally 中释放
                        raise KeyboardInterrupt
                
                # 创建深度可视化（归一化并应用颜色映射）
                depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # 调整深度图尺寸以匹配彩色图
                if depth_colormap.shape[:2] != display.shape[:2]:
                    depth_colormap = cv2.resize(depth_colormap, 
                                               (display.shape[1], display.shape[0]))
                
                # 半透明叠加（按 'd' 键切换显示）
                # display = cv2.addWeighted(display, 0.7, depth_colormap, 0.3, 0)
            except:
                pass
            
            if clicked_point is not None:
                x, y = clicked_point
                cv2.drawMarker(display, (x, y), (0, 255, 0), 
                             cv2.MARKER_CROSS, 20, 2)
                cv2.circle(display, (x, y), 5, (0, 255, 0), 2)
                cv2.putText(display, f"({x},{y})", (x+10, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(display, "Click for 3D coords | Press Q to exit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n✅ 已退出")


if __name__ == "__main__":
    main()


def color_click_get_once(timeout=30, in_meters=False):
    """
    Programmatic API: open camera and window, wait for one mouse click and return 3D point.

    Args:
        timeout (float): seconds to wait for a click before returning None. None means wait forever.
        in_meters (bool): if True, return coordinates in meters instead of millimeters.

    Returns:
        tuple (x, y, z) floats in mm (or meters if in_meters=True), or None if timeout/no click.
    """
    global clicked_point, current_color_frame, current_depth_frame, current_color_image

    # 清除上次结果
    globals().pop('_color_last_3d_point', None)

    # 初始化 pipeline（和 main 中一致的最小配置）
    config = ob.Config()
    pipeline = ob.Pipeline()
    try:
        # 启用深度传感器
        depth_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        assert depth_profile_list is not None
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        config.enable_stream(depth_profile)
        
        # 获取深度分辨率
        depth_width = depth_profile.get_width()
        depth_height = depth_profile.get_height()
        
        # 启用彩色传感器 - 选择与深度相同分辨率的配置
        color_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        assert color_profile_list is not None
        
        # 尝试找到与深度分辨率匹配的彩色配置
        color_profile = None
        for i in range(len(color_profile_list)):
            profile = color_profile_list[i]
            if profile.get_width() == depth_width and profile.get_height() == depth_height:
                color_profile = profile
                break
        
        # 如果没找到匹配的，使用默认配置
        if color_profile is None:
            color_profile = color_profile_list.get_default_video_stream_profile()
        
        config.enable_stream(color_profile)
    except Exception as e:
        print(f"color_click_get_once: failed to configure streams: {e}")
        return None

    pipeline.start(config)

    window = "Color Image - Click for 3D coordinates"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 800)  # 设置为图像原始分辨率（1:1 显示）
    cv2.setMouseCallback(window, mouse_callback)

    start_ts = time.time()
    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                # check timeout
                if timeout is not None and (time.time() - start_ts) > timeout:
                    break
                continue

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if depth_frame is None or color_frame is None:
                if timeout is not None and (time.time() - start_ts) > timeout:
                    break
                continue

            # 保存当前帧到全局，mouse_callback 使用这些全局变量
            current_color_frame = color_frame
            current_depth_frame = depth_frame

            # 更新显示
            current_color_image = frame_to_bgr(color_frame)
            if current_color_image is None:
                if timeout is not None and (time.time() - start_ts) > timeout:
                    break
                continue

            display = current_color_image.copy()
            if clicked_point is not None:
                x, y = clicked_point
                cv2.drawMarker(display, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(display, (x, y), 5, (0, 255, 0), 2)

            cv2.imshow(window, display)

            # 如果 mouse_callback 已经写入结果，返回
            if '_color_last_3d_point' in globals():
                res = globals().pop('_color_last_3d_point')
                # 转换单位
                if in_meters:
                    res = (res[0] / 1000.0, res[1] / 1000.0, res[2] / 1000.0)
                pipeline.stop()
                cv2.destroyAllWindows()
                return res

            # 超时检查
            if timeout is not None and (time.time() - start_ts) > timeout:
                break

            # allow UI events
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return None

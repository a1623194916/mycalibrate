#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import time
import uuid
import json
import sys
from minio import Minio
import numpy as np
import cv2
from pyorbbecsdk import *
from pyorbbecsdk import save_point_cloud_to_ply

# ============= 全局变量：存储当前实例的相机设备索引 =============
CAMERA_DEVICE_INDEX = 0  # 默认值，会在初始化时被覆盖

def frame_to_bgr_image(frame):
    """将彩色帧转换为BGR图像"""
    try:
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        
        if color_format == OBFormat.RGB:
            image = data.reshape((height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:
            image = data.reshape((height, width, 3))
        elif color_format == OBFormat.YUYV:
            image = data.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        elif color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif color_format == OBFormat.UYVY:
            image = data.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        else:
            print(f"不支持的彩色格式: {color_format}")
            return None
        
        return image
    except Exception as e:
        print(f"彩色图像转换失败: {e}")
        return None

# MinIO配置
minioEndpoint = '10.23.21.2:9000'
minioAsscessKey = 'minioadmin'
minioSecret_key = 'Admin@hd2019'
minioClient = Minio(minioEndpoint,
                    access_key=minioAsscessKey,
                    secret_key=minioSecret_key,
                    secure=False)
bucketName = "camera-picture"

# 保存目录
saveDir = '.' + os.sep + 'data'
IMAGE_COLOR_FILE_PREFIX = "Color"
IMAGE_DEPTH_FILE_PREFIX = "Depth"
POINT_CLOUD_FILE_PREFIX = "PointCloud"

def init_camera_config():
    """初始化相机配置，从启动参数中读取相机设备索引"""
    global CAMERA_DEVICE_INDEX
    
    if len(sys.argv) > 1:
        try:
            config_json = sys.argv[1]
            config = json.loads(config_json)
            
            # 从配置中读取相机设备索引，并确保转换为整数
            camera_index_raw = config.get('camera.device-index', 0)
            
            # 处理字符串转整数的情况
            if isinstance(camera_index_raw, str):
                CAMERA_DEVICE_INDEX = int(camera_index_raw)
            else:
                CAMERA_DEVICE_INDEX = int(camera_index_raw)
            
            print(f"从配置读取相机设备索引: {CAMERA_DEVICE_INDEX} (类型: {type(CAMERA_DEVICE_INDEX)})")
            
            # 验证相机设备是否存在
            validate_camera_device()
            
        except ValueError as e:
            print(f"相机索引转换为整数失败: {e}")
            print(f"原始值: {camera_index_raw}, 类型: {type(camera_index_raw)}")
            CAMERA_DEVICE_INDEX = 0
        except json.JSONDecodeError as e:
            print(f"配置JSON解析失败: {e}")
            CAMERA_DEVICE_INDEX = 0
        except Exception as e:
            print(f"初始化相机配置失败: {e}")
            CAMERA_DEVICE_INDEX = 0
    else:
        print("未提供配置参数，使用默认相机设备索引: 0")
        CAMERA_DEVICE_INDEX = 0

def validate_camera_device():
    """验证指定的相机设备是否存在"""
    global CAMERA_DEVICE_INDEX
    
    try:
        ctx = Context()
        device_list = ctx.query_devices()
        
        if device_list.get_count() == 0:
            raise Exception("未找到任何Orbbec设备!")
        
        print(f"发现设备数量: {device_list.get_count()}")
        
        # 确保两个值都是整数进行比较
        device_count = int(device_list.get_count())
        camera_index = int(CAMERA_DEVICE_INDEX)
        
        if camera_index >= device_count:
            print(f"警告: 指定的设备索引 {camera_index} 超出范围，共有 {device_count} 个设备")
            print("使用默认设备索引: 0")
            CAMERA_DEVICE_INDEX = 0
        
        # 获取设备信息并打印
        target_device = device_list[CAMERA_DEVICE_INDEX]
        device_info = target_device.get_device_info()
        device_name = device_info.get_name()
        device_serial = device_info.get_serial_number()
        
        print(f"当前实例绑定设备: {device_name} (序列号: {device_serial}, 索引: {CAMERA_DEVICE_INDEX})")
        
    except Exception as e:
        print(f"验证相机设备失败: {e}")
        print("使用默认设备索引: 0")
        CAMERA_DEVICE_INDEX = 0
        
def getDevice():
    """获取当前实例绑定的Orbbec设备"""
    global CAMERA_DEVICE_INDEX
    
    try:
        ctx = Context()
        device_list = ctx.query_devices()
        
        if device_list.get_count() == 0:
            raise Exception("未找到任何Orbbec设备!")
        
        if CAMERA_DEVICE_INDEX >= device_list.get_count():
            raise Exception(f"设备索引 {CAMERA_DEVICE_INDEX} 超出范围，共有 {device_list.get_count()} 个设备")
        
        # 使用实例绑定的设备索引
        target_device = device_list[CAMERA_DEVICE_INDEX]
        if target_device is None:
            raise Exception(f"无法获取索引为 {CAMERA_DEVICE_INDEX} 的设备")
        
        pipeline = Pipeline(target_device)
        device = pipeline.get_device()
        
        device_info = device.get_device_info()
        device_name = device_info.get_name()
        device_serial = device_info.get_serial_number()
        print(f"使用设备: {device_name} (序列号: {device_serial}, 索引: {CAMERA_DEVICE_INDEX})")
        
        # 检查设备支持的传感器
        sensor_list = device.get_sensor_list()
        print(f"设备支持的传感器数量: {len(sensor_list)}")
        
        # 创建配置
        config = Config()
        
        # 固定选择深度配置：1280x800@30fps Y16格式
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            raise Exception("未找到深度传感器配置")
        
        depth_profile = None
        for i in range(depth_profile_list.get_count()):
            profile = depth_profile_list[i]
            if profile.is_video_stream_profile():
                video_profile = profile.as_video_stream_profile()
                width = video_profile.get_width()
                height = video_profile.get_height()
                fps = video_profile.get_fps()
                format = video_profile.get_format()
                
                # 固定选择：1280x800@30fps, Y16格式
                if (width == 1280 and height == 800 and fps == 10 and 
                    format == OBFormat.Y16):
                    depth_profile = video_profile
                    break
        
        # 如果没找到Y16格式，尝试RLE格式
        if depth_profile is None:
            for i in range(depth_profile_list.get_count()):
                profile = depth_profile_list[i]
                if profile.is_video_stream_profile():
                    video_profile = profile.as_video_stream_profile()
                    width = video_profile.get_width()
                    height = video_profile.get_height()
                    fps = video_profile.get_fps()
                    format = video_profile.get_format()
                    
                    if (width == 1280 and height == 800 and fps == 30 and 
                        format == OBFormat.RLE):
                        depth_profile = video_profile
                        break
        
        if depth_profile is None:
            raise Exception("未找到指定的深度配置 1280x800@30fps")
        
        print("选择的深度配置:", depth_profile)
        config.enable_stream(depth_profile)
        
        # 固定选择彩色配置：1280x800@30fps RGB格式
        has_color_sensor = False
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                color_profile = None
                for i in range(profile_list.get_count()):
                    profile = profile_list[i]
                    if profile.is_video_stream_profile():
                        video_profile = profile.as_video_stream_profile()
                        width = video_profile.get_width()
                        height = video_profile.get_height()
                        fps = video_profile.get_fps()
                        format = video_profile.get_format()
                        
                        # 固定选择：1280x800@30fps, RGB格式
                        if (width == 1280 and height == 800 and fps == 30 and 
                            format == OBFormat.RGB):
                            color_profile = video_profile
                            break
                
                if color_profile is not None:
                    print("选择的彩色配置:", color_profile)
                    config.enable_stream(color_profile)
                    has_color_sensor = True
                else:
                    print("未找到指定的彩色配置，跳过彩色流")
        except OBError as e:
            print(f"彩色传感器配置失败: {e}")
        
        # 只有当彩色传感器成功配置且分辨率一致时才启用帧同步
        if has_color_sensor:
            pipeline.enable_frame_sync()
            print("已启用帧同步")
        else:
            print("未启用帧同步（仅深度传感器）")
        
        # 启动Pipeline
        pipeline.start(config)
        
        return pipeline, has_color_sensor
    except Exception as e:
        raise Exception(f"获取设备失败: {str(e)}")

def savePointCloudImage_alone():
    """采集点云，使用当前实例绑定的设备"""
    global CAMERA_DEVICE_INDEX
    
    try:
        # 根据实例绑定的设备索引获取指定设备
        ctx = Context()
        device_list = ctx.query_devices()
        
        if device_list.get_count() == 0:
            raise Exception("未找到任何Orbbec设备!")
        
        if CAMERA_DEVICE_INDEX >= device_list.get_count():
            raise Exception(f"设备索引 {CAMERA_DEVICE_INDEX} 超出范围，共有 {device_list.get_count()} 个设备")
        
        target_device = device_list[CAMERA_DEVICE_INDEX] 
        pipeline = Pipeline(target_device)
        
        device_info = target_device.get_device_info()
        print(f"点云采集使用设备: {device_info.get_name()} (索引: {CAMERA_DEVICE_INDEX})")
        
        config = Config()
        
        # 配置深度流 - 使用高分辨率配置
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            raise Exception("未找到深度传感器配置")
            
        # 尝试获取高分辨率深度配置
        depth_profile = None
        for i in range(depth_profile_list.get_count()):
            profile = depth_profile_list[i]
            if profile.is_video_stream_profile():
                video_profile = profile.as_video_stream_profile()
                width = video_profile.get_width()
                height = video_profile.get_height()
                fps = video_profile.get_fps()
                format = video_profile.get_format()
                print(f"点云采集-可用深度配置: {width}x{height} @ {fps}fps, 格式: {format}")
                # 寻找最接近3072x2048的分辨率
                if width >= 1280 and height >= 720:  # 至少要HD分辨率
                    if depth_profile is None or (width > depth_profile.get_width() and height > depth_profile.get_height()):
                        depth_profile = video_profile
        
        # 如果没有找到合适的高分辨率配置，则使用默认配置
        if depth_profile is None:
            depth_profile = depth_profile_list.get_default_video_stream_profile()
        
        print("点云采集-选择的深度配置:", depth_profile)
        config.enable_stream(depth_profile)
        
        # 配置彩色流（如果可用）- 使用高分辨率配置
        has_color_sensor = False
        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                # 尝试获取高分辨率彩色配置
                color_profile = None
                for i in range(profile_list.get_count()):
                    profile = profile_list[i]
                    if profile.is_video_stream_profile():
                        video_profile = profile.as_video_stream_profile()
                        width = video_profile.get_width()
                        height = video_profile.get_height()
                        fps = video_profile.get_fps()
                        format = video_profile.get_format()
                        print(f"点云采集-可用彩色配置: {width}x{height} @ {fps}fps, 格式: {format}")
                        # 寻找最接近3072x2048的分辨率
                        if width >= 1920 and height >= 1080:  # 至少要Full HD分辨率
                            if color_profile is None or (width > color_profile.get_width() and height > color_profile.get_height()):
                                color_profile = video_profile
                                # 如果找到接近3072x2048的分辨率，优先使用
                                if width >= 3000 or height >= 2000:
                                    break
                
                # 如果没有找到合适的高分辨率配置，则使用默认配置
                if color_profile is None:
                    color_profile = profile_list.get_default_video_stream_profile()
                
                print("点云采集-选择的彩色配置:", color_profile)
                config.enable_stream(color_profile)
                has_color_sensor = True
        except OBError as e:
            print(f"彩色传感器配置失败: {e}")
        
        pipeline.enable_frame_sync()
        pipeline.start(config)
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        point_cloud_filter = PointCloudFilter()
        
        # 丢弃前5帧
        for _ in range(5):
            pipeline.wait_for_frames(100)
        
        for _ in range(20):
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue
            color_frame = frames.get_color_frame()
            if has_color_sensor and color_frame is None:
                continue
            frame = align_filter.process(frames)
            point_format = OBFormat.RGB_POINT if has_color_sensor and color_frame is not None else OBFormat.POINT
            point_cloud_filter.set_create_point_format(point_format)
            point_cloud_frame = point_cloud_filter.process(frame)
            if point_cloud_frame is None:
                continue
            now = time.strftime("%Y%m%d%H%M%S")
            # 添加设备索引到文件名，避免冲突
            ply_filename = f"{POINT_CLOUD_FILE_PREFIX}_{CAMERA_DEVICE_INDEX}_{now}.ply"
            ply_local_path = os.path.join(saveDir, ply_filename)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            
            save_point_cloud_to_ply(ply_local_path, point_cloud_frame)
            
            print("点云保存完成")
            point_cloud_url = uploadToMinio(ply_local_path)
            os.remove(ply_local_path)
            pipeline.stop()
            return {"pointCloudUrl": point_cloud_url}
        
        pipeline.stop()
        raise Exception("20次尝试未获取到有效点云帧")
    except Exception as e:
        raise Exception(f"保存点云失败: {str(e)}")

def savedep(pipeline):
    """采集深度图并上传到MinIO"""
    global CAMERA_DEVICE_INDEX
    
    try:
        max_attempts = 10
        for attempt in range(max_attempts):
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                print(f"采集深度帧 attempt {attempt+1}: frames is None")
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                print(f"采集深度帧 attempt {attempt+1}: depth_frame is None")
                continue
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            data_bytes = depth_frame.get_data()
            expected_size = width * height * 2
            if len(data_bytes) != expected_size:
                print(f"采集深度帧 attempt {attempt+1}: 深度帧数据不完整: {len(data_bytes)} != {expected_size}")
                continue
            print("成功采集到深度帧，开始保存")
            now = time.strftime("%Y%m%d%H%M%S")
            # 添加设备索引到文件名，避免冲突
            depth_filename = f"{IMAGE_DEPTH_FILE_PREFIX}Image_{CAMERA_DEVICE_INDEX}_{now}.tiff"
            depth_local_path = os.path.join(saveDir, depth_filename)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(data_bytes, dtype=np.uint16).reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            MIN_DEPTH = 20
            MAX_DEPTH = 10000
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)

            # 图像增强处理（保持原有代码逻辑）
            valid_mask = depth_data > 0
            if np.sum(valid_mask) > 0:
                valid_depth = depth_data[valid_mask]
                noise_estimate = np.std(np.diff(valid_depth.flatten()))
                
                if noise_estimate > 50:
                    depth_filtered = cv2.bilateralFilter(depth_data.astype(np.float32), 3, 30, 30)
                    depth_filtered = depth_filtered.astype(np.uint16)
                else:
                    depth_filtered = depth_data.copy()
            else:
                depth_filtered = depth_data.copy()

            depth_image = cv2.normalize(depth_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            hist = cv2.calcHist([depth_image], [0], None, [256], [0, 256])
            hist_std = np.std(hist)

            if hist_std < 800:
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
                depth_enhanced = clahe.apply(depth_image)
            else:
                depth_enhanced = depth_image.copy()

            depth_image = cv2.applyColorMap(depth_enhanced, cv2.COLORMAP_JET)

            gray_depth = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray_depth, cv2.CV_64F).var()

            if blur_score < 300:
                kernel = np.array([[0, -0.1, 0], 
                                [-0.1, 1.4, -0.1], 
                                [0, -0.1, 0]])
                sharpened = cv2.filter2D(depth_image, -1, kernel)
                depth_image = cv2.addWeighted(depth_image, 0.85, sharpened, 0.15, 0)

            cv2.imwrite(depth_local_path.replace('.png', '.tiff'), depth_image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            depth_url = uploadToMinio(depth_local_path)
            print(f"上传到MinIO返回URL: {depth_url}")
            os.remove(depth_local_path)
            return {"depthurl": depth_url}
        raise Exception("10秒内未获取到有效的深度帧数据")
    except Exception as e:
        print(f"深度图采集失败: {str(e)}")
        raise Exception(f"深度图采集失败: {str(e)}")

def saveclor(pipeline):
    """采集彩色图并上传到MinIO"""
    global CAMERA_DEVICE_INDEX
    
    try:
        max_attempts = 10
        for attempt in range(max_attempts):
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                print(f"采集彩色帧 attempt {attempt+1}: frames is None")
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                print(f"采集彩色帧 attempt {attempt+1}: color_frame is None")
                continue
            color_data = frame_to_bgr_image(color_frame)
            if color_data is None:
                print(f"采集彩色帧 attempt {attempt+1}: color_data is None")
                continue
            now = time.strftime("%Y%m%d%H%M%S")
            # 添加设备索引到文件名，避免冲突
            color_filename = f"{IMAGE_COLOR_FILE_PREFIX}Image_{CAMERA_DEVICE_INDEX}_{now}.tiff"
            color_local_path = os.path.join(saveDir, color_filename)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
                
            # 图像增强处理（保持原有代码逻辑）
            noise_level = cv2.Laplacian(cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if noise_level < 100:
                color_denoised = cv2.fastNlMeansDenoisingColored(color_data, None, 2, 2, 7, 21)
            else:
                color_denoised = color_data.copy()

            gray = cv2.cvtColor(color_denoised, cv2.COLOR_BGR2GRAY)
            hist_std = np.std(cv2.calcHist([gray], [0], None, [256], [0, 256]))

            if hist_std < 1000:
                lab = cv2.cvtColor(color_denoised, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(16, 16))
                l = clahe.apply(l)
                lab = cv2.merge((l, a, b))
                color_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                color_enhanced = color_denoised.copy()

            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 500:
                kernel = np.array([[0, -0.25, 0], 
                                [-0.25, 2, -0.25], 
                                [0, -0.25, 0]])
                sharpened = cv2.filter2D(color_enhanced, -1, kernel)
                color_enhanced = cv2.addWeighted(color_enhanced, 0.8, sharpened, 0.2, 0)

            cv2.imwrite(color_local_path.replace('.png', '.tiff'), color_enhanced, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            color_url = uploadToMinio(color_local_path)
            print(f"上传到MinIO返回URL: {color_url}")
            os.remove(color_local_path)
            return {"colorurl": color_url}
        raise Exception("10秒内未获取到有效的彩色帧数据")
    except Exception as e:
        print(f"彩色图采集失败: {str(e)}")
        raise Exception(f"彩色图采集失败: {str(e)}")

def uploadToMinio(localPath):
    """上传文件到MinIO"""
    try:
        # 获取文件名
        objectName = os.path.basename(localPath)
        # 构建MinIO访问URL
        minioPath = "http://" + minioEndpoint + "/" + bucketName + "/" + objectName
        
        # 读取本地文件并上传
        with open(localPath, "br") as f:
            data = f.read()
            # 上传到MinIO
            minioClient.put_object(bucket_name=bucketName, 
                                 object_name=objectName, 
                                 data=io.BytesIO(data),
                                 length=len(data))
            # 验证上传成功
            minioClient.stat_object(bucket_name=bucketName, object_name=objectName)
        
        return minioPath
    except Exception as e:
        raise Exception(f"上传到MinIO失败: {str(e)}")

# ============= 修改后的API接口（不再需要传入index参数） =============

def photographDepToMinio(data):
    """采集深度图并上传到MinIO - 使用当前实例绑定的设备"""
    global CAMERA_DEVICE_INDEX
    print(f"进入photographDepToMinio接口，使用设备索引: {CAMERA_DEVICE_INDEX}")
    pipeline = None
    try:
        pipeline, has_color_sensor = getDevice()
        print("设备连接成功")
        result = savedep(pipeline)
        return {"msg": "photographDepToMinioSuccess", "data": result}
    except Exception as e:
        print(f"深度图采集失败: {str(e)}")
        raise Exception(f"深度图采集失败: {str(e)}")
    finally:
        if pipeline is not None:
            pipeline.stop()
            time.sleep(1)
            print("设备已关闭")

def photographCloToMinio(data):
    """采集彩色图并上传到MinIO - 使用当前实例绑定的设备"""
    global CAMERA_DEVICE_INDEX
    print(f"进入photographCloToMinio接口，使用设备索引: {CAMERA_DEVICE_INDEX}")
    pipeline = None
    try:
        pipeline, has_color_sensor = getDevice()
        print("设备连接成功")
        if not has_color_sensor:
            raise Exception("设备不支持彩色传感器")
        result = saveclor(pipeline)
        return {"msg": "photographCloToMinioSuccess", "data": result}
    except Exception as e:
        print(f"彩色图采集失败: {str(e)}")
        raise Exception(f"彩色图采集失败: {str(e)}")
    finally:
        if pipeline is not None:
            pipeline.stop()
            time.sleep(1)
            print("设备已关闭")

def photographPointCloud(data):
    """采集点云并保存到MinIO - 使用当前实例绑定的设备"""
    global CAMERA_DEVICE_INDEX
    try:
        print(f"进入photographPointCloud接口，使用设备索引: {CAMERA_DEVICE_INDEX}")
        result = savePointCloudImage_alone()
        print("点云保存到MinIO成功")
        return {"msg": "photographPointCloudSuccess", "data": result}
    except Exception as e:
        print(f"点云采集失败: {str(e)}")
        raise Exception(f"点云采集失败: {str(e)}")

def photographtoMinio(data):
    """同时采集彩色图和深度图并上传到MinIO - 使用当前实例绑定的设备"""
    global CAMERA_DEVICE_INDEX
    print(f"进入photographtoMinio接口，使用设备索引: {CAMERA_DEVICE_INDEX}")
    pipeline = None
    try:
        pipeline, has_color_sensor = getDevice()
        print("设备连接成功")
        
        # 检查是否支持彩色传感器
        if not has_color_sensor:
            print("警告: 设备不支持彩色传感器，仅采集深度图")
        
        result = {}
        
        # 采集深度图
        try:
            depth_result = savedep(pipeline)
            result.update(depth_result)
            print("深度图采集成功")
        except Exception as e:
            print(f"深度图采集失败: {str(e)}")
            result["depthurl"] = None
            result["depth_error"] = str(e)
        
        # 采集彩色图（如果支持）
        if has_color_sensor:
            try:
                color_result = saveclor(pipeline)
                result.update(color_result)
                print("彩色图采集成功")
            except Exception as e:
                print(f"彩色图采集失败: {str(e)}")
                result["colorurl"] = None
                result["color_error"] = str(e)
        else:
            result["colorurl"] = None
            result["color_error"] = "设备不支持彩色传感器"
        
        # 检查是否至少有一个图像采集成功
        success_count = 0
        if result.get("depthurl"):
            success_count += 1
        if result.get("colorurl"):
            success_count += 1
            
        if success_count == 0:
            raise Exception("深度图和彩色图采集均失败")
        
        print(f"图像采集完成，成功采集 {success_count} 种图像")
        return {"msg": "photographtoMinioSuccess", "data": result}
        
    except Exception as e:
        print(f"同时采集图像失败: {str(e)}")
        raise Exception(f"同时采集图像失败: {str(e)}")
    finally:
        if pipeline is not None:
            pipeline.stop()
            time.sleep(1)
            print("设备已关闭")
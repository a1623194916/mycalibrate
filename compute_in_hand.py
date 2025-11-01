# coding=utf-8

"""
眼在手上 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os
import logging

import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from libs.auxiliary import find_latest_data_folder
from libs.log_setting import CommonLog

from save_poses import poses_main

# 控制 numpy 输出精度，方便在日志里阅读矩阵数值
np.set_printoptions(precision=8,suppress=True)

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)


# 数据根目录：默认查找当前脚本下的 eye_hand_data D:\Desktop\fr\hand_eye_calibration\Color
current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"sorted_out")

# 兼容未分批保存的数据：若找不到 dataYYYYMMDD 文件夹，则直接使用根目录
latest_folder = find_latest_data_folder(current_path)
images_path = os.path.join(current_path, latest_folder) if latest_folder else current_path  # 若没有分批文件夹则退回根目录

# 在采集标定板图片时保存的机械臂末端位姿，每一行需与对应的图像文件匹配
file_path = os.path.join(images_path,"robot_poses.txt")  # 对应图片序列的末端位姿记录


with open("config.yaml", 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)  # 读取棋盘参数与其他配置

XX = data.get("checkerboard_args").get("XX")  # 棋盘在 X 方向的角点数量
YY = data.get("checkerboard_args").get("YY")  # 棋盘在 Y 方向的角点数量
L = data.get("checkerboard_args").get("L")    # 棋盘单格尺寸（米）


def func():
    """运行手眼标定流程，返回相机到机械臂末端的旋转矩阵和平移向量。"""

    path = os.path.dirname(__file__)
    logger_.info(f"开始手眼标定流程，数据目录: {images_path}")

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)  # 存储棋盘角点在棋盘坐标系下的三维坐标
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)  # 角点在平面网格中的整数坐标，mgrid生成网格点坐标，转置并重塑为 N x 2 数组，N是角点总数
    objp = L * objp  # 根据单格尺寸 L（米）缩放到真实尺寸

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点

    image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith('.png')])
    logger_.info(f"检测到 {len(image_files)} 张候选图像，默认按 000.png 起连续编号处理")

    for i in range(len(image_files)):   # 遍历期望的文件名顺序（000.png, 001.png ...）

        image_file = os.path.join(images_path, f"{i:03d}.png")

        if os.path.exists(image_file):

            logger_.info(f'读 {image_file}')

            img = cv2.imread(image_file)
            if img is None:
                logger_.warning(f"图像 {image_file} 读取失败")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 手眼标定仅需灰度图

            size = gray.shape[::-1]  # 记录图像尺寸（宽，高），::-1 用于反转元组顺序
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)  # 检测棋盘角点

            if ret:

                obj_points.append(objp)  # 世界坐标系下的角点坐标

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)  # 图像坐标系下的角点坐标
                else:
                    img_points.append(corners)
                logger_.debug(f"图像 {image_file} 角点检测成功")
            else:
                logger_.warning(f"图像 {image_file} 未找到棋盘角点")
        else:
            logger_.warning(f"按序号期望的文件 {image_file} 不存在")

    N = len(img_points)
    logger_.info(f"共有 {N} 张图像用于求解")
    if N == 0:
        raise RuntimeError("角点提取失败，没有可用的标定图像")

    # 标定,得到图案在相机坐标系下的位姿
    # 相机内外参标定：返回内参矩阵、畸变系数、每张图像对应的旋转/平移向量
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    logger_.info(f"相机标定完成，均方重投影误差 {ret:.6f}")
    logger_.debug(f"相机内参:\n{mtx}")
    logger_.debug(f"畸变系数:\n{dist}")

    # logger_.info(f"内参矩阵:\n:{mtx}" ) # 内参数矩阵
    # logger_.info(f"畸变系数:\n:{dist}")  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("-----------------------------------------------------")  # 分隔相机标定与后续处理的输出

    poses_main(file_path)  # 将 poses.txt 中的欧拉角转换为旋转向量
    logger_.info("已根据 poses.txt 生成 RobotToolPose.csv")
    # 机器人末端在基座标系下的位姿

    csv_file = os.path.join(path,"RobotToolPose.csv")
    tool_pose = np.loadtxt(csv_file,delimiter=',')  # 每列块为一个 4x4 齐次矩阵（列堆叠）
    logger_.debug(f"RobotToolPose.csv 形状 {tool_pose.shape}")

    R_tool = []
    t_tool = []

    for i in range(int(N)):

        # 位姿矩阵按照列堆叠存储：前三列为旋转，第四列为平移
        R_tool.append(tool_pose[0:3,4*i:4*i+3])
        t_tool.append(tool_pose[0:3,4*i+3])
    logger_.debug(f"第 {i + 1} 帧末端旋转矩阵行列式 {np.linalg.det(R_tool[-1]):.6f}")  # 行列式用于快速检查正交性

    # 使用Tsai方法计算手眼关系：求解末端坐标系到相机坐标系的外参
    R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    logger_.info("手眼标定完成，获得末端->相机外参")

    return R,t

if __name__ == '__main__':

    # 旋转矩阵，外参
    rotation_matrix, translation_vector = func()

    # 将旋转矩阵转换为四元数
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    x, y, z = translation_vector.flatten()

    logger_.info(f"旋转矩阵是:\n {            rotation_matrix}")  # 相机相对末端的旋转

    logger_.info(f"平移向量是:\n {            translation_vector}")  # 相机相对末端的平移

    logger_.info(f"四元数是：\n {             quaternion}")  # 旋转的四元数表示，便于某些控制系统使用



import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------- 常量：手眼标定得到的变换 ----------------------------
# rotation_matrix: 3x3 矩阵，将相机坐标系（camera）旋转到机械臂末端（end-effector）坐标系
# 注意：此处的矩阵是手眼标定得到的旋转矩阵，表示 R_camera_to_ee
# 更新时间: 2026-01-26 (使用 robottrue.txt 实际位姿标定)
rotation_matrix = np.array([[ 0.98108533, -0.19353332,  0.00405374],
                            [ 0.19341089,  0.98089886,  0.02072788],
                            [-0.00798784, -0.01955178,  0.99977694]])

# translation_vector: 3x1 向量，将相机坐标系原点平移到末端的坐标
# 形式上表示 t_camera_to_ee，使得：p_ee = R_camera_to_ee * p_cam + t_camera_to_ee
# 注意单位：该向量以米为单位，因此使用时需要保证输入坐标单位一致
translation_vector = np.array([-0.00703272, -0.07832151, -0.22621523])



def convert(x, y, z, x1, y1, z1, rx, ry, rz):
    """
    将深度相机在相机坐标系中检测到的物体点 (x, y, z)
    转换为机械臂基座（base）坐标系下的坐标。

    输入参数：
      - x, y, z: 物体在相机坐标系下的坐标（注意单位应与 translation_vector 一致，通常为米）
      - x1, y1, z1: 机械臂末端（end-effector）在基座（base）坐标系下的位置（平移，单位同上）
      - rx, ry, rz: 机械臂末端的姿态，用欧拉角表示（顺序与下面 from_euler 中的顺序对应），单位为弧度

    输出：
      - 返回物体在机械臂基座坐标系下的 [x, y, z] 列表

    假设和说明：
      - rotation_matrix 和 translation_vector 表示的是相机到末端的变换：
          p_ee = R_camera_to_ee @ p_cam + t_camera_to_ee
      - 函数中将先把相机坐标转换到末端坐标系，再把末端坐标转换到基座坐标系
      - 使用齐次坐标进行矩阵相乘以方便变换拼接
    """

    # ---------- 1) 把输入的相机坐标点打包为 numpy 向量 ----------
    # obj_camera_coordinates: (3,) 向量，表示点在相机坐标系中的 (x,y,z)
    obj_camera_coordinates = np.array([x, y, z])

    # ---------- 2) 把末端位姿（基座下）打包为向量 ----------
    # end_effector_pose 中前 3 个元素是位置 (x1,y1,z1)，后 3 个是欧拉角 (rx,ry,rz)
    # 这里约定欧拉角单位为弧度（函数注释也说明了）
    end_effector_pose = np.array([x1, y1, z1,
                                  rx, ry, rz])

    # ---------- 3) 构造相机坐标系到末端坐标系的齐次变换矩阵 ----------
    # T_camera_to_end_effector 是 4x4 的齐次矩阵，格式为 [[R, t],[0,1]]
    T_camera_to_end_effector = np.eye(4)  # 初始化为单位矩阵
    T_camera_to_end_effector[:3, :3] = rotation_matrix  # 填入旋转部分 (3x3)
    T_camera_to_end_effector[:3, 3] = translation_vector  # 填入平移部分 (3,)

    # 备注：上述矩阵的含义是对齐为 p_ee_homo = T_camera_to_end_effector @ p_cam_homo

    # ---------- 4) 将末端在基座坐标系下的位姿转为齐次变换矩阵 ----------
    # 位置部分
    position = end_effector_pose[:3]

    # 使用 scipy 的 Rotation.from_euler 将给定的欧拉角转换为 3x3 旋转矩阵
    # 这里使用 xyz，小写是外旋！！！！大写内旋
    # 与传入参数的角度顺序必须一致，否则会产生错误的姿态。
    orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()  # 3x3 旋转矩阵

    # T_base_to_end_effector: 4x4 矩阵，表示基座到末端的变换
    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation  # 旋转部分
    T_base_to_end_effector[:3, 3] = position  # 平移部分

    # ---------- 5) 用齐次坐标进行坐标变换：camera -> end-effector -> base ----------
    # 将相机坐标的三维点扩展为齐次坐标 (x, y, z, 1)
    obj_camera_coordinates_homo = np.append(
        obj_camera_coordinates, [1])  # 形状 (4,)

    # 先把相机坐标系下的点变换到末端坐标系下（p_ee = T_cam_to_ee * p_cam）
    obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(
        obj_camera_coordinates_homo)

    # 再把末端坐标系下的点变换到基座坐标系下（p_base = T_base_to_ee * p_ee）
    obj_base_coordinates_homo = T_base_to_end_effector.dot(
        obj_end_effector_coordinates_homo)

    # 提取齐次结果的前三个分量作为 x,y,z
    obj_base_coordinates = list(
        obj_base_coordinates_homo[:3])  # 返回 [x, y, z]

    return obj_base_coordinates

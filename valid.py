import os
import cv2
import numpy as np
from compute_in_hand import func, XX, YY, L, images_path, file_path
from save_poses import poses_main

R_cam2ee, t_cam2ee = func()
T_cam2ee = np.eye(4)
T_cam2ee[:3, :3] = R_cam2ee
T_cam2ee[:3, 3] = t_cam2ee.flatten()

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
objp = np.zeros((XX * YY, 3), np.float32)
objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
objp *= L

obj_points, img_points, size = [], [], None

image_files = sorted(
    [f for f in os.listdir(images_path) if f.lower().endswith('.png')]
)

for image_name in image_files:
    image_file = os.path.join(images_path, image_name)
    img = cv2.imread(image_file)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
    if not ret:
        continue
    obj_points.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    img_points.append(corners2)

if not obj_points:
    raise RuntimeError("valid: 未检测到任何棋盘角点，无法评估手眼标定。")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

poses_main(file_path)
robot_pose = np.loadtxt(os.path.join(os.path.dirname(__file__), "RobotToolPose.csv"), delimiter=',')

Ts_base2ee = []
for i in range(robot_pose.shape[1] // 4):
    T = np.eye(4)
    T[:3, :3] = robot_pose[:3, 4*i:4*i+3]
    T[:3, 3] = robot_pose[:3, 4*i+3]
    Ts_base2ee.append(T)

Ts_cam2board = []
for rvec, tvec in zip(rvecs, tvecs):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.flatten()
    Ts_cam2board.append(T)

rot_errs, trans_errs = [], []
for i in range(len(Ts_base2ee) - 1):
    A = np.linalg.inv(Ts_base2ee[i+1]) @ Ts_base2ee[i]
    B = Ts_cam2board[i+1] @ np.linalg.inv(Ts_cam2board[i])
    left = A @ T_cam2ee
    right = T_cam2ee @ B
    R_diff = left[:3, :3] @ right[:3, :3].T
    angle = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    trans = np.linalg.norm(left[:3, 3] - right[:3, 3])
    rot_errs.append(angle)
    trans_errs.append(trans)

print(f"平均旋转误差: {np.mean(rot_errs):.4f}°")
print(f"平均平移误差: {np.mean(trans_errs):.6f} m")
print(f"R·R^T 偏差: {np.linalg.norm(R_cam2ee @ R_cam2ee.T - np.eye(3)):.2e}")
print(f"det(R): {np.linalg.det(R_cam2ee):.6f}")
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# ========================================
# 配置路径 - 使用之前标定时的数据
# ========================================
# 标定时的图片目录
IMAGES_PATH = "/home/u22/kyz/mycalibrate/calib_images"
# 标定时的机器人位姿文件
ROBOT_POSES_PATH = "/home/u22/kyz/mycalibrate/calib_images/robottrue.txt"

# 棋盘格参数 - 从 config.yaml 读取以保持一致
import yaml
with open("config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
XX = config["checkerboard_args"]["XX"]  # X方向角点数
YY = config["checkerboard_args"]["YY"]  # Y方向角点数
L = config["checkerboard_args"]["L"]    # 单格尺寸（米）
print(f"📐 棋盘格参数: {XX}x{YY}, 单格 {L*1000:.1f}mm")

# ========================================
# 直接使用已有的手眼标定外参（不重新计算）
# ========================================
USE_EXISTING_CALIBRATION = False  # True: 使用下面的外参, False: 重新计算

if USE_EXISTING_CALIBRATION:
    # 用户提供的外参矩阵 (2026-01-20 标定结果)
    R_cam2ee = np.array([[0.98108533, -0.19353332,  0.00405374],
                         [0.19341089,  0.98089886,  0.02072788],
                         [-0.00798784, -0.01955178,  0.99977694]])
    t_cam2ee = np.array([[-0.00703272],
                         [-0.07832151],
                         [-0.23121523]])
    print("📌 使用已保存的手眼标定外参")
else:
    from compute_in_hand import func
    R_cam2ee, t_cam2ee = func()
    print("🔄 重新计算手眼标定外参")

T_cam2ee = np.eye(4)
T_cam2ee[:3, :3] = R_cam2ee
T_cam2ee[:3, 3] = t_cam2ee.flatten()

print("\n" + "="*60)
print("手眼标定外参 (相机 -> 末端)")
print("="*60)
print(f"旋转矩阵:\n{R_cam2ee}")
print(f"平移向量: {t_cam2ee.flatten()}")
print(f"det(R): {np.linalg.det(R_cam2ee):.6f} (应为 1.0)")
print(
    f"R·R^T 偏差: {np.linalg.norm(R_cam2ee @ R_cam2ee.T - np.eye(3)):.2e} (应为 0)")
print("="*60 + "\n")

# ========================================
# 1. 检测棋盘格角点 + 相机标定
# ========================================
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
objp = np.zeros((XX * YY, 3), np.float32)
objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
objp *= L

obj_points, img_points = [], []
valid_indices = []  # 记录成功检测到棋盘的图片索引
size = None

image_files = sorted(
    [f for f in os.listdir(IMAGES_PATH) if f.lower().endswith('.png')]
)

print(f"📂 扫描图片目录: {IMAGES_PATH}")
print(f"   找到 {len(image_files)} 张图片")

for idx, image_name in enumerate(image_files):
    image_file = os.path.join(IMAGES_PATH, image_name)
    img = cv2.imread(image_file)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
    if not ret:
        print(f"   ⚠️ {image_name}: 未检测到棋盘")
        continue
    obj_points.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    img_points.append(corners2)
    valid_indices.append(idx)

print(f"\n✅ 成功检测 {len(obj_points)}/{len(image_files)} 张图片的棋盘角点")

if not obj_points:
    raise RuntimeError("valid: 未检测到任何棋盘角点，无法评估手眼标定。")

# 相机标定获取内参
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, size, None, None)
print(f"\n📷 相机内参标定完成 (RMS误差: {ret:.4f})")
print(
    f"   fx={mtx[0,0]:.2f}, fy={mtx[1,1]:.2f}, cx={mtx[0,2]:.2f}, cy={mtx[1,2]:.2f}")

# ========================================
# 2. 读取机器人位姿
# ========================================
robot_poses_raw = np.loadtxt(ROBOT_POSES_PATH)
print(f"\n🤖 读取机器人位姿: {robot_poses_raw.shape[0]} 条")


def pose_to_matrix(pose):
    """将 [x, y, z, rx, ry, rz] 转换为 4x4 齐次变换矩阵
    
    注意: rx, ry, rz 是欧拉角（弧度）
    fairmove.get_current_pose() 返回的是 [x,y,z,rx,ry,rz] 毫米+度
    automain.py 中用 mmdeg_to_mrad() 转换为 米+弧度(欧拉角)
    
    使用 xyz 内旋顺序（小写表示内旋）
    """
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', pose[3:6]).as_matrix()
    T[:3, 3] = pose[0:3]
    return T


# 只取有效图片对应的位姿
Ts_base2ee = []
for idx in valid_indices:
    if idx < len(robot_poses_raw):
        Ts_base2ee.append(pose_to_matrix(robot_poses_raw[idx]))

# 相机到棋盘格的变换
Ts_cam2board = []
for rvec, tvec in zip(rvecs, tvecs):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.flatten()
    Ts_cam2board.append(T)

# ========================================
# 3. AX=XB 一致性检验
# ========================================

rot_errs, trans_errs = [], []
for i in range(min(len(Ts_base2ee), len(Ts_cam2board)) - 1):
    A = np.linalg.inv(Ts_base2ee[i+1]) @ Ts_base2ee[i]
    B = Ts_cam2board[i+1] @ np.linalg.inv(Ts_cam2board[i])
    left = A @ T_cam2ee
    right = T_cam2ee @ B
    R_diff = left[:3, :3] @ right[:3, :3].T
    trace_val = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    angle = np.degrees(np.arccos(trace_val))
    trans = np.linalg.norm(left[:3, 3] - right[:3, 3])
    rot_errs.append(angle)
    trans_errs.append(trans)

print("\n" + "="*60)
print("手眼标定误差评估 (AX=XB 一致性检验)")
print("="*60)
print(f"📊 共评估 {len(rot_errs)} 对相邻帧")
print(f"📐 旋转误差:")
print(f"   - 平均: {np.mean(rot_errs):.4f}°")
print(f"   - 最大: {np.max(rot_errs):.4f}°")
print(f"   - 最小: {np.min(rot_errs):.4f}°")
print(f"   - 标准差: {np.std(rot_errs):.4f}°")
print(f"📏 平移误差:")
print(f"   - 平均: {np.mean(trans_errs)*1000:.2f} mm")
print(f"   - 最大: {np.max(trans_errs)*1000:.2f} mm")
print(f"   - 最小: {np.min(trans_errs)*1000:.2f} mm")
print(f"   - 标准差: {np.std(trans_errs)*1000:.2f} mm")
print("="*60)

# ========================================
# 评估标准
# ========================================
print("\n📋 误差评估标准:")
if np.mean(rot_errs) < 1.0 and np.mean(trans_errs)*1000 < 5:
    print("   ✅ 优秀: 旋转<1°, 平移<5mm")
elif np.mean(rot_errs) < 2.0 and np.mean(trans_errs)*1000 < 10:
    print("   ⚠️ 良好: 旋转<2°, 平移<10mm")
elif np.mean(rot_errs) < 5.0 and np.mean(trans_errs)*1000 < 20:
    print("   ⚠️ 一般: 旋转<5°, 平移<20mm (建议重新标定)")
else:
    print("   ❌ 较差: 误差过大，强烈建议重新标定！")

# 逐帧误差详情
print("\n📝 逐帧误差详情 (前15帧):")
for i, (r, t) in enumerate(zip(rot_errs[:15], trans_errs[:15])):
    status = "✓" if r < 2.0 and t*1000 < 10 else "⚠"
    print(
        f"   帧 {valid_indices[i]:02d}->{valid_indices[i+1]:02d}: 旋转 {r:.3f}°, 平移 {t*1000:.2f}mm {status}")

# ========================================
# 4. 重投影误差 (更直观)
# ========================================
print("\n" + "="*60)
print("🎯 方法2: 重投影误差 (最直观的指标)")
print("="*60)
print("将棋盘格3D点投影到图像，与检测的2D角点对比")

reproj_errors = []
reproj_errors_per_image = []

for i in range(len(obj_points)):
    # 投影到图像
    proj_pts, _ = cv2.projectPoints(
        objp, rvecs[i], tvecs[i], mtx, dist
    )
    proj_pts = proj_pts.reshape(-1, 2)
    detected_pts = img_points[i].reshape(-1, 2)

    # 计算每个点的误差
    errors = np.linalg.norm(proj_pts - detected_pts, axis=1)
    reproj_errors.extend(errors)
    reproj_errors_per_image.append(np.mean(errors))

reproj_errors = np.array(reproj_errors)
print(f"\n📊 重投影误差统计 ({len(reproj_errors)} 个角点):")
print(f"   - 平均: {np.mean(reproj_errors):.3f} 像素")
print(f"   - 最大: {np.max(reproj_errors):.3f} 像素")
print(f"   - 中位数: {np.median(reproj_errors):.3f} 像素")
print(f"   - 标准差: {np.std(reproj_errors):.3f} 像素")

# ========================================
# 5. 综合评估
# ========================================
print("\n" + "="*60)
print("📋 综合评估结论")
print("="*60)

avg_rot = np.mean(rot_errs)
avg_trans = np.mean(trans_errs) * 1000  # mm
avg_reproj = np.mean(reproj_errors)

print(f"\n指标汇总:")
print(f"  • AX=XB 旋转误差: {avg_rot:.3f}°")
print(f"  • AX=XB 平移误差: {avg_trans:.2f} mm")
print(f"  • 重投影误差: {avg_reproj:.3f} 像素")

# 评估标准
print(f"\n诊断结果:")
if avg_rot < 1.0 and avg_trans < 5 and avg_reproj < 1.0:
    print("  ✅ 优秀: 手眼标定精度很高，相机位置正常")
elif avg_rot < 2.0 and avg_trans < 10 and avg_reproj < 2.0:
    print("  ⚠️ 良好: 精度可接受，但建议观察")
elif avg_rot < 5.0 and avg_trans < 20 and avg_reproj < 5.0:
    print("  ⚠️ 一般: 精度较低，相机可能有轻微偏移")
    print("       建议: 重新采集数据进行标定")
else:
    print("  ❌ 较差: 误差过大！相机很可能已经移位")
    print("       强烈建议: 立即重新进行手眼标定！")

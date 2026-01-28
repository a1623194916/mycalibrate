cd /home/u22/kyz/mycalibrate && python3 -c "
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

# 点击像素: (783, 47), 深度: 321.00 mm
u, v, depth = 662, 193, 330.0

# 旧内参
fx_old, fy_old, cx_old, cy_old = 604.25993192, 604.03556638, 643.75378798, 363.27535391
x_old = (u - cx_old) * depth / fx_old
y_old = (v - cy_old) * depth / fy_old
print(f'旧内参相机坐标 (mm): ({x_old:.2f}, {y_old:.2f}, {depth:.2f})')

# 新内参 (手眼标定时)
fx_new, fy_new, cx_new, cy_new = 601.85552584, 601.10594863, 641.45224297, 401.85953898
x_new = (u - cx_new) * depth / fx_new
y_new = (v - cy_new) * depth / fy_new
print(f'新内参相机坐标 (mm): ({x_new:.2f}, {y_new:.2f}, {depth:.2f})')

# 用新内参的相机坐标做坐标变换
rotation_matrix = np.array([[ 0.98108533, -0.19353332,  0.00405374],
                            [ 0.19341089,  0.98089886,  0.02072788],
                            [-0.00798784, -0.01955178,  0.99977694]])
translation_vector = np.array([-0.00703272, -0.07832151, -0.22621523])

tcp = [21.01093864440918, -163.8605499267578, 300.0216979980469, -171.43031311035156, 0.1314552128314972, -129.777099609375]
x1, y1, z1 = tcp[0]/1000, tcp[1]/1000, tcp[2]/1000
rx, ry, rz = math.radians(tcp[3]), math.radians(tcp[4]), math.radians(tcp[5])

T_cam2ee = np.eye(4)
T_cam2ee[:3, :3] = rotation_matrix
T_cam2ee[:3, 3] = translation_vector

T_base2ee = np.eye(4)
T_base2ee[:3, :3] = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
T_base2ee[:3, 3] = [x1, y1, z1]

# 新内参的相机坐标
p_cam_new = np.array([x_new/1000, y_new/1000, depth/1000, 1])
p_ee_new = T_cam2ee @ p_cam_new
p_base_new = T_base2ee @ p_ee_new

print(f'\\n新内参基座坐标 (mm): ({p_base_new[0]*1000:.2f}, {p_base_new[1]*1000:.2f}, {p_base_new[2]*1000:.2f})')
print(f'真值 (mm): (154.3, -303.34, 221.6)')
print(f'误差 (mm): X={154.3-p_base_new[0]*1000:.2f}, Y={-303.34-p_base_new[1]*1000:.2f}, Z={221.6-p_base_new[2]*1000:.2f}')
"
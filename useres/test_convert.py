#!/usr/bin/env python3
"""直接测试convert函数，不连接机器人"""

import math
from convert import convert

# 从点击得到的相机坐标 (mm)
cam_x, cam_y, cam_z = -47.11, -85.01, 344.00

# 拍照时的TCP位姿 (从机器人教学器读取)
tcp_pose = [-30.599445343017578, -351.2419738769531, 248.12408447265625, -
            172.76284790039062, 2.9248266220092773, -163.400634765625]

# 真实基座坐标 (从机器人教学器读取，用于对比)
real_base = [56.769, -514.785, 152.854]

print("="*60)
print("手眼标定验证测试")
print("="*60)
print(f"相机坐标 (mm): ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})")
print(f"TCP位姿 (mm,deg): {tcp_pose}")
print(
    f"真实基座坐标 (mm): ({real_base[0]:.2f}, {real_base[1]:.2f}, {real_base[2]:.2f})")
print("-"*60)

# 转换单位：mm -> m, deg -> rad
x, y, z = cam_x/1000.0, cam_y/1000.0, cam_z/1000.0
x1, y1, z1, rx, ry, rz = tcp_pose
x1, y1, z1 = x1/1000.0, y1/1000.0, z1/1000.0
rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)

# 调用convert
final_x, final_y, final_z = convert(x, y, z, x1, y1, z1, rx, ry, rz)

# 转回mm
final_x, final_y, final_z = final_x*1000.0, final_y*1000.0, final_z*1000.0

print(f"计算基座坐标 (mm): ({final_x:.2f}, {final_y:.2f}, {final_z:.2f})")
print("-"*60)

# 计算误差
error_x = abs(final_x - real_base[0])
error_y = abs(final_y - real_base[1])
error_z = abs(final_z - real_base[2])
error_total = (error_x**2 + error_y**2 + error_z**2)**0.5

print(f"误差 (mm):")
print(f"  ΔX = {error_x:.2f}")
print(f"  ΔY = {error_y:.2f}")
print(f"  ΔZ = {error_z:.2f}")
print(f"  总误差 = {error_total:.2f}")
print("="*60)

if error_total < 10:
    print("✓ 标定精度良好 (<10mm)")
elif error_total < 30:
    print("⚠ 标定精度一般 (10-30mm)")
else:
    print("✗ 标定精度差 (>30mm)，需要重新标定")

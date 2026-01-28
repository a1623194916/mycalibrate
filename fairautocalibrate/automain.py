#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动手眼标定 - 使用 CuRobo 规划轨迹
"""

import math
import os
import time
import sys
import numpy as np

# 添加 curobo/frplan 路径（使用其中的 fairmove.py 和 real_robot_plan.py）
sys.path.insert(0, "/home/u22/kyz/curobo/frplan")


def mrad_to_mmdeg(pose_mrad):
	"""[x,y,z,rx,ry,rz] from meters+radians -> millimeters+degrees."""
	x_m, y_m, z_m, rx_r, ry_r, rz_r = pose_mrad
	return [
		x_m * 1000.0,
		y_m * 1000.0,
		z_m * 1000.0,
		rx_r * 180.0 / math.pi,
		ry_r * 180.0 / math.pi,
		rz_r * 180.0 / math.pi,
	]


def mmdeg_to_mrad(pose_mmdeg):
	"""[x,y,z,rx,ry,rz] from millimeters+degrees -> meters+radians."""
	x_mm, y_mm, z_mm, rx_d, ry_d, rz_d = pose_mmdeg
	return [
		x_mm / 1000.0,
		y_mm / 1000.0,
		z_mm / 1000.0,
		rx_d * math.pi / 180.0,
		ry_d * math.pi / 180.0,
		rz_d * math.pi / 180.0,
	]


def read_poses_txt(txt_path):
	poses = []
	with open(txt_path, "r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			stripped = line.strip()
			if not stripped or stripped.startswith("#"):
				continue
			parts = stripped.split()
			if len(parts) != 6:
				raise ValueError(f"{txt_path}:{line_no} 每行需要 6 个数，实际 {len(parts)} 个: {stripped}")
			poses.append([float(x) for x in parts])
	return poses


def recv_image_response(socket):
	"""接收来自相机/边缘盒子的图像应答（与 zmqclient.py 协议兼容）。"""
	import cv2
	import numpy as np

	md = socket.recv_json()
	msg = socket.recv(copy=False)

	if md.get("status") != "ok":
		print(f"[相机错误] {md.get('msg', 'Unknown error')}")
		return None

	if md.get("compressed", False):
		buf = np.frombuffer(msg, dtype=np.uint8)
		array = cv2.imdecode(buf, cv2.IMREAD_COLOR)
	else:
		buf = memoryview(msg)
		array = np.frombuffer(buf, dtype=md["dtype"]).reshape(md["shape"])
	return array


def pose_close(actual_mmdeg, target_mmdeg, pos_tol_mm=2.0, ang_tol_deg=2.0):
	if actual_mmdeg is None:
		return False
	dp = max(abs(actual_mmdeg[i] - target_mmdeg[i]) for i in range(3))
	da = max(abs(actual_mmdeg[i] - target_mmdeg[i]) for i in range(3, 6))
	return (dp <= pos_tol_mm) and (da <= ang_tol_deg)


def wait_until_reach(robot_controller, target_mmdeg, timeout_s=8.0, poll_dt=0.2, pos_tol_mm=2.0, ang_tol_deg=2.0):
    # 等待机械臂到达目标位姿，通过比较当前位姿和目标位姿
	deadline = time.time() + timeout_s
	while time.time() < deadline:
		actual = robot_controller.get_current_pose()
		if pose_close(actual, target_mmdeg, pos_tol_mm=pos_tol_mm, ang_tol_deg=ang_tol_deg):
			return True
		time.sleep(poll_dt)
	return False


def build_camera_socket(context, mode, addr, port):
	import zmq

	socket = context.socket(zmq.REQ)
	endpoint = f"tcp://{addr}:{port}"
	if mode == "bind":
		# 与现有 zmqclient.py 一致：主机 REQ + bind，边缘盒子连接过来
		socket.bind(endpoint)
		print(f"[相机] REQ bind: {endpoint}")
	else:
		socket.connect(endpoint)
		print(f"[相机] REQ connect: {endpoint}")
	return socket


def main():
	# ====== 直接在这里改配置（无需传参） ======
	POSES_FILE = os.getenv("POSES_FILE", "robot_poses_filtered.txt")  # 使用过滤后的位姿文件
	SAVE_DIR = os.getenv("SAVE_DIR", "calib_images")
	START_INDEX = int(os.getenv("START_INDEX", "0"))
	MAX_POINTS = int(os.getenv("MAX_POINTS", "0"))  # 0 表示全部
	SLEEP_AFTER_MOVE = float(os.getenv("SLEEP_AFTER_MOVE", "0.5"))  # 秒，等待稳定

	REACH_CHECK = os.getenv("REACH_CHECK", "1") != "0"  # 1/0
	REACH_TIMEOUT_S = float(os.getenv("REACH_TIMEOUT", "15.0"))  # CuRobo 轨迹较长
	POS_TOL_MM = float(os.getenv("POS_TOL_MM", "2.0"))
	ANG_TOL_DEG = float(os.getenv("ANG_TOL_DEG", "2.0"))

	CAM_MODE = os.getenv("CAM_MODE", "bind")  # bind 或 connect
	CAM_ADDR = os.getenv("CAM_ADDR", "*")  # bind 用 *，connect 用对端 IP
	CAM_PORT = os.getenv("CAM_PORT", "5555")
	CAM_CMD = os.getenv("CAM_CMD", "GET_RGB")
	
	# CuRobo 轨迹执行参数
	TRAJ_VEL = float(os.getenv("TRAJ_VEL", "30.0"))  # 速度百分比
	TRAJ_BLEND = float(os.getenv("TRAJ_BLEND", "100"))  # 平滑时间 ms
	# ========================================

	poses_path = os.path.abspath(POSES_FILE)
	save_dir = os.path.abspath(SAVE_DIR)
	os.makedirs(save_dir, exist_ok=True)
	print(f"[标定] poses: {poses_path}")
	print(f"[标定] save_dir: {save_dir}")

	poses_mrad = read_poses_txt(poses_path)
	if MAX_POINTS and MAX_POINTS > 0:
		poses_mrad = poses_mrad[:MAX_POINTS]
	if not poses_mrad:
		print("[标定] poses 为空，退出")
		return 2

	# 相机
	try:
		import zmq
	except Exception as e:
		print(f"[标定] 缺少依赖: pyzmq，错误: {e}")
		print("[提示] 可尝试: pip install pyzmq")
		return 10

	try:
		import cv2  # noqa: F401
	except Exception as e:
		print(f"[标定] 缺少依赖: opencv-python，错误: {e}")
		print("[提示] 可尝试: pip install opencv-python")
		return 11

	try:
		import numpy  # noqa: F401
	except Exception as e:
		print(f"[标定] 缺少依赖: numpy，错误: {e}")
		print("[提示] 可尝试: pip install numpy")
		return 12

	context = zmq.Context()
	cam_socket = build_camera_socket(context, CAM_MODE, CAM_ADDR, CAM_PORT)

	# 机械臂（从 curobo/frplan 导入，包含 followcurobo 方法）
	try:
		from fairmove import RobotController
	except Exception as e:
		print(f"[标定] 导入 RobotController 失败: {e}")
		print("[提示] 这通常是 Robot.so 与当前 Python 版本 ABI 不匹配；请用与 Robot.so 编译一致的 python 运行。")
		return 3

	# 初始化 CuRobo 规划器
	try:
		from real_robot_plan import initialize_curobo
		print("\n[CuRobo] 初始化运动规划器...")
		planner = initialize_curobo()
		print("[CuRobo] 规划器初始化完成\n")
	except Exception as e:
		print(f"[标定] 导入/初始化 CuRobo 失败: {e}")
		import traceback
		traceback.print_exc()
		return 5

	robot = RobotController()
	if not robot.connect():
		print("[标定] 机械臂连接失败")
		return 4

	# 临时轨迹文件路径
	traj_file = os.path.join(os.path.dirname(__file__), "calib_traj.txt")
	
	# 实际位姿记录文件（米+弧度）
	robottrue_file = os.path.join(save_dir, "robottrue.txt")
	# 清空或创建文件
	with open(robottrue_file, 'w') as f:
		f.write("# 实际TCP位姿 (米+弧度): x y z rx ry rz\n")
	print(f"[标定] 位姿记录: {robottrue_file}")

	try:
		img_index = 0
		for i, pose_mrad in enumerate(poses_mrad):
			target_mmdeg = mrad_to_mmdeg(pose_mrad)
			print(f"\n{'='*60}")
			print(f"[标定] 点 {i}/{len(poses_mrad)-1}")
			print(f"  目标位姿 (mm/deg): {[f'{v:.2f}' for v in target_mmdeg]}")
			print(f"{'='*60}")

			# 获取当前关节角度作为规划起点
			current_q = robot.GetActualJointPos(flag=1)  # 弧度
			if current_q is None:
				print("[标定] 获取当前关节角度失败，跳过该点")
				continue

			# 使用 CuRobo 规划轨迹（输入 TCP 位姿，米+弧度）
			print(f"[CuRobo] 规划中...")
			success, traj_deg, solve_time, traj_path = planner.plan(
				current_q,          # 起始关节角度（弧度）
				pose_mrad,          # 目标 TCP 位姿（米+弧度）
				save_trajectory=True,
				visualize=False,
				traj_filename=traj_file
			)

			if not success:
				print(f"[标定] CuRobo 规划失败，跳过该点")
				continue

			print(f"[CuRobo] 规划成功，轨迹点数: {len(traj_deg)}，耗时: {solve_time:.3f}s")

			# 执行轨迹
			print(f"[轨迹] 执行中...")
			robot.followcurobo(traj_path, vel=TRAJ_VEL, blendT=TRAJ_BLEND)

			# 等待稳定
			if SLEEP_AFTER_MOVE > 0:
				time.sleep(SLEEP_AFTER_MOVE)

			# 检查是否到位
			if REACH_CHECK:
				ok = wait_until_reach(
					robot,
					target_mmdeg,
					timeout_s=REACH_TIMEOUT_S,
					pos_tol_mm=POS_TOL_MM,
					ang_tol_deg=ANG_TOL_DEG,
				)
				if not ok:
					actual = robot.get_current_pose()
					print(f"[标定] 未在超时内到位")
					print(f"  目标: {[f'{v:.2f}' for v in target_mmdeg]}")
					print(f"  实际: {[f'{v:.2f}' for v in actual] if actual else 'None'}")
					print(f"[标定] 跳过拍照")
					continue

			# 拍照
			print(f"[相机] 请求拍照: {CAM_CMD}")
			t0 = time.time()
			cam_socket.send_string(CAM_CMD)
			img = recv_image_response(cam_socket)
			cost_ms = (time.time() - t0) * 1000.0

			if img is None:
				print("[标定] 拍照失败（无有效图像）")
				continue

			# 记录当前实际TCP位姿（米+弧度）
			actual_pose_mmdeg = robot.get_current_pose()
			if actual_pose_mmdeg is not None:
				actual_pose_mrad = mmdeg_to_mrad(actual_pose_mmdeg)
				with open(robottrue_file, 'a') as f:
					f.write(' '.join([str(v) for v in actual_pose_mrad]) + '\n')
				print(f"[标定] 已记录实际位姿: [{actual_pose_mrad[0]:.4f}, {actual_pose_mrad[1]:.4f}, {actual_pose_mrad[2]:.4f}] m")
			else:
				print("[标定] 警告: 无法获取当前位姿")

			out_path = os.path.join(save_dir, f"{img_index:03d}.png")
			import cv2
			ok = cv2.imwrite(out_path, img)
			if not ok:
				print(f"[标定] 保存失败: {out_path}")
				continue
			print(f"[标定] ✓ 已保存: {out_path} (shape={img.shape}, {cost_ms:.1f}ms)")
			img_index += 1

		print(f"\n{'='*60}")
		print(f"[标定] 完成！共保存 {img_index} 张图像")
		print(f"{'='*60}")

	finally:
		# 清理临时轨迹文件
		if os.path.exists(traj_file):
			try:
				os.remove(traj_file)
			except Exception:
				pass
		try:
			cam_socket.close(0)
		except Exception:
			pass
		try:
			context.term()
		except Exception:
			pass
		try:
			robot.disconnect()
		except Exception:
			pass

	return 0


if __name__ == "__main__":
	raise SystemExit(main())


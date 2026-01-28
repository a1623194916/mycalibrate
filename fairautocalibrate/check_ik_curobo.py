#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 CuRobo 检查手眼标定位姿的 IK 可解性
直接复用 real_robot_plan.py 的 initialize_curobo 函数
fr3c.yml 中已经定义了工具偏移，直接用 TCP 位姿规划即可
"""

import numpy as np
import math
import os
import sys

# 添加 curobo/frplan 路径
sys.path.insert(0, "/home/u22/kyz/curobo/frplan")

# 直接导入 real_robot_plan 中的函数
from real_robot_plan import initialize_curobo, parse_pose_with_unit_detection


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


def read_poses_txt(txt_path):
    """读取位姿文件 (米+弧度)"""
    poses = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 6:
                raise ValueError(f"{txt_path}:{line_no} 每行需要 6 个数，实际 {len(parts)} 个")
            poses.append([float(x) for x in parts])
    return poses


class CuroboIKChecker:
    """使用 CuRobo 检查 IK 可解性，直接复用 real_robot_plan.py 的 planner"""
    
    def __init__(self):
        print("=" * 60)
        print("🚀 初始化 CuRobo (复用 real_robot_plan)")
        print("=" * 60)
        
        # 直接使用 real_robot_plan.py 的初始化函数
        self.planner = initialize_curobo()
        
        # 使用 fr3c.yml 中的 retract_config 作为起始状态（有效的关节角度）
        # retract_config: [-1.21,-0.95,-0.76,-2.3,1.65,0.0006]
        self.default_q = np.array([-1.21, -0.95, -0.76, -2.3, 1.65, 0.0006])
        print("✅ CuRobo IK Checker 就绪!\n")
    
    def check_single_pose(self, pose_mrad):
        """
        检查单个位姿是否有 IK 解
        直接使用 planner.plan() 方法，fr3c.yml 已包含工具偏移
        
        Args:
            pose_mrad: [x, y, z, rx, ry, rz] 米+弧度 (TCP 位姿)
            
        Returns:
            (has_solution, joint_angles): 
                - has_solution: bool
                - joint_angles: 关节角度(弧度) 或 None
        """
        try:
            # 直接使用 planner.plan()，输入是 TCP 位姿
            # save_trajectory=False, visualize=False 只检查可解性
            success, positions_deg, solve_time, _ = self.planner.plan(
                self.default_q,  # 起始角度（弧度）
                pose_mrad,       # 目标 TCP 位姿（米+弧度）
                save_trajectory=False,
                visualize=False
            )
            
            if success and positions_deg is not None:
                # 返回最终关节角度（转换为弧度）
                joint_angles = np.deg2rad(positions_deg[-1])
                return True, joint_angles
            else:
                return False, None
        except Exception as e:
            print(f"   ⚠️ 求解异常: {e}")
            return False, None
    
    def check_all_poses(self, poses_mrad, verbose=True):
        """
        批量检查所有位姿
        
        Args:
            poses_mrad: 位姿列表 [[x,y,z,rx,ry,rz], ...]
            verbose: 是否打印详细信息
            
        Returns:
            results: [(index, success, joint_angles, pose_mmdeg), ...]
        """
        results = []
        success_count = 0
        failed_indices = []
        
        print(f"\n📋 检查 {len(poses_mrad)} 个位姿的 IK 可解性...")
        print("-" * 60)
        
        for i, pose_mrad in enumerate(poses_mrad):
            pose_mmdeg = mrad_to_mmdeg(pose_mrad)
            success, joint_angles = self.check_single_pose(pose_mrad)
            
            results.append((i, success, joint_angles, pose_mmdeg))
            
            if success:
                success_count += 1
                if verbose:
                    print(f"✅ [{i:3d}] 有解 | 位姿: [{pose_mmdeg[0]:.1f}, {pose_mmdeg[1]:.1f}, {pose_mmdeg[2]:.1f}] mm")
            else:
                failed_indices.append(i)
                if verbose:
                    print(f"❌ [{i:3d}] 无解 | 位姿: [{pose_mmdeg[0]:.1f}, {pose_mmdeg[1]:.1f}, {pose_mmdeg[2]:.1f}] mm")
        
        print("-" * 60)
        print(f"\n📊 统计结果:")
        print(f"   总数: {len(poses_mrad)}")
        print(f"   有解: {success_count} ({100*success_count/len(poses_mrad):.1f}%)")
        print(f"   无解: {len(failed_indices)} ({100*len(failed_indices)/len(poses_mrad):.1f}%)")
        
        if failed_indices:
            print(f"\n❌ 无解索引: {failed_indices}")
        else:
            print(f"\n🎉 所有位姿均可求解!")
        
        return results, failed_indices
    
    def filter_solvable_poses(self, poses_mrad):
        """
        过滤出可解的位姿
        
        Returns:
            solvable_poses: 可解的位姿列表
            solvable_indices: 可解位姿的原始索引
        """
        results, failed_indices = self.check_all_poses(poses_mrad, verbose=True)
        
        solvable_poses = []
        solvable_indices = []
        
        for i, success, joint_angles, pose_mmdeg in results:
            if success:
                solvable_poses.append(poses_mrad[i])
                solvable_indices.append(i)
        
        return solvable_poses, solvable_indices


def save_filtered_poses(poses_mrad, output_path):
    """保存过滤后的位姿到文件"""
    with open(output_path, 'w') as f:
        for pose in poses_mrad:
            f.write(' '.join([str(x) for x in pose]) + '\n')
    print(f"💾 已保存 {len(poses_mrad)} 个可解位姿到: {output_path}")


def main():
    # 配置
    poses_path = "/home/u22/kyz/mycalibrate/fairautocalibrate/robot_poses.txt"
    output_path = "/home/u22/kyz/mycalibrate/fairautocalibrate/robot_poses_filtered.txt"
    
    # 读取位姿
    poses_mrad = read_poses_txt(poses_path)
    if not poses_mrad:
        print(f"❌ 位姿文件为空: {poses_path}")
        return 1
    
    print(f"📂 读取位姿文件: {poses_path}")
    print(f"   共 {len(poses_mrad)} 个位姿\n")
    
    # 初始化 CuRobo IK 检查器（直接复用 real_robot_plan.py 的配置）
    checker = CuroboIKChecker()
    
    # 检查所有位姿
    results, failed_indices = checker.check_all_poses(poses_mrad)
    
    # 过滤并保存可解位姿
    if failed_indices:
        solvable_poses, solvable_indices = checker.filter_solvable_poses(poses_mrad)
        save_filtered_poses(solvable_poses, output_path)
        
        print(f"\n💡 建议: 使用过滤后的位姿文件进行标定")
        print(f"   原始: {poses_path} ({len(poses_mrad)} 个)")
        print(f"   过滤: {output_path} ({len(solvable_poses)} 个)")
    else:
        print(f"\n✅ 所有位姿均可解，无需过滤")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

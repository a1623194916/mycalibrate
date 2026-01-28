import Robot
from config import MoveConfig
import time
import traceback
import os
import math


class RobotController:
    """机械臂控制器封装类"""

    def __init__(self, robot_ip=None):
        """
        初始化机械臂控制器

        参数:
            robot_ip: 机器人 IP 地址,默认使用配置值
        """
        self.robot_ip = robot_ip or MoveConfig.ROBOT_IP
        self.robot = None
        self.is_connected = False

    def connect(self):
        """
        连接机械臂（连接成功后自动设置工具坐标系）

        返回:
            success: bool, 是否成功连接
        """
        try:
            print(f"[移动] 连接机械臂: {self.robot_ip}")
            # 取消注释以启用实际连接
            self.robot = Robot.RPC(self.robot_ip)
            self.is_connected = True
            print("[移动] 机械臂连接成功")

            # 连接成功后立即设置工具坐标系
            if not self.set_tool_coord():
                print("[移动] 警告: 工具坐标系设置失败，但连接已建立")

            return True

        except Exception as e:
            print(f"[移动] 连接失败: {e}")
            return False

    def disconnect(self):
        """断开机械臂连接"""
        if self.robot:
            self.robot.CloseRPC()
            time.sleep(1)
            self.robot = None
            self.is_connected = False
            print("[移动] 机械臂已断开")

    def get_current_pose(self):
        """
        获取当前 TCP 位姿

        返回:
            pose: [x, y, z, rx, ry, rz] (毫米, 度)
        """
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None

        try:
            error, tcp = self.robot.GetActualTCPPose()
            if error != 0:
                print(f"[移动] 获取 TCP 位姿失败: {error}")
                return None
            return list(tcp)
        except Exception as e:
            print(f"[移动] 获取 TCP 位姿异常: {e}")
            return None

    def is_connection_alive(self):
        """
        检查机器人连接是否仍然有效

        返回:
            alive: bool, 连接是否有效
        """
        if not self.is_connected or self.robot is None:
            return False

        try:
            # 使用轻量级接口测试连接
            err, _ = self.robot.GetActualTCPPose()
            if err == 0:
                return True
            else:
                print(f"[移动] 连接检查失败，错误码: {err}")
                self.is_connected = False
                return False
        except Exception as e:
            print(f"[移动] 连接检查异常: {e}")
            self.is_connected = False
            return False

    def ensure_connected(self):
        """
        确保机器人连接有效，如果无效则重新连接

        返回:
            success: bool, 是否连接成功
        """
        if self.is_connection_alive():
            return True

        print("[移动] 检测到连接失效，尝试重新连接...")
        self.robot = None
        self.is_connected = False
        return self.connect()

    def set_tool_coord(self, tool_coord=None, retry_count=3, debug=None):
        """
        设置工具坐标系（带重试机制）

        参数:
            tool_coord: 工具坐标 [x, y, z, rx, ry, rz],默认使用配置值
            retry_count: 重试次数，默认3次

        返回:
            success: bool
        """
        if tool_coord is None:
            tool_coord = MoveConfig.TOOL_COORD
        if debug is None:
            debug = os.getenv("FAIR_DEBUG_STACK", "0") == "1"

        print(f"[移动] 使用的工具坐标系: {tool_coord}")
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return False

        for attempt in range(retry_count):
            try:
                if attempt > 0:
                    print(f"[移动] 设置工具坐标系重试 {attempt}/{retry_count-1}...")
                else:
                    print(f"[移动] 设置工具坐标系: {tool_coord}")
                    if debug:
                        print(
                            f"[移动][debug] tool_coord(module={MoveConfig.__module__}, id={id(tool_coord)})")
                        traceback.print_stack(limit=6)

                err = self.robot.SetToolCoord(2, t_coord=tool_coord, type=0,
                                              install=0, toolID=1, loadNum=1)
                if err != 0:
                    print(f"[移动] 设置工具坐标系失败，错误码: {err}")

                    if debug:
                        traceback.print_stack(limit=6)

                    # 打印更多诊断信息
                    try:
                        err_tcp, tcp_num = self.robot.GetActualTCPNum(1)
                        print(
                            f"[诊断] GetActualTCPNum: err={err_tcp}, toolNum={tcp_num}")
                    except:
                        pass

                    if attempt < retry_count - 1:
                        import time
                        time.sleep(0.5)  # 等待0.5秒后重试
                        continue
                    return False

                print("[移动] 工具坐标系设置成功")
                return True
            except Exception as e:
                print(f"[移动] 设置工具坐标系异常: {e}")
                if debug:
                    traceback.print_exc()
                if attempt < retry_count - 1:
                    import time
                    time.sleep(0.5)
                    continue
                return False

        return False

    def move_to_pose(self, pose, tool=1, user=0, vel=None, acc=None, ovl=None, blendT=-1.0, config=-1):
        pass

    def MoveJ(
        self, joint_pos, tool=1, user=0, desc_pos=None, vel=20.0, acc=0.0, ovl=100.0, exaxis_pos=None, blendT=-1.0, offset_flag=0,
        offset_pos=None,
    ):
        """
        关节空间运动（支持自动单位检测）

        必选参数:
            joint_pos: 目标关节位置，单位[°]或[rad]（自动检测）
                      - 所有关节角度绝对值 ≤ 2π (≈6.28): 视为弧度，自动转为角度
                      - 否则: 视为角度，直接使用
            tool: 工具号，[0~14]
            user: 工件号，[0~14]

        默认参数:
            desc_pos: 目标笛卡尔位姿，单位[mm][°]，默认 None（由控制器正解）
            vel: 速度百分比，[0~100] 默认 20.0
            acc: 加速度百分比，[0~100] 默认 0.0
            ovl: 速度缩放因子，[0~100] 默认 100.0
            exaxis_pos: 外部轴 1~4 位置，默认 [0.0,0.0,0.0,0.0]
            blendT: [-1.0]-到位阻塞，[0~500.0]-平滑时间(非阻塞)，单位[ms]
            offset_flag: [0]-不偏移，[1]-工件/基坐标系偏移，[2]-工具坐标系偏移
            offset_pos: 位姿偏移量，单位[mm][°]

        返回值:
            错误码 成功-0 失败-errcode
        """
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None

        if desc_pos is None:
            desc_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if exaxis_pos is None:
            exaxis_pos = [0.0, 0.0, 0.0, 0.0]
        if offset_pos is None:
            offset_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 自动检测单位：如果所有角度绝对值都 <= 2π，视为弧度
        max_abs_value = max(abs(v) for v in joint_pos)
        TWO_PI = 2.0 * math.pi  # ≈ 6.28

        if max_abs_value <= TWO_PI:
            # 视为弧度，转换为角度
            try:
                joint_pos_deg = [math.degrees(v) for v in joint_pos]
                print(f"[MoveJ] 检测到弧度输入，已转换为角度")
            except Exception as e:
                print(f"[移动] 关节角度转换失败: {e}")
                return None
        else:
            # 视为角度，直接使用
            joint_pos_deg = list(joint_pos)

        try:
            ret = self.robot.MoveJ(
                joint_pos_deg,
                tool,
                user,
                desc_pos=desc_pos,
                vel=vel,
                acc=acc,
                ovl=ovl,
                exaxis_pos=exaxis_pos,
                blendT=blendT,
                offset_flag=offset_flag,
                offset_pos=offset_pos,
            )
            if ret != 0:
                print(f"[移动] 关节空间运动失败: {ret}")
            return ret
        except Exception as e:
            print(f"[移动] 关节空间运动异常: {e}")
            return None

    def MoveL(self, desc_pos, tool=1, user=0, joint_pos=None, vel=20.0, acc=0.0, ovl=100.0,
              blendR=-1.0, blendMode=0, exaxis_pos=None, search=0, offset_flag=0,
              offset_pos=None, overSpeedStrategy=0, speedPercent=10, in_meters=False):
        """
        笛卡尔空间直线运动

        必选参数:
            desc_pos: 目标笛卡尔位姿，单位[mm][°]
            tool: 工具号，[0~14]
            user: 工件号，[0~14]

        默认参数:
            joint_pos: 目标关节位置，单位[°]，默认调用逆运动学求解
            vel: 速度百分比，[0~100] 默认 20.0
            acc: 加速度百分比，[0~100] 默认 0.0
            ovl: 速度缩放因子，[0~100] 默认 100.0
            blendR: [-1.0]-运动到位(阻塞)，[0~1000]-平滑半径(非阻塞)，单位[mm]
            blendMode: 过渡方式；0-内切过渡；1-角点过渡
            exaxis_pos: 外部轴1~4位置
            search: [0]-不焊丝寻位，[1]-焊丝寻位
            offset_flag: [0]-不偏移，[1]-工件/基坐标系偏移，[2]-工具坐标系偏移
            offset_pos: 位姿偏移量，单位[mm][°]
            overSpeedStrategy: 超速处理策略，0-关闭；1-标准；2-报错停止；3-自适应降速
            speedPercent: 允许降速阈值百分比[0-100]
            in_meters: desc_pos 是否为米单位（自动转换为毫米）

        返回值:
            错误码 成功-0 失败-errcode
        """
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None

        # 设置默认值
        if joint_pos is None:
            joint_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if exaxis_pos is None:
            exaxis_pos = [0.0, 0.0, 0.0, 0.0]
        if offset_pos is None:
            offset_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 单位转换：米转毫米
        if in_meters:
            desc_pos_mm = [
                desc_pos[0] * 1000.0,  # x
                desc_pos[1] * 1000.0,  # y
                desc_pos[2] * 1000.0,  # z
                desc_pos[3],  # rx
                desc_pos[4],  # ry
                desc_pos[5]   # rz
            ]
        else:
            desc_pos_mm = list(desc_pos)
            
        # 如果Z小于0，则强制为0，防止撞地
        if desc_pos_mm[2] < 0.0:
            print(f"[移动] 警告: 目标Z位置小于0，已强制设为0以防撞地")
            desc_pos_mm[2] = 0.0

        try:
            ret = self.robot.MoveL(
                desc_pos_mm, tool, user,
                joint_pos=joint_pos,
                vel=vel,
                acc=acc,
                ovl=ovl,
                blendR=blendR,
                blendMode=blendMode,
                exaxis_pos=exaxis_pos,
                search=search,
                offset_flag=offset_flag,
                offset_pos=offset_pos,
                overSpeedStrategy=overSpeedStrategy,
                speedPercent=speedPercent
            )
            if ret != 0:
                print(f"[移动] 笛卡尔直线运动失败: {ret}")
            return ret
        except Exception as e:
            print(f"[移动] 笛卡尔直线运动异常: {e}")
            return None

    def GetActualJointPos(self, flag):
        """
        获取当前关节角度,flag=0，返回角度，flag=1，返回弧度

        返回:
            joints: [j1, j2, j3, j4, j5, j6] (弧度)
        """
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None

        try:
            if flag:
                error, joints = self.robot.GetActualJointPosRadian(1)
                if error != 0:
                    print(f"[移动] 获取关节弧度失败: {error}")
                    return None
                return list(joints)
            else:
                error, joints = self.robot.GetActualJointPosDegree(1)
                if error != 0:
                    print(f"[移动] 获取关节角度失败: {error}")
                    return None
                return list(joints)
        except Exception as e:
            print(f"[移动] 获取关节角度异常: {e}")
            return None

    def GetInverseKin(self, type=0, desc_pos=None, config=-1):
        """
        逆运动学，笛卡尔位姿求解关节位置（支持自动单位检测）

        必选参数:
            type: 0-绝对位姿(基坐标系)，1-相对位姿（基坐标系），2-相对位姿（工具坐标系）
            desc_pos: [x,y,z,rx,ry,rz]，工具位姿，支持自动单位检测:
                     - 位置: 绝对值>10视为毫米，否则视为米(自动转毫米)
                     - 旋转: 绝对值>2π(6.28)视为度，否则视为弧度(自动转度)

        默认参数:
            config: 关节配置，[-1]-参考当前关节位置求解，[0~7]-依据关节配置求解 默认-1

        返回值:
            joint_pos=[j1,j2,j3,j4,j5,j6]：关节角度(度)
            失败返回 None
        """
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None

        if desc_pos is None or len(desc_pos) < 6:
            print("[移动] 位姿参数无效")
            return None

        # 自动单位检测和转换
        pos = desc_pos[:3]
        rot = desc_pos[3:6]
        
        # 位置单位检测：>10 视为毫米，否则视为米
        max_pos = max(abs(pos[0]), abs(pos[1]), abs(pos[2]))
        if max_pos <= 10.0:
            # 视为米，转换为毫米
            pos_mm = [pos[0] * 1000.0, pos[1] * 1000.0, pos[2] * 1000.0]
            print(f"[逆解] 检测到米单位，已转换为毫米: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]m -> [{pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}]mm")
        else:
            # 视为毫米，直接使用
            pos_mm = list(pos)
        
        # 旋转单位检测：>2π 视为度，否则视为弧度
        max_rot = max(abs(rot[0]), abs(rot[1]), abs(rot[2]))
        TWO_PI = 2.0 * math.pi  # ≈ 6.28
        
        if max_rot <= TWO_PI:
            # 视为弧度，转换为度
            rot_deg = [math.degrees(rot[0]), math.degrees(rot[1]), math.degrees(rot[2])]
            print(f"[逆解] 检测到弧度单位，已转换为度: [{rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}]rad -> [{rot_deg[0]:.1f}, {rot_deg[1]:.1f}, {rot_deg[2]:.1f}]°")
        else:
            # 视为度，直接使用
            rot_deg = list(rot)
        
        # 组合为SDK需要的格式（毫米+度）
        desc_pos_mm_deg = pos_mm + rot_deg

        try:
            error, joint_pos = self.robot.GetInverseKin(type, desc_pos_mm_deg, config)
            if error != 0:
                print(f"[移动] 逆运动学求解失败: {error}")
                return None
            print(f"[逆解] 求解成功: {[f'{j:.2f}°' for j in joint_pos]}")
            return list(joint_pos)
        except Exception as e:
            print(f"[移动] 逆运动学求解异常: {e}")
            return None

    def wait_arrival(self, target_pose, error_threshold=5.0, timeout=10.0):
        """
        阻塞等待机器人到达目标点
        :param target_pose: 目标 [x, y, z, ...]
        :param error_threshold: 允许误差 (mm)
        :param timeout: 超时时间 (秒)
        """
        import time
        import math

        start_time = time.time()
        print("[等待] 正在等待机器人到位...")

        while time.time() - start_time < timeout:
            # 获取当前位置
            curr = self.get_current_pose()
            if not curr:
                continue

            # 计算当前点和目标点的空间距离
            # 简化计算：只算 XYZ 距离
            dist = math.sqrt((curr[0]-target_pose[0])**2 +
                             (curr[1]-target_pose[1])**2 +
                             (curr[2]-target_pose[2])**2)

            if dist < error_threshold:
                print(f"[到位] 误差 {dist:.2f} mm，动作完成")
                return True

            time.sleep(0.1)  # 别刷太快，给 CPU 喘口气

        print("[超时] 机器人未在规定时间内到位")
        return False

    def moveGripper(self, index=1, pos=100, vel=20, force=0, maxtime=1000, block=0, type=0, rotNum=0, rotVel=0, rotTorque=0):
        """
        夹爪运动（支持自动单位检测，归一化到SDK的0-100范围）

        法奥夹爪规格: 最大张开90mm (9cm), 深度95mm, 高度15mm, 厚度10mm

        参数:
            index: 夹爪编号，默认 1
            pos: 位置值，支持三种输入方式（自动识别）:
                 - <= 1.0: 视为米单位 (0.0~0.09m)，归一化到 SDK 的 0-100
                 - > 1.0 且 <= 100: 视为 SDK 位置值 (0~100)，直接使用
                 - > 100: 视为毫米 (0~90mm)，归一化到 SDK 的 0-100
            vel: 速度百分比 [0~100]
            force: 力矩百分比 [0~100]
            block: [0]-非阻塞，[1]-阻塞到位

        """
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None

        # 自动检测单位并归一化到 0-100
        MAX_WIDTH_MM = 90.0  # 法奥夹爪最大张开 90mm

        if pos <= 1.0:
            # 视为米单位，转换为毫米再归一化
            width_mm = pos * 1000.0
            width_mm = max(0.0, min(width_mm, MAX_WIDTH_MM))
            gripper_pos = int((width_mm / MAX_WIDTH_MM) * 100)
            gripper_pos = max(0, min(100, gripper_pos))
            print(f"[夹爪] 宽度 {width_mm:.1f}mm -> SDK位置值 {gripper_pos}")
        elif pos > 100:
            # 视为毫米，归一化到 0-100
            width_mm = max(0.0, min(pos, MAX_WIDTH_MM))
            gripper_pos = int((width_mm / MAX_WIDTH_MM) * 100)
            gripper_pos = max(0, min(100, gripper_pos))
            print(f"[夹爪] 宽度 {width_mm:.1f}mm -> SDK位置值 {gripper_pos}")
        else:
            # 视为 SDK 位置值 (0-100)，直接使用
            gripper_pos = int(pos)
            gripper_pos = max(0, min(100, gripper_pos))
            width_mm = (gripper_pos / 100.0) * MAX_WIDTH_MM
            print(f"[夹爪] SDK位置值 {gripper_pos} (约 {width_mm:.1f}mm)")

        try:
            error = self.robot.MoveGripper(
                index, gripper_pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque)
            if error != 0:
                print(f"[移动] 夹爪运动失败: {error}")
                return None
            return error
        except Exception as e:
            print(f"[移动] 夹爪异常: {e}")
            return None

    def LoadTrajectoryJ(self, name, ovl=20, opt=1):
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None
        try:
            name = "/usr/local/etc/controller/lua/traj/" + name
            print(f"[移动] 加载轨迹文件: {name}")
            error = self.robot.LoadTrajectoryJ(name, ovl, opt)
            if error != 0:
                print(f"[移动] 加载轨迹失败: {error}")
                return None
            return error
        except Exception as e:
            print(f"[移动] 加载轨迹异常: {e}")
            return None

    def GetTrajectoryStartPose(self, name):
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None
        try:
            name = "/usr/local/etc/controller/lua/traj/" + name
            print(f"[移动] 获取轨迹起始位姿文件: {name}")
            error, pose = self.robot.GetTrajectoryStartPose(name)
            if error != 0:
                print(f"[移动] 获取轨迹起始位姿失败: {error}")
                return None
            return list(pose)
        except Exception as e:
            print(f"[移动] 获取轨迹起始位姿异常: {e}")
            return None

    def GetTrajectoryPointNum(self):
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None
        try:
            error, pointnum = self.robot.GetTrajectoryPointNum()
            if error != 0:
                print(f"[移动] 获取轨迹点数失败: {error}")
                return None
            return pointnum
        except Exception as e:
            print(f"[移动] 获取轨迹点数异常: {e}")
            return None

    def SetTrajectoryJSpeed(self, ovl):
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None
        try:
            print(f"[移动] 设置轨迹速度百分比: {ovl}")
            error = self.robot.SetTrajectoryJSpeed(ovl)
            if error != 0:
                print(f"[移动] 设置轨迹速度失败: {error}")
                return None
            return error
        except Exception as e:
            print(f"[移动] 设置轨迹速度异常: {e}")
            return None

    def MoveTrajectoryJ(self):
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None
        try:
            print(f"[移动] 执行轨迹运动")
            error = self.robot.MoveTrajectoryJ()
            if error != 0:
                print(f"[移动] 执行轨迹运动失败: {error}")
                return None
            return error
        except Exception as e:
            print(f"[移动] 执行轨迹运动异常: {e}")
            return None

    def followcurobo(self, filename="trajectory_deg.txt", vel=20.0, blendT=-1, adaptive_blend=True):
        """
        读取轨迹文件并逐行执行 MoveJ 该函数用于跟随 curobo 生成的轨迹

        Args:
            filename: 轨迹文件名
            vel: 速度百分比 [0~100]
            blendT: 平滑时间 (ms)
                - -1: 阻塞运动（到位）- 会卡顿但精确
                - 0~500: 平滑过渡时间 - 流畅但会有路径偏差
            adaptive_blend: 是否使用自适应平滑（首尾点阻塞，中间点平滑）
        """
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None

        traj_path = os.path.join(os.path.dirname(__file__), filename)

        # 读取轨迹文件并逐行执行 MoveJ
        traj_points = []
        if not os.path.exists(traj_path):
            print(f"[轨迹] 文件不存在: {traj_path}")
            return None

        try:
            with open(traj_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        print(f"[轨迹] 行格式不足6列，已跳过: {line}")
                        continue
                    try:
                        traj_points.append([float(v) for v in parts[:6]])
                    except ValueError as e:
                        print(f"[轨迹] 行解析失败，已跳过: {line} ({e})")
        except Exception as e:
            print(f"[轨迹] 读取文件异常: {e}")
            return None

        if not traj_points:
            print(f"[轨迹] 文件为空或解析失败: {traj_path}")
            return None

        total_points = len(traj_points)
        print(f"[轨迹] 开始跟随，共 {total_points} 个点")

        # 自适应平滑策略
        if adaptive_blend and blendT > 0:
            print(f"[轨迹] 使用自适应平滑: 首尾点阻塞，中间点 blendT={blendT}ms")
        elif blendT > 0:
            print(f"[轨迹] 使用固定平滑时间: blendT={blendT}ms")
        else:
            print(f"[轨迹] 阻塞模式（每点到位）")

        for idx, joints in enumerate(traj_points, 1):
            # 自适应平滑：首尾点精确到位，中间点平滑过渡
            if adaptive_blend:
                if idx == 1 or idx == total_points:
                    # 第一个和最后一个点必须精确到位
                    current_blend = -1.0
                    print(f"[轨迹] 点 {idx}/{total_points}: 阻塞到位")
                else:
                    current_blend = blendT
            else:
                current_blend = blendT

            ret = self.MoveJ(joint_pos=joints, vel=vel, blendT=current_blend)
            if ret != 0:
                print(f"[轨迹] MoveJ 失败，错误码 {ret}，停止执行")
                break

            # 非阻塞模式需要短暂延时避免命令堆积
            if current_blend >= 0:
                time.sleep(0.01)

        print("[轨迹] 跟随结束")

    def followcurobo_servo(self, filename="trajectory_deg.txt", cmdT=0.01, speed_scale=1.0, safe_mode=True):
        """
        使用ServoJ进行轨迹跟踪（推荐方案）

        ServoJ的优势：
        - 专为轨迹跟踪设计，运动更流畅
        - 高频率指令下发，无卡顿
        - 适合密集轨迹点（curobo生成的轨迹）

        Args:
            filename: 轨迹文件名
            cmdT: 指令下发周期 (s)，默认0.01秒（与curobo的interpolation_dt=0.01匹配）
                 范围 [0.001~0.016]
            speed_scale: 速度缩放因子 [0.1~2.0]，通过调整cmdT来控制速度
                        - <1.0: 减速（增大cmdT）
                        - =1.0: 原始速度（cmdT=0.01）
                        - >1.0: 加速（减小cmdT，但要注意机器人响应能力）
            safe_mode: 是否启用安全模式（自动检测并过滤重复点）

        速度控制说明：
        - curobo轨迹点间隔为0.01秒（interpolation_dt=0.01）
        - ServoJ的vel/acc参数暂不开放，通过speed_scale调整cmdT来控制速度
        - 例如：speed_scale=0.5 -> cmdT从0.01变为0.02（速度减半）

        注意：
        - ServoJ运动期间不能执行其他运动指令
        - speed_scale过大（cmdT过小）可能导致速度超限
        - 建议speed_scale范围 [0.3~1.5]
        """
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None

        traj_path = os.path.join(os.path.dirname(__file__), filename)

        # 读取轨迹文件
        traj_points = []
        if not os.path.exists(traj_path):
            print(f"[轨迹] 文件不存在: {traj_path}")
            return None

        try:
            with open(traj_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        print(f"[轨迹] 行格式不足6列，已跳过: {line}")
                        continue
                    try:
                        traj_points.append([float(v) for v in parts[:6]])
                    except ValueError as e:
                        print(f"[轨迹] 行解析失败，已跳过: {line} ({e})")
        except Exception as e:
            print(f"[轨迹] 读取文件异常: {e}")
            return None

        if not traj_points:
            print(f"[轨迹] 文件为空或解析失败: {traj_path}")
            return None

        # 安全模式：过滤重复点和检查速度
        if safe_mode:
            filtered_points = [traj_points[0]]  # 保留第一个点
            duplicate_count = 0

            for i in range(1, len(traj_points)):
                # 计算与上一个点的最大关节角度差
                max_diff = max(
                    abs(traj_points[i][j] - filtered_points[-1][j]) for j in range(6))

                # 如果差值太小（<0.01度），视为重复点，跳过
                if max_diff < 0.01:
                    duplicate_count += 1
                    continue

                filtered_points.append(traj_points[i])

            if duplicate_count > 0:
                print(f"[轨迹-ServoJ] 过滤了 {duplicate_count} 个重复点")

            traj_points = filtered_points

        # 根据速度缩放调整cmdT
        actual_cmdT = cmdT / speed_scale
        # 只限制最小值，不限制最大值（允许任意降速）
        actual_cmdT = max(0.001, actual_cmdT)

        total_points = len(traj_points)
        print(f"[轨迹-ServoJ] 开始跟随，共 {total_points} 个点")
        print(
            f"[轨迹-ServoJ] 速度缩放: {speed_scale:.2f}x，周期: {actual_cmdT*1000:.1f}ms")

        # 检查首个点与当前位置的差距，必要时先移动到起始点
        current_joints = self.GetActualJointPos(flag=0)  # 获取角度
        if current_joints:
            max_init_diff = max(
                abs(traj_points[0][j] - current_joints[j]) for j in range(6))
            if max_init_diff > 1.0:  # 差距超过1度就需要先移动
                print(f"[轨迹-ServoJ] 起始点与当前位置差距: {max_init_diff:.2f}°")
                print(f"[轨迹-ServoJ] 先用MoveJ移动到轨迹起始点...")
                ret = self.MoveJ(joint_pos=traj_points[0], vel=20.0)
                if ret != 0:
                    print(f"[轨迹-ServoJ] 移动到起始点失败，错误码: {ret}")
                    return False
                print(f"[轨迹-ServoJ] 已到达起始点")
                time.sleep(0.5)  # 等待稳定

        # 开始伺服运动
        error = self.robot.ServoMoveStart()
        if error != 0:
            print(f"[轨迹-ServoJ] 伺服运动开始失败，错误码: {error}")
            return None
        print("[轨迹-ServoJ] 伺服运动开始")

        # 逐点发送ServoJ指令
        error_count = 0
        start_time = time.time()

        for idx, joints in enumerate(traj_points, 1):
            error = self.robot.ServoJ(
                joint_pos=joints,
                axisPos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 外部轴位置
                acc=0.0,
                cmdT=actual_cmdT,
                id=idx  # 使用点序号作为指令ID
            )

            if error != 0:
                error_count += 1
                if error_count <= 5:  # 只打印前5个错误
                    print(
                        f"[轨迹-ServoJ] 点 {idx}/{total_points} 失败，错误码: {error}")
                    if error == 14:
                        print(f"[轨迹-ServoJ] 错误14：速度超限，建议减小speed_scale或增大cmdT")
                        # 尝试降速继续
                        actual_cmdT = actual_cmdT * 1.5
                        print(f"[轨迹-ServoJ] 自动降速至 {actual_cmdT*1000:.1f}ms")

            # 精确控制发送周期
            time.sleep(actual_cmdT)

        # 结束伺服运动
        error = self.robot.ServoMoveEnd()
        if error != 0:
            print(f"[轨迹-ServoJ] 伺服运动结束失败，错误码: {error}")

        elapsed = time.time() - start_time
        print(f"[轨迹-ServoJ] 跟随结束，共 {total_points} 点，失败 {error_count} 点")
        print(f"[轨迹-ServoJ] 总耗时: {elapsed:.2f}秒")

        if error_count > total_points * 0.1:  # 失败率超过10%
            print(f"[轨迹-ServoJ] 建议：使用MoveJ方案或减小speed_scale")

        return error_count == 0

    def __del__(self):
        """析构函数"""
        self.disconnect()


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


if __name__ == "__main__":
    robot_controller = RobotController()
    if not robot_controller.connect():
        raise SystemExit(3)

    # 执行 curobo 轨迹跟随 - 推荐使用MoveJ方案（实测更丝滑）
    print("\n=== 开始执行MoveJ轨迹跟踪（优化模式）===")
    # robot_controller.followcurobo("trajectory_deg.txt", vel=20, blend_time=20)

    # 如果想测试ServoJ（理论上好，但实际可能不如MoveJ流畅）
    # print("\n=== 测试ServoJ方案 ===")
    # robot_controller.followcurobo_servo("trajectory_deg.txt", speed_scale=0.3)

    # 获取最终位置，flag=1表示弧度
    JointPos = robot_controller.GetActualJointPos(flag=1)
    print("当前关节角度: ", JointPos)
    # ret=robot_controller.moveGripper(pos=0)  # 打开夹爪到9cm

    # 初始位姿角度：
    
    # joint = robot_controller.GetInverseKin(desc_pos=[0.012381874522442255, -0.6124931561926148, 0.04465229419030974, -176.6281598614809, 2.184291443880272, 51.60707091551445])
    # print("逆运动学求解关节角度(度): ", joint)
    # robot_controller.MoveJ(joint_pos=joint, vel=20)
    robot_controller.MoveJ(joint_pos=MoveConfig.joint_init, vel=40)

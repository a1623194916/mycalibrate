import Robot
from config import MoveConfig
import time
import traceback
import os


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
        self, joint_pos, tool, user, desc_pos=None, vel=20.0, acc=0.0, ovl=100.0, exaxis_pos=None, blendT=-1.0, offset_flag=0,
        offset_pos=None,
    ):
        """
        关节空间运动

        必选参数:
            joint_pos: 目标关节位置，单位[°]
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

        try:
            ret = self.robot.MoveJ(
                joint_pos,
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
        描述

        逆运动学，笛卡尔位姿求解关节位置

        必选参数
        type:0-绝对位姿(基坐标系)，1-相对位姿（基坐标系），2-相对位姿（工具坐标系）
        desc_pose:[x,y,z,rx,ry,rz],工具位姿，单位[mm][°]

        默认参数
        config:关节配置，[-1]-参考当前关节位置求解，[0~7]-依据关节配置求解 默认-1

        返回值
        错误码 成功-0 失败- errcode
        joint_pos=[j1,j2,j3,j4,j5,j6]：逆运动学解，笛卡尔位姿求解关节位置
        """
        if not self.is_connected or self.robot is None:
            print("[移动] 机械臂未连接")
            return None

        try:
            error, joint_pos = self.robot.GetInverseKin(type, desc_pos, config)
            if error != 0:
                print(f"[移动] 逆运动学求解失败: {error}")
                return None
            return list(joint_pos)
        except Exception as e:
            print(f"[移动] 逆运动学求解异常: {e}")
            return None

    def __del__(self):
        """析构函数"""
        self.disconnect()


if __name__ == "__main__":
    # 原来的坐标（米和弧度）
    # 0.02179060935974121 -0.34381378173828125 0.2554534454345703 -3.015476795956328 0.07517019351387323 2.9922457754514578
    # xyz(mm)+rxryrz(度)
    desc_mmdegree = [21.79060935974121, -343.81378173828125, 255.4534454345703,
                     -172.82517605429688, 4.308732537926536, 171.4936574892081]
    robot_controller = RobotController()
    robot_controller.connect()
    ret = robot_controller.GetInverseKin(0, desc_mmdegree, config=-1)
    print("逆运动学，笛卡尔位姿求解关节位置", ret)
    ret=robot_controller.MoveJ(ret, tool=1, user=0, vel=10.0, acc=0.0, ovl=100.0, blendT=-1.0)
    print("关节运动结果:",ret)
    time.sleep(2)

    # pose=robot_controller.GetActualJointPos(flag=1)
    # print("当前关节弧度:",pose)

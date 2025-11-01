import Robot
import math
robot=Robot.RPC("10.21.221.21") # 创建RPC对象，RPC是Robot模块中的一个类，用于与机器人进行通信

# print("tcp:",x,y,z,rx,ry,rz)

# 调用D2C的main函数
from d2cmouse import d2crun
from convert import convert
from color_click_3d import color_click_get_once
d2c=1

if(d2c == 1):
    x,y,z = d2crun() # 获取点击的3D坐标，单位毫米
    print('clicked point in color camera coordinates (mm):', x,y,z)
    x,y,z=x/1000.0,y/1000.0,z/1000.0 # 转换为米
    error,tcp=robot.GetActualTCPPose()
    x1,y1,z1,rx,ry,rz=tcp
    x1,y1,z1=x1/1000.0,y1/1000.0,z1/1000.0
    rx,ry,rz=math.radians(rx),math.radians(ry),math.radians(rz)
    final_x,final_y,final_z=convert(x, y, z, x1, y1, z1, rx, ry, rz)
    final_x,final_y,final_z=final_x*1000.0,final_y*1000.0,final_z*1000.0 # 转换回毫米
    print('D2C 基坐标系(mm):', final_x,final_y,final_z)

# 外参
if(d2c == 0):
    x,y,z = color_click_get_once(timeout=60, in_meters=False)  # 返回毫米
    print('相机坐标系(mm):', x,y,z)
    x,y,z=x/1000.0,y/1000.0,z/1000.0  # 转换为米
    error,tcp=robot.GetActualTCPPose()
    x1,y1,z1,rx,ry,rz=tcp
    x1,y1,z1=x1/1000.0,y1/1000.0,z1/1000.0
    rx,ry,rz=math.radians(rx),math.radians(ry),math.radians(rz)
    final_x,final_y,final_z=convert(x, y, z, x1, y1, z1, rx, ry, rz)
    final_x,final_y,final_z=final_x*1000.0,final_y*1000.0,final_z*1000.0
    print('基坐标系手动转换(mm):', final_x,final_y,final_z)



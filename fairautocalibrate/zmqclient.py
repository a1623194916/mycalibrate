#!/usr/bin/env python
# -*- coding: utf-8 -*-
import zmq
import cv2
import numpy as np
import time
import os
import json

def recv_response(socket):
    """
    接收来自边缘盒子的应答
    协议：先收 JSON 元数据，再收二进制数据
    """
    try:
        # 1. 接收元数据 JSON
        md = socket.recv_json()
        
        # 2. 接收二进制数据 (copy=False 减少内存拷贝)
        msg = socket.recv(copy=False)
        
        if md.get('status') != 'ok':
            print(f"[服务端错误] {md.get('msg', 'Unknown error')}")
            return None, None, None

        data_type = md.get('data_type', 'image')
        
        # --- A. 处理点云 (PLY文件流) ---
        if data_type == 'pointcloud':
            return bytes(msg), data_type, md

        # --- B. 处理 Numpy 深度数组 (DN命令) ---
        if data_type == 'depth_numpy':
            buf = memoryview(msg)
            try:
                dtype = np.dtype(md['dtype'])
                shape = tuple(md['shape'])
                array = np.frombuffer(buf, dtype=dtype).reshape(shape)
                return array, data_type, md
            except Exception as e:
                print(f"[解析错误] Numpy转换失败: {e}")
                return None, None, None

        # --- C. 处理图像 (彩色/深度/可视化) ---
        if md.get('compressed', False):
            # 压缩格式 (JPG/PNG) -> 解码
            buf = np.frombuffer(msg, dtype=np.uint8)
            array = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        else:
            # 原始字节流 (如 Raw Depth uint16)
            buf = memoryview(msg)
            dtype = np.dtype(md['dtype']) # 通常是 uint16
            shape = tuple(md['shape'])
            array = np.frombuffer(buf, dtype=dtype).reshape(shape)
            
        return array, data_type, md

    except Exception as e:
        print(f"[通信异常] {e}")
        return None, None, None

def main():
    # --- 配置 ---
    PORT = "5555"
    SAVE_DIR = "received_data"
    save_dir_abs = os.path.abspath(SAVE_DIR)
    
    # 检查是否为无头模式（无显示器）
    headless = False # 如果在服务器运行请设为 True
    
    context = zmq.Context()
    
    # 【重要】根据服务端的 Reverse Connect 模式：
    # 服务端是 REP + Connect
    # 客户端必须是 REQ + Bind
    print(f"[启动] 正在监听端口 {PORT} (等待边缘盒子连接)...")
    socket = context.socket(zmq.REQ)
    socket.bind(f"tcp://*:{PORT}")
    
    print(f"[提示] 数据将保存到: {save_dir_abs}")
    print(f"[指令] \n C  = 彩色图\n D  = 原始深度图(PNG)\n DV = 深度可视化(Jet)\n DN = 深度数据(Numpy .npy)\n P  = 点云(PLY)\n Q  = 退出")
    
    # 计数器
    cnt = {'C': 0, 'D': 0, 'DV': 0, 'DN': 0, 'P': 0}

    try:
        while True:
            cmd = input("\n指令 > ").strip().upper()
            
            if cmd == 'Q':
                break
            
            if cmd not in ['C', 'D', 'DV', 'DN', 'P']:
                print("[警告] 无效指令")
                continue
            
            # 1. 发送请求
            start_time = time.time()
            try:
                socket.send_string(cmd)
            except zmq.ZMQError as e:
                print(f"[发送失败] {e}")
                continue
            
            # 2. 接收数据
            data, data_type, md = recv_response(socket)
            cost = (time.time() - start_time) * 1000
            
            if data is not None:
                os.makedirs(SAVE_DIR, exist_ok=True)
                timestamp = int(time.time())
                
                # --- 处理点云 ---
                if data_type == 'pointcloud':
                    filename = os.path.join(SAVE_DIR, f"pc_{cnt['P']:03d}_{timestamp}.ply")
                    with open(filename, 'wb') as f:
                        f.write(data)
                    print(f"[接收] PLY点云 | 大小: {md['size_bytes']/1024:.1f}KB | 耗时: {cost:.1f}ms")
                    print(f"       已保存: {filename}")
                    cnt['P'] += 1

                # --- 处理 Numpy 深度 ---
                elif data_type == 'depth_numpy':
                    filename = os.path.join(SAVE_DIR, f"depth_raw_{cnt['DN']:03d}_{timestamp}.npy")
                    np.save(filename, data)
                    print(f"[接收] Numpy深度 | 形状: {data.shape} | 耗时: {cost:.1f}ms")
                    print(f"       已保存: {filename}")
                    cnt['DN'] += 1

                # --- 处理图像 (彩色/深度图/可视化) ---
                else:
                    if cmd == 'C':
                        prefix = "color"
                        ext = ".jpg"
                        idx = 'C'
                    elif cmd == 'D':
                        prefix = "depth_u16"
                        ext = ".png" # uint16 必须存为 png
                        idx = 'D'
                    elif cmd == 'DV':
                        prefix = "depth_vis"
                        ext = ".jpg"
                        idx = 'DV'
                    
                    filename = os.path.join(SAVE_DIR, f"{prefix}_{cnt[idx]:03d}_{timestamp}{ext}")
                    cv2.imwrite(filename, data)
                    
                    print(f"[接收] 图像数据 | 形状: {data.shape} | 耗时: {cost:.1f}ms")
                    print(f"       已保存: {filename}")
                    cnt[idx] += 1
                    
                    # 可视化显示 (如果不是无头模式)
                    # if not headless and cmd != 'D': # 原始深度图直接显示通常一片黑，跳过
                    #     cv2.imshow("Remote Preview", data)
                    #     cv2.waitKey(1)
            else:
                print("[失败] 未收到有效数据或发生错误")

    except KeyboardInterrupt:
        print("\n[退出] 用户中断")
    finally:
        socket.close()
        context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
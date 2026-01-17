#!/usr/bin/env python
# -*- coding: utf-8 -*-
import zmq
import cv2
import numpy as np
import time
import os

def recv_image_response(socket):
    """
    接收来自边缘盒子的图像应答
    """
    # 接收元数据 JSON
    md = socket.recv_json()
    
    # 接收二进制数据
    msg = socket.recv(copy=False)
    
    if md.get('status') != 'ok':
        print(f"[服务端错误] {md.get('msg', 'Unknown error')}")
        return None

    if md.get('compressed', False):
        # 解压 JPEG
        buf = np.frombuffer(msg, dtype=np.uint8)
        array = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        # 原始数据
        buf = memoryview(msg)
        array = np.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])
        
    return array

def main():
    # --- 配置 ---
    PORT = "5555"
    SAVE_DIR = "received_images"
    save_dir_abs = os.path.abspath(SAVE_DIR)
    headless = not bool(os.environ.get("DISPLAY"))
    print(f"[提示] 图像将保存到: {save_dir_abs}")
    
    context = zmq.Context()
    # 主机作为 REQ (请求端)，同时 BIND (绑定端口)
    # 这意味着主机有固定的 IP，边缘盒子连上来
    socket = context.socket(zmq.REQ)
    socket.bind(f"tcp://*:{PORT}")
    
    print(f"[主机] 服务已启动，监听端口 {PORT}...")
    print(f"[提示] 按 'Enter' 键请求拍照，按 'q' 退出")

    try:
        while True:
            cmd = input("Command > ")
            
            if cmd == 'q':
                break
            
            # 发送请求
            print("[请求] 正在请求边缘盒子拍照...")
            start_time = time.time()
            socket.send_string("GET_RGB")
            
            # 等待回复 (阻塞)
            img = recv_image_response(socket)
            cost = (time.time() - start_time) * 1000
            
            if img is not None:
                print(f"[成功] 收到图像: {img.shape}, 耗时: {cost:.1f}ms")
                os.makedirs(SAVE_DIR, exist_ok=True)
                filename = os.path.join(SAVE_DIR, f"rgb_{int(time.time()*1000)}.png")
                cv2.imwrite(filename, img)
                print(f"[已保存] {filename}")
                if not headless:
                    cv2.imshow("Remote RGB", img)
                    cv2.waitKey(1) # 刷新界面
            else:
                print("[失败] 未收到有效图像")

    except KeyboardInterrupt:
        print("\n停止")
    finally:
        socket.close()
        context.term()
        if not headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
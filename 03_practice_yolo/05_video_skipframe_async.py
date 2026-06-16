# -*- coding: utf-8 -*-
"""
真正并行的推理：主线程读摄像头+显示，子线程异步推理

vs 关卡5的demo:
  关卡5:  单线程 → 推理阻塞 → 只能跳帧"模拟"流畅
  本文件:  双线程 → 推理不阻塞显示 → 真正的实时

架构:
  主线程 (快): cap.read() → 显示上一轮结果 → 等键盘
  推理线程 (慢): model.predict() → 画框 → 把结果交给主线程

  两个线程通过一个共享变量通信:
    主线程写 frame (给推理线程)，读 annotated (显示)
    推理线程读 frame (做推理)，写 annotated (给主线程显示)
"""

import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def main():
    # ═══ 1. 加载模型 ═══
    pt = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    if not pt.exists():
        pt = "yolo11n.pt"
    model = YOLO(str(pt))

    # ═══ 2. 打开摄像头 ═══
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打不开")
        return

    # ═══ 3. 共享变量 ═══
    # 两个线程之间通过这几个变量通信:
    latest_frame = None      # 主线程写: 最新读到的帧
    latest_result = None     # 推理线程写: 最新推理结果(画好框的图)
    running = True           # 两个线程一起检查: 是否继续跑
    lock = threading.Lock()  # 锁: 防止两个线程同时读写同一个变量

    # ═══ 4. 推理线程 ═══
    def inference_worker():
        """后台线程: 不停拿最新帧做推理"""
        nonlocal latest_result
        while running:
            # 拿最新帧 (加锁防止冲突)
            with lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            # with lock: 进入时获取锁，出去时自动释放

            if frame is None:
                time.sleep(0.001)   # 主线程还没读到帧，等一下
                continue

            # 推理 (这一步耗时、阻塞，但在子线程里，不影响主线程显示)
            r = model.predict(frame, conf=0.25, verbose=False, workers=0)
            annotated = r[0].plot()

            # 把结果写回共享变量
            with lock:
                latest_result = (annotated, len(r[0].boxes))

    # 启动推理线程
    # daemon=True: 主线程退出时自动杀掉子线程
    t = threading.Thread(target=inference_worker, daemon=True)
    t.start()

    # ═══ 5. 主循环: 读帧 + 显示 ═══
    fps = 0.0
    frame_count = 0

    print("主线程=读帧+显示 | 子线程=异步推理 | 按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))

        # 把新帧发给推理线程
        t0 = time.time()
        with lock:
            latest_frame = frame

        # 拿推理线程的最新结果 (可能比 frame 旧)
        display = frame  # 默认显示原图
        n_boxes = 0
        with lock:
            if latest_result is not None:
                display, n_boxes = latest_result

        # FPS
        dt = max(time.time() - t0, 0.001)
        fps = 0.9 * fps + 0.1 / dt

        cv2.putText(display, f"FPS:{fps:.0f} boxes:{n_boxes} frame:{frame_count}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("异步推理 — 显示不卡", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ═══ 6. 清理 ═══
    running = False   # 通知子线程退出
    cap.release()
    cv2.destroyAllWindows()
    print("[OK] 异步推理 完成")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
关卡 5 YOLO版: 视频检测 — 逐帧 vs 跳帧 实时对比

做什么:
  左边 = 每帧都跑推理（真实负载，慢但准）
  右边 = 跳帧复用上次结果（跳过一些帧，快但有延迟）
  按 s 切换跳帧间隔 (1/3/5/10)

学什么:
  - cv2.VideoCapture: 读摄像头/视频文件
  - 跳帧策略: 不是"跳过不检测"，而是"用旧结果顶替"
  - 指数移动平均 (EMA) 平滑 FPS
  - np.hstack: 两张图横着拼一起

跑法: python 03_practice_yolo/05_video_skipframe.py
"""

import cv2          # OpenCV: 读摄像头、画图、显示窗口
import time          # time.time(): 计时（秒级精度）
import numpy as np   # np.hstack: 水平拼接图像
from ultralytics import YOLO
from pathlib import Path


def main():
    # ═══ 1. 加载模型 ═══
    # Path: Windows/Linux 自动处理路径分隔符
    pt = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    # 兜底: 训练模型不存在就用 COCO 预训练的 yolo11n
    if not pt.exists():
        pt = "yolo11n.pt"
    # YOLO(str(pt)): 加载模型，参数必须是字符串（Path 对象要先转）
    model = YOLO(str(pt))

    # ═══ 2. 打开摄像头 ═══
    # cv2.VideoCapture(设备编号): 打开视频源
    #   0   = 默认摄像头（笔记本自带的那个）
    #   1   = 外接摄像头
    #   "video.mp4" = 视频文件
    source = 0
    cap = cv2.VideoCapture(source)

    # .isOpened(): 检查视频源是否成功打开
    #   返回 True/False
    if not cap.isOpened():
        print("摄像头打不开")
        return

    # ═══ 3. 初始化变量 ═══
    # skip: 跳帧间隔（每 N 帧做一次推理，其他帧复用旧结果）
    #   skip=1: 不跳（每帧推理）
    #   skip=3: 每 3 帧推理一次（跳 2 帧）
    #   skip=5: 每 5 帧推理一次（跳 4 帧）
    skip = 3

    # frame_count: 总帧数计数器（从 1 开始）
    frame_count = 0

    # fps 用 EMA (指数移动平均) 平滑:
    #   new_fps = 0.9 * old_fps + 0.1 * current_fps
    #   新值占 10%，旧值占 90% → FPS 显示不会剧烈抖动
    fps_full, fps_skip = 0.0, 0.0

    # last_annotated: 缓存上一次推理结果
    #   跳帧时直接用这个缓存，不用重新推理
    last_annotated = None

    print(f"跳帧={skip} | 左=逐帧推理 | 右=跳帧复用 | 按 s 切换跳帧 | 按 q 退出")

    # ═══ 4. 主循环: 逐帧读取 ═══
    # cap.isOpened(): 摄像头/视频是否还开着
    #   摄像头掉了或视频播完 → 返回 False → 退出循环
    while cap.isOpened():

        # ── 4.1 读一帧 ──
        # cap.read(): 从视频流读下一帧
        #   返回 (是否成功, 帧数据)
        #   ret = True/False → 是否成功读到
        #   frame = numpy 数组 (H, W, 3) BGR 格式
        ret, frame = cap.read()
        if not ret:     # 视频播完 / 摄像头断连
            break
        frame_count += 1

        # cv2.resize(图片, (宽, 高)): 缩放图片到指定尺寸
        #   (640, 480): 缩到 640×480，减少计算量 + 方便拼接
        frame = cv2.resize(frame, (640, 480))

        # ── 4.2 左半: 每帧都推理 ──
        # time.time(): 返回当前时间戳（从 1970-01-01 起算，单位秒）
        #   用来算"这一帧推理花了多久"
        t0 = time.time()

        # model.predict(): YOLO 推理
        #   frame: 输入图片（numpy 数组 BGR）
        #   conf=0.25: 置信度阈值（低于 25% 的框不显示）
        #   verbose=False: 不刷屏打印
        #   workers=0: Windows 下必须设，避免子进程 spawn 无限递归
        r = model.predict(frame, conf=0.25, verbose=False, workers=0)

        # 推理耗时 = 现在 - 开始
        t_full = time.time() - t0

        # r[0].plot(): YOLO 内置画框函数
        #   在原图上画检测框 + 类别名 + 置信度，返回画好的图片
        left = r[0].plot()

        # len(r[0].boxes): 这一帧检测到几个目标
        n_full = len(r[0].boxes)

        # ── 4.3 右半: 跳帧（只在特定帧跑推理）──
        t1 = time.time()

        # frame_count % skip == 1: 推理帧 → 复用左边的结果，不重复跑
        #   其他帧 → 用上一次缓存的框（有滞后感）
        if frame_count % skip == 1:
            last_annotated = left   # 缓存左边刚算完的结果
        right = last_annotated if last_annotated is not None else frame

        t_skip = time.time() - t1
        # 推理帧: t_skip ≈ 0（只做了赋值）
        # 跳帧:   t_skip ≈ 0（也是赋值）

        # ── 4.4 FPS 平滑 ──
        # EMA (Exponential Moving Average) 指数移动平均:
        #   fps = 0.9 * fps_old + 0.1 * fps_new
        #
        #   为什么不用算术平均？
        #     算术平均要存历史所有 fps，内存涨
        #     EMA 只存一个数，每帧更新
        #
        #   0.9/0.1 这个比例:
        #     0.9 大 → FPS 变化平滑，但对真实变化响应慢
        #     0.1 大 → FPS 更新快，但跳动剧烈
        #
        #   max(t_full, 0.001): 防止除以 0（time.time() 精度有限）
        fps_full = 0.9 * fps_full + 0.1 / max(t_full, 0.001)
        fps_skip = 0.9 * fps_skip + 0.1 / max(t_skip, 0.001)

        # ── 4.5 在图上标信息 ──
        # cv2.putText(图片, 文字, (x,y), 字体, 字号, 颜色BGR, 粗细):
        #   (x,y) = 文字左下角坐标
        #   颜色是 BGR 格式（不是 RGB！）
        #   fps_full:.0f: 取整（FPS 不需要小数）
        cv2.putText(left, f"FULL FPS:{fps_full:.0f} frames:{n_full}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #                                  字号 0.5   绿色      粗细 1

        cv2.putText(right, f"SKIP{skip} FPS:{fps_skip:.0f}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        #                                                     黄色

        # ── 4.6 显示 ──
        # np.hstack([图1, 图2]): 水平拼接
        #   两张 640×480 的图 → 1280×480
        #   左边逐帧推理结果，右边跳帧结果
        # cv2.imshow(窗口标题, 图片): 弹窗显示
        cv2.imshow("左=逐帧 | 右=跳帧(按s切换)", np.hstack([left, right]))

        # ── 4.7 键盘交互 ──
        # cv2.waitKey(毫秒): 等待键盘输入
        #   参数 1: 等 1 毫秒，没按就继续（视频不能用 0，否则卡住）
        #   & 0xFF: 取低 8 位（跨平台兼容，mac 上按键码是 16 位）
        key = cv2.waitKey(1) & 0xFF

        # ord('q'): 字符 q 的 ASCII 码
        if key == ord('q'):
            break   # 按 q 退出

        # ord('s'): 字符 s 的 ASCII 码
        if key == ord('s'):
            # 循环切换跳帧间隔: 1→3→5→10→1...
            # dict.get(skip, 3): 如果 skip 不在字典里，返回默认值 3
            skip = {1: 3, 3: 5, 5: 10, 10: 1}.get(skip, 3)
            print(f"跳帧切换为 {skip}")

    # ═══ 5. 清理 ═══
    # cap.release(): 释放摄像头/视频文件句柄
    #   不调的话摄像头灯可能一直亮着
    cap.release()
    # cv2.destroyAllWindows(): 关闭所有 OpenCV 弹窗
    cv2.destroyAllWindows()
    print("[OK] 关卡 5 YOLO版 完成")




if __name__ == "__main__":
    main()

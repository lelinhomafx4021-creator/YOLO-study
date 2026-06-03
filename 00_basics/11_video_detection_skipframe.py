# -*- coding: utf-8 -*-
"""
11_video_detection_skipframe.py — 视频检测（单线程 + 跳帧 + 旧框复用）

跑法：
    python 00_basics/11_video_detection_skipframe.py

    默认用摄像头 source=0。改成 "xxx.mp4" 可跑本地视频。
"""

import cv2, time
from ultralytics import YOLO

# ═══════════════════════════════════════
# 找你的 best.pt — 用 baseline 版本
# ═══════════════════════════════════════
from pathlib import Path
pt = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
if not pt.exists():
    print("没找到 best.pt，用的 yolo11n.pt（预训练 80 类）")
    pt = "yolo11n.pt"

model = YOLO(str(pt))
print(f"加载模型: {pt}")

# ═══════════════════════════════════════
# 打开视频源（摄像头 0 或者视频文件路径）
# ═══════════════════════════════════════
source = 0                               # ← 摄像头；改成 "xxx.mp4" 跑本地视频
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("打不开摄像头，用静态图片")
    source = "result.jpg"
    cap = cv2.VideoCapture(source)

# ═══════════════════════════════════════
# 跳帧参数
# ═══════════════════════════════════════
SKIP = 3                                 # 每 3 帧推理 1 帧
frame_count = 0
last_annotated = None                    # 上一轮画好框的帧
fps_smooth = 0.0                         # 平滑 FPS

# ═══════════════════════════════════════
# 主循环
# ═══════════════════════════════════════
print(f"跳帧间隔: {SKIP}（每 {SKIP} 帧推理一次）")
print("按 q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    t0 = time.time()

    if frame_count % SKIP == 1:
        # ── 推理 ──
        results = model.predict(frame, conf=0.25, verbose=False)
        last_annotated = results[0].plot()   # YOLO 自动画框
        display = last_annotated
    else:
        # ── 复用旧框 ──
        if last_annotated is not None:
            display = last_annotated
        else:
            display = frame                  # 第一轮还没框，显示原图

    # ── FPS 平滑 ──
    fps_smooth = 0.9 * fps_smooth + 0.1 * (1 / max(time.time() - t0, 0.001))
    cv2.putText(display, f"FPS: {fps_smooth:.0f}  Skip: {SKIP}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Video Detection (skip-frame)", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

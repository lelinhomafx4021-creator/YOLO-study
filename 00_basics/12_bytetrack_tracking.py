# -*- coding: utf-8 -*-
"""
12_bytetrack_tracking.py — YOLO + ByteTrack 多目标跟踪

跑法：
    python 00_basics/12_bytetrack_tracking.py

    YOLO 内置 ByteTrack，不需要额外安装。

效果：
    每个人/安全帽持续带 ID，即使某帧漏检也能靠卡尔曼滤波预测补位。
"""

import cv2
from ultralytics import YOLO

# ═══════════════════════════════════════
# 加载模型
# ═══════════════════════════════════════
from pathlib import Path
pt = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
if not pt.exists():
    print("没找到 best.pt，用的 yolo11n.pt")
    pt = "yolo11n.pt"

model = YOLO(str(pt))

# ═══════════════════════════════════════
# 打开视频源
# ═══════════════════════════════════════
source = 0
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("打不开摄像头，请检查")

print("ByteTrack 多目标跟踪启动...")
print("每个物体持续带 ID，按 q 退出")

# ═══════════════════════════════════════
# 跟踪循环
# ═══════════════════════════════════════
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ── model.track()：自动启用 ByteTrack ──
    # persist=True：跨帧保持 ID（核心参数！否则每帧 ID 全变）
    results = model.track(
        source=frame,
        persist=True,          # ← 跨帧 ID 持久化
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False,
    )

    # ── 画框 ──
    annotated = results[0].plot()

    # ── 显示 ──
    cv2.imshow("ByteTrack Tracking — 每个物体持续带 ID", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ═══════════════════════════════════════
# 面试知识点
# ═══════════════════════════════════════
print("""
ByteTrack 核心原理：
  1. 高分框 → 直接匹配（确定是物体）
  2. 低分框 → 不丢弃，先留着（可能是遮挡等边界情况）
  3. 卡尔曼滤波 → 预测每个跟踪目标的下一帧位置
  4. 匈牙利算法 → 把"检测到的框"和"已有的跟踪目标"做最优配对

对比其他跟踪器：
  SORT:     卡尔曼 + 匈牙利，简单但 ID switch 多
  DeepSORT: SORT + 深度学习外观特征（更准，但更慢）
  ByteTrack: 低分框复用策略（YOLO 默认，速度和准度最平衡）
""")

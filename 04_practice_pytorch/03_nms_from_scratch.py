# -*- coding: utf-8 -*-
"""
关卡 3 PyTorch版: 手写 NMS 算法 + 可视化对比

不依赖任何检测框架。自己造框、自己写 NMS、自己画图。
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg"); import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

# ═══ 1. 造假框（模拟 YOLO 输出）═══
boxes = np.array([
    # 目标A: 猫, 3个重叠框 (同一物体多框)
    [50, 50,   150, 150,  0.95, 0],
    [60, 55,   145, 148,  0.82, 0],
    [45, 60,   155, 140,  0.71, 0],
    # 目标B: 另一只猫, 离A很近 (密集场景)
    [130, 40,  220, 140,  0.88, 0],
    [125, 45,  215, 135,  0.65, 0],
    # 目标C: 狗
    [250, 60,  350, 160,  0.91, 1],
    [255, 55,  345, 165,  0.78, 1],
    # 背景误检
    [300, 200, 340, 240,  0.42, 0],
])
names = {0: "猫", 1: "狗"}
colors = {0: "blue", 1: "orange"}

# ═══ 2. 手写 IoU 计算 ═══
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-8)

# ═══ 3. 手写传统 NMS ═══
def nms(boxes, iou_thresh=0.5):
    order = np.argsort(boxes[:,4])[::-1]   # 按置信度降序
    keep = []
    while len(order) > 0:
        cur = order[0]; keep.append(cur)
        if len(order) == 1: break
        ious = [compute_iou(boxes[cur], boxes[i]) for i in order[1:]]
        order = order[1:][np.array(ious) <= iou_thresh]
    return keep

# ═══ 4. 手写 Soft-NMS ═══
def soft_nms(boxes, iou_thresh=0.5, sigma=0.5):
    boxes = boxes.copy().astype(float)
    order = list(np.argsort(boxes[:,4])[::-1])
    keep = []
    while len(order) > 0:
        cur = order.pop(0); keep.append(cur)
        for idx in list(order):
            iou_val = compute_iou(boxes[cur], boxes[idx])
            if iou_val >= iou_thresh:
                boxes[idx,4] *= np.exp(-iou_val*iou_val / sigma)  # 降分不归零
    return keep

# ═══ 5. 画 3 张对比图 ═══
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
titles = ["NMS 前 (8框)", "传统 NMS IoU=0.5", "Soft-NMS IoU=0.5"]

for ax, title, keep in zip(axes, titles,
    [list(range(8)), nms(boxes, 0.5), soft_nms(boxes, 0.5)]):
    ax.set_xlim(0, 400); ax.set_ylim(0, 300); ax.invert_yaxis()
    ax.set_title(title, fontweight="bold")
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b[:4]
        if i in keep:
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2,
                                      edgecolor=colors[int(b[5])], facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{names[b[5]]} {b[4]:.2f}", fontsize=8, color=colors[b[5]])
        else:
            rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=1,
                                      edgecolor="gray", facecolor="none", linestyle="--")
            ax.add_patch(rect)
    ax.text(10, 275, f"保留 {len(keep)}/{len(boxes)} 框",
            bbox=dict(boxstyle="round", facecolor="wheat"))

plt.tight_layout(); plt.show()
print("✅ 关卡 3 PyTorch版 完成 — 手写 NMS + IoU + Soft-NMS, 全不依赖框架")

# -*- coding: utf-8 -*-
"""
10_nms_demo.py — NMS 可视化演示
展示 NMS 如何去除重叠框，以及 IoU 阈值如何影响结果。
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 中文显示
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.patches as patches

# ============================================================
# 1. 造一批重叠的框（模拟模型输出）
# ============================================================
# 格式：[x1, y1, x2, y2, confidence, class_id]
boxes = np.array([
    # ── 目标 A：一只猫，模型预测了 3 个重叠框 ──
    [50,  50,  150, 150, 0.95, 0],   # A1 最高置信度
    [60,  55,  145, 148, 0.82, 0],   # A2 和 A1 重叠很大
    [45,  60,  155, 140, 0.71, 0],   # A3
    # ── 目标 B：另一只猫，离 A 很近 ──
    [130, 40,  220, 140, 0.88, 0],   # B1 和 A 有部分重叠
    [125, 45,  215, 135, 0.65, 0],   # B2
    # ── 目标 C：一只狗 ──
    [250, 60,  350, 160, 0.91, 1],   # C1
    [255, 55,  345, 165, 0.78, 1],   # C2
    # ── 一个离群的框（背景误检）──
    [300, 200, 340, 240, 0.42, 0],
])

CLASS_NAMES = {0: "猫", 1: "狗"}
COLORS = {0: "blue", 1: "orange"}


def iou(box1, box2):
    """计算两个框的 IoU（交并比）"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def nms(boxes, iou_threshold):
    """传统 NMS：重叠的框直接删除（硬抑制）"""
    if len(boxes) == 0:
        return []

    order = np.argsort(boxes[:, 4])[::-1]   # 按置信度降序排列
    keep = []

    while len(order) > 0:
        current = order[0]                   # 当前最高分框 → 保留
        keep.append(current)

        if len(order) == 1:
            break

        other = order[1:]
        ious = np.array([iou(boxes[current], boxes[i]) for i in other])
        # 硬删除：IoU > 阈值的框直接扔掉
        order = other[ious <= iou_threshold]

    return [int(k) for k in keep]


def soft_nms(boxes, iou_threshold=0.5, sigma=0.5, conf_threshold=0.3):
    """
    Soft-NMS：不直接删框，而是降低重叠框的置信度（软抑制）

    公式：score = score × exp(-IoU² / sigma)
           └── 当前分数 ─┘   └──── 衰减因子 ────┘
           IoU 越大 → 衰减越狠 → 分数降越多
           但不会降到 0 → 密集场景下还有机会被保留

    参数：
        iou_threshold: IoU 超过这个值才开始衰减（低于的不受影响）
        sigma: 衰减强度，越大衰减越温和
        conf_threshold: 最终分数低于这个值的框会被删掉
    """
    if len(boxes) == 0:
        return []

    # 复制一份，因为要修改置信度
    boxes = boxes.copy().astype(float)
    keep = []

    # 按置信度降序排列
    order = list(np.argsort(boxes[:, 4])[::-1])

    while len(order) > 0:
        current = order.pop(0)   # 取当前最高分框 → 保留
        keep.append(current)

        if len(order) == 0:
            break

        # 计算当前框和其余框的 IoU
        rm_indices = []  # 要移除的（分数太低或 IoU 太高）
        for idx in order:
            iou_val = iou(boxes[current], boxes[idx])

            if iou_val >= iou_threshold:
                # ── 关键区别：不是直接删，而是降分 ──
                boxes[idx, 4] *= np.exp(-iou_val * iou_val / sigma)
                #                  └────────┬────────┘
                #                 衰减因子：IoU 越大 → 越接近 0

                # 降分后太低 → 删掉
                if boxes[idx, 4] < conf_threshold:
                    rm_indices.append(idx)

        # 删掉分数降到阈值以下的框
        for idx in rm_indices:
            order.remove(idx)

    return [int(k) for k in keep]


def draw_boxes(ax, boxes, keep_indices, title):
    """画框：保留的用实线，被抑制的用虚线"""
    ax.set_xlim(0, 400); ax.set_ylim(0, 300)
    ax.invert_yaxis()  # 图像坐标系：左上角为原点
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = box
        w, h = x2 - x1, y2 - y1
        color = COLORS[int(cls)]

        if i in keep_indices:
            # 保留 → 实线
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2.5,
                                      edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{CLASS_NAMES[int(cls)]} {conf:.2f} ✓",
                    fontsize=8, color=color, fontweight="bold")
        else:
            # 被抑制 → 虚线 + 灰色
            rect = patches.Rectangle((x1, y1), w, h, linewidth=1.5,
                                      edgecolor="gray", facecolor="none",
                                      linestyle="--")
            ax.add_patch(rect)
            ax.text(x1, y2 + 12, f"❌ {conf:.2f}", fontsize=7, color="gray")


# ============================================================
# 2. 对比不同 NMS 阈值
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

scenarios = [
    # (子图, 标题, 处理方式)
    (axes[0, 0], "NMS 前（原始输出）\n8 个框全部显示", "none"),
    (axes[0, 1], "传统 NMS IoU=0.5\n硬删除：重叠>50%直接干掉", "nms"),
    (axes[1, 0], "Soft-NMS IoU=0.5\n软降分：重叠框降置信度而非删除", "soft"),
    (axes[1, 1], "对比：框的置信度变化\n(红色=原始, 蓝色=Soft-NMS后)", "score_compare"),
]

for ax, title, mode in scenarios:
    if mode == "none":
        keep = list(range(len(boxes)))
        draw_boxes(ax, boxes, keep, title)
        n_kept = len(keep)
        n_removed = 0

    elif mode == "nms":
        keep = nms(boxes.copy(), 0.5)
        draw_boxes(ax, boxes, keep, title)
        n_kept = len(keep)
        n_removed = len(boxes) - n_kept

    elif mode == "soft":
        keep = soft_nms(boxes.copy(), iou_threshold=0.5, sigma=0.5)
        draw_boxes(ax, boxes, keep, title)
        n_kept = len(keep)
        n_removed = len(boxes) - n_kept

    elif mode == "score_compare":
        # 对比原始分数和 Soft-NMS 降分后的效果
        ax.set_xlim(0, 8); ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("置信度")
        ax.set_xticks(range(8))
        ax.set_xticklabels(["A1","A2","A3","B1","B2","C1","C2","BG"], fontsize=9)

        # 原始分数
        original_scores = boxes[:, 4]
        ax.bar(np.arange(8) - 0.15, original_scores, 0.3,
               color="#e94560", label="原始分数")

        # Soft-NMS 后的分数
        temp = boxes.copy().astype(float)
        _ = soft_nms(temp, iou_threshold=0.5, sigma=0.5)
        soft_scores = temp[:, 4]
        ax.bar(np.arange(8) + 0.15, soft_scores, 0.3,
               color="#00d4ff", label="Soft-NMS 后")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # 标注被抑制的框
        keep_set = set(soft_nms(boxes.copy(), iou_threshold=0.5, sigma=0.5))
        for i in range(8):
            if i not in keep_set and soft_scores[i] < 0.3:
                ax.annotate("❌", (i, soft_scores[i] + 0.05), fontsize=14,
                           ha="center", color="gray")

    if mode != "score_compare":
        ax.text(10, 275, f"保留 {n_kept} 框 / 移除 {n_removed} 框",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

plt.tight_layout()
plt.show()

# ============================================================
# 3. 打印 NMS vs Soft-NMS 对比
# ============================================================
print("\n" + "=" * 55)
print("NMS vs Soft-NMS 对比总结")
print("=" * 55)
print()
print("传统 NMS：")
print("  IoU > 阈值 → 直接删除 → 得分归零")
print("  缺点：密集场景容易误杀相邻目标")
print()
print("Soft-NMS：")
print("  IoU > 阈值 → 得分衰减（不归零）→ 仍有机会保留")
print("  公式：score = score × exp(-IoU² / σ)")
print()
print("  例子：两个挨得很近的安全帽")
print("    框A（置信度 0.95）→ 保留")
print("    框B（置信度 0.72）→ 和 A 的 IoU=0.6")
print("    传统 NMS：0.72 → 0  ❌ 第二个安全帽被误杀")
print("    Soft-NMS：0.72 → 0.72×exp(-0.36/0.5) = 0.35  ✅ 还活着！")
print()
print("代码里的三个参数：")
print("  iou_threshold: IoU 超过多少开始衰减（默认 0.5）")
print("  sigma:        衰减强度，越小衰减越狠（默认 0.5）")
print("  conf_threshold: 分降到多低才彻底删掉（默认 0.3）")
print()
print("YOLOv8/v11 默认用的就是 Soft-NMS 变体")

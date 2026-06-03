# -*- coding: utf-8 -*-
"""
关卡 2 YOLO版: 训练结果深度诊断
不做新训练，只读你的 baseline 训练日志（results.csv）
自动画 6 张诊断图 + 打印诊断报告

跑法: python 03_practice_yolo/02_error_analysis.py
不做训练，纯粹读数据画图，很快（几秒）
"""

import pandas as pd             # 读 CSV 表格
import numpy as np              # 数组计算（mean/argmax/std...）
import matplotlib
matplotlib.use("TkAgg")         # 用桌面窗口显示图（不依赖浏览器）
import matplotlib.pyplot as plt

# 中文字体设置（否则中文标签全变方框乱码）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False   # 负号也正常显示

# ═══════════════════════════════════════════════════════════
# 第 1 步：读 results.csv
# ═══════════════════════════════════════════════════════════

# results.csv 由 YOLO 训练自动生成，每个 epoch 结束后追加一行
# 这个文件在 runs/safety_helmet/yolo11n_baseline_v1/ 里
csv_path = r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\results.csv"
# pd.read_csv: 把 CSV 读成 DataFrame（表格对象）
# .dropna() : 删掉有空值(NaN)的行——CSV 末尾有时有空白行
df = pd.read_csv(csv_path).dropna()

# results.csv 的列名一览：
#
#   训练 Loss（越小越好，越低说明模型在训练集上拟合得越好）:
#     train/box_loss    — 框的位置损失（框画偏了多少）
#     train/cls_loss    — 分类损失（类别猜错扣多少分）
#     train/dfl_loss    — 边界精调损失（框的边缘精细度）
#
#   验证 Loss（监控过拟合——如果 val 不降反而升，就是过拟合）:
#     val/box_loss      — 同上，但在验证集上测的
#     val/cls_loss
#     val/dfl_loss
#
#   评估指标:
#     metrics/precision(B) — Precision: 预测出来的框里，有多少是对的 (B=所有类平均)
#     metrics/recall(B)    — Recall: 真实存在的目标，找到了多少
#     metrics/mAP50(B)     — mAP @ IoU=0.5 (宽松评分)
#     metrics/mAP50-95(B)  — mAP @ IoU=0.5:0.95 (严格评分)
#
#   其他:
#     epoch       — 第几轮
#     time        — 这轮花了多少秒
#     lr/pg0      — 学习率（可能逐轮衰减）
#     lr/pg1
#     lr/pg2

# df["列名"] → 返回这一列（pandas Series）
# .values     → 转成 numpy 数组（matplotlib 最喜欢吃这个格式）
epochs = df["epoch"].values               # [1, 2, 3, ..., 50]

# ── 训练 Loss ──
train_box = df["train/box_loss"].values   # [1.47, 1.26, 1.20, ...]
train_cls = df["train/cls_loss"].values   # [3.35, 2.68, 2.12, ...]
train_dfl = df["train/dfl_loss"].values   # [1.58, 1.34, 1.30, ...]

# ── 验证 Loss ──
val_box = df["val/box_loss"].values       # [1.24, 1.09, 1.20, ...]
val_cls = df["val/cls_loss"].values       # [3.11, 2.94, 2.77, ...]
val_dfl = df["val/dfl_loss"].values       # [1.38, 1.26, 1.36, ...]

# ── 评估指标 ──
precision = df["metrics/precision(B)"].values   # [0.007, 0.009, 0.008, 0.90, ...]
recall    = df["metrics/recall(B)"].values      # [0.896, 0.961, 1.000, 0.17, ...]
mAP50     = df["metrics/mAP50(B)"].values       # [0.327, 0.615, 0.558, 0.68, ...]
mAP50_95  = df["metrics/mAP50-95(B)"].values    # 比 mAP50 低是正常的

# ═══════════════════════════════════════════════════════════
# 第 2 步：自动诊断（纯数字，不看图也能做）
# ═══════════════════════════════════════════════════════════

# argmax(): 找到 mAP50 最高的位置（索引）
#   比如 [0.32, 0.61, 0.68, 0.55] → argmax() = 2（第3个最大）
best_idx = mAP50.argmax()
# int(epochs[best_idx]): 那是第几轮
best_epoch = int(epochs[best_idx])

# ═══ 误检/漏检判断 ═══
# [-1] 是 Python 取最后一个元素的快捷写法，即最后一轮的 P 和 R
p_final = precision[-1]     # 最后一轮 Precision
r_final = recall[-1]        # 最后一轮 Recall

# 诊断逻辑（阈值 0.6 = 经验值）:
if p_final < 0.6 and r_final >= 0.6:
    # P 低 R 正常 → 模型框出来的东西有太多是错的（FP 多）
    error_tag = "误检偏多 (P低 → FP多)"
elif r_final < 0.6 and p_final >= 0.6:
    # R 低 P 正常 → 很多真实目标没被找到（FN 多）
    error_tag = "漏检偏多 (R低 → FN多)"
elif p_final < 0.6 and r_final < 0.6:
    # 两个都低 → 模型还没学好
    error_tag = "误检漏检都多"
else:
    error_tag = "P/R 在合理范围"

# ═══ 过拟合判断 ═══
# gap = val_loss - train_loss
# 如果 gap 在后期扩大 → 训练集在变好但验证集跟不上 → 过拟合
gap = (val_box + val_cls + val_dfl) - (train_box + train_cls + train_dfl)

# split: 把 epoch 分成前 2/3 和后 1/3
# // = 整数除法（丢掉小数部分）
split = len(epochs) * 2 // 3
# [:split] = 前 split 个元素, [split:] = 从 split 到末尾
early_gap = gap[:split].mean()     # 前 2/3 的平均 gap
late_gap  = gap[split:].mean()     # 后 1/3 的平均 gap

# 后期 gap > 前期的 1.5 倍 → 过拟合迹象
overfit = late_gap > early_gap * 1.5

# ═══ Loss 下降趋势（最简单的判断：最后 < 第一 = 在降）═══
# [0] = 第一个元素（第一轮）, [-1] = 最后一个（最后一轮）
box_drop = train_box[-1] < train_box[0]    # box_loss 降了吗
cls_drop = train_cls[-1] < train_cls[0]    # cls_loss 降了吗
dfl_drop = train_dfl[-1] < train_dfl[0]    # dfl_loss 降了吗

# ═══════════════════════════════════════════════════════════
# 第 3 步：画 6 张诊断图
# ═══════════════════════════════════════════════════════════

# subplots(2行, 3列): 创建 2×3 = 6 个子图的画布
#   返回: fig=整张画布, axes=6 个子图对象（按行列访问）
#   figsize=(宽18英寸, 高10英寸): 控制画布大小
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# suptitle: 整张图的顶部大标题
fig.suptitle(
    f"训练诊断报告 | best mAP50={mAP50[best_idx]:.3f} (epoch {best_epoch}) | {error_tag}",
    fontsize=14, fontweight="bold")

# ═══════════════════════
# 图 1（左上）: 3 种 Loss（train + val）
# ═══════════════════════
ax = axes[0, 0]   # 第 0 行第 0 列

# 一个循环画 train（实线），一个循环画 val（虚线）
# alpha=0.7 = 70% 不透明度
# 用元组列表 [(数据,名字,颜色), ...] 循环画，比写 6 个 plot 简洁
for y, name, c in [(train_box, "train_box_loss", "blue"),
                    (train_cls, "train_cls_loss", "green"),
                    (train_dfl, "train_dfl_loss", "red")]:
    ax.plot(epochs, y, color=c, label=name, alpha=0.7, lw=1.5)

for y, name, c in [(val_box, "val_box_loss", "blue"),
                    (val_cls, "val_cls_loss", "green"),
                    (val_dfl, "val_dfl_loss", "red")]:
    ax.plot(epochs, y, color=c, linestyle="--", alpha=0.7, lw=1.5)
    # linestyle="--" = 虚线，一眼区分 train/val

ax.set_title("3 种 Loss (实线=train, 虚线=val)")
ax.legend(fontsize=7)    # 图例字体小一点，6 条线挤得下
ax.grid(alpha=0.3)       # 半透明网格

# ═══════════════════════
# 图 2（中上）: Precision + Recall 趋势
# ═══════════════════════
ax = axes[0, 1]

# lw=2 = linewidth=2（比默认稍粗）
ax.plot(epochs, precision, "b-", label=f"Precision ({p_final:.2f})", lw=2)
ax.plot(epochs, recall,    "r-", label=f"Recall ({r_final:.2f})",    lw=2)

# axhline: 画水平参考线，y=0.7 是"及格线"
#   ls=":" = linestyle=":"（点线）
ax.axhline(0.7, color="gray", ls=":")

ax.legend()
ax.grid(alpha=0.3)
ax.set_title("P/R 趋势")
# 观察：P 和 R 的趋势通常相反——一个涨另一个就跌
#   P 高 = 框得准但框得少, R 高 = 框得多但有乱框的

# ═══════════════════════
# 图 3（右上）: mAP 曲线
# ═══════════════════════
ax = axes[0, 2]

ax.plot(epochs, mAP50,    "g-",     label=f"mAP50 ({mAP50[best_idx]:.3f})", lw=2)
ax.plot(epochs, mAP50_95, "orange", label=f"mAP50-95 ({mAP50_95[best_idx]:.3f})", lw=2)

# axvline: 画竖直标记线，标出最佳 epoch
ax.axvline(best_epoch, color="green", ls="--", alpha=0.5)
# annotate: 在图上写文字标注
ax.annotate(f"best@{best_epoch}",
            xy=(best_epoch, mAP50[best_idx]),   # 箭头指向的位置
            color="green", fontsize=9)
ax.legend()
ax.grid(alpha=0.3)
ax.set_title("mAP 曲线")

# ═══════════════════════
# 图 4（左下）: 过拟合诊断
# ═══════════════════════
ax = axes[1, 0]

# 把三个 loss 加在一起 → 总 loss（train 和 val 各一条线）
train_tot = train_box + train_cls + train_dfl
val_tot   = val_box + val_cls + val_dfl

ax.plot(epochs, train_tot, "b-", label="train total", lw=2)
ax.plot(epochs, val_tot,   "r-", label="val total",   lw=2)

# fill_between: 把两条线之间的区域填上颜色
#   红色区域 = val 高出 train 的部分 = 过拟合的量
#   alpha=0.15 = 填色很淡，不遮挡曲线
ax.fill_between(epochs, train_tot, val_tot, alpha=0.15, color="red")

ax.set_title(f"过拟合诊断 (gap 早期={early_gap:.2f} → 后期={late_gap:.2f})")
ax.legend()
ax.grid(alpha=0.3)

# 如果有过拟合 → 在图上标注出来
if overfit:
    # text(x坐标, y坐标, 文字, ...)
    ax.text(epochs[len(epochs)//2], val_tot.max(),
            "⚠️ 过拟合", fontsize=12, color="red")

# ═══════════════════════
# 图 5（中下）: P-R 散点图（每个 epoch 一个点）
# ═══════════════════════
ax = axes[1, 1]

# scatter: 画散点图
#   x=Recall, y=Precision — 每个点是一个 epoch
#   c=epochs — 点的颜色按 epoch 渐变
#   cmap="viridis" — 颜色映射（蓝色→绿色→黄色，epoch 从小到大）
#   s=30 — 点的大小
ax.scatter(recall, precision, c=epochs, cmap="viridis", s=30)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"P-R 分布 (颜色=epoch, 最终 R={r_final:.2f} P={p_final:.2f})")
ax.grid(alpha=0.3)
# 观察：如果点从左下往右上走 → P 和 R 同步提升（理想）
#       如果点往左上走 → P 在涨但 R 在跌（模型变保守了）

# ═══════════════════════
# 图 6（右下）: Loss 下降率（柱状图）
# ═══════════════════════
ax = axes[1, 2]

# barh: 画水平柱状图 (horizontal bar)
#   第一列 = Y 轴标签（三个 loss 名字）
#   第二列 = 柱子的长度 = (最后-第一)/第一 = 相对下降率
#   负数 = loss 变大了（不是好事）
#   例如 (0.8 - 1.5)/1.5 = -0.47 ≈ 下降了 47%
rates = [
    (train_box[-1] - train_box[0]) / train_box[0],  # box_loss 下降率
    (train_cls[-1] - train_cls[0]) / train_cls[0],  # cls_loss 下降率
    (train_dfl[-1] - train_dfl[0]) / train_dfl[0],  # dfl_loss 下降率
]
ax.barh(["box_loss", "cls_loss", "dfl_loss"], rates,
        color=["blue", "green", "red"])

ax.set_title(f"Loss 下降率 (box={box_drop} cls={cls_drop} dfl={dfl_drop})")
ax.axvline(0, color="gray")   # x=0 的竖线（往左=变差，往右=变好）
ax.grid(alpha=0.3, axis="x")  # 只画纵向网格

# tight_layout(): 自动调子图间距，防止标题和标签重叠
plt.tight_layout()
# show(): 弹出桌面窗口显示 6 张图
# ⚠️ 代码会暂停在这里，等你关了窗口才继续
plt.show()

# ═══════════════════════════════════════════════════════════
# 第 4 步：终端打印文字诊断报告
# ═══════════════════════════════════════════════════════════

# 预先算好几个复合判断值（避免 f-string 里写太复杂的表达式）
overfit_msg = f"⚠️ 过拟合 (后期 gap 是前期的 {late_gap/early_gap:.1f} 倍)" if overfit else "✅ 正常收敛"

# 拼出每行建议
suggestions = []
if overfit:
    suggestions.append("  → 过拟合: 加大 weight_decay 或加强数据增强")
if r_final < 0.6:
    suggestions.append("  → 漏检多: 提高 imgsz 或降低 conf 阈值")
if p_final < 0.6:
    suggestions.append("  → 误检多: 加入负样本(背景图) 或提高 conf 阈值")
if p_final < 0.6 and r_final < 0.6:
    suggestions.append("  → 模型还没学好: 增加 epochs 或检查数据标注")
suggestion_text = "\n".join(suggestions) if suggestions else "  (无需特别调整)"

print(f"""
{'=' * 60}
📋 训练诊断报告
{'=' * 60}
总轮次: {len(epochs)}    最佳 mAP50: {mAP50[best_idx]:.4f} (epoch {best_epoch})
最终 mAP50: {mAP50[-1]:.4f}   mAP50-95: {mAP50_95[-1]:.4f}

── 误检/漏检 ──
  最终 P={p_final:.3f}  R={r_final:.3f} → {error_tag}

── 过拟合 ──
  train-val gap: 早期平均值 {early_gap:.3f} → 后期平均值 {late_gap:.3f}
  诊断: {overfit_msg}

── Loss 趋势 ──
  box_loss: {'✅ 下降' if box_drop else '⚠️ 不降或反弹'} (控制框的位置)
  cls_loss: {'✅ 下降' if cls_drop else '⚠️ 不降或反弹'} (控制分类)
  dfl_loss: {'✅ 下降' if dfl_drop else '⚠️ 不降或反弹'} (控制边界精调)

── 调参建议 ──
{suggestion_text}
{'=' * 60}
""")

print("✅ 关卡 2 YOLO版 完成 — 6 张诊断图 + 文字报告")

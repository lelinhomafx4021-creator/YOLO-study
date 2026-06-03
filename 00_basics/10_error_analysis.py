# -*- coding: utf-8 -*-
"""
10_error_analysis.py — 训练结果诊断工具
训练完 YOLO 模型后跑这个：自动读训练日志，画出 4 张诊断图，打印诊断建议。

跑法：
    python 00_basics/10_error_analysis.py

前置：你必须已经跑过 YOLO 训练（01_helmet_detect/01_train.py），
     训练结果在 runs/safety_helmet/yolo11n_baseline_v1/results.csv
"""

# ============================================================
# 第 0 步：import 需要的库（就是"工具包"）
# ============================================================

import sys                          # sys：和 Python 解释器打交道（比如装包）
from pathlib import Path            # Path：处理文件路径，跨 Windows/Linux

# --- 装 pandas（如果没装的话）---
# pandas = Python Data Analysis Library
# 核心功能：读表格数据（CSV/Excel），像 Python 里的 Excel
# pd.read_csv(文件) → 把 CSV 文件读成一个"DataFrame 表格对象"
# 然后你可以像操作 Excel 列一样操作数据：df["列名"].values
try:
    import pandas as pd             # 尝试 import
except ImportError:                 # 如果没装（抛 ImportError）
    print("需要 pandas，安装中...")
    import subprocess               # subprocess：在 Python 里跑命令行命令
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd             # 装完再 import

# --- matplotlib：画图的库 ---
# 不用 OpenCV！matplotlib 是 Python 最主流的画图工具
# 可以画折线图、柱状图、散点图、等高线...
# plt.plot(x, y)  → 画一条线
# plt.show()      → 弹出窗口显示图
import matplotlib
matplotlib.use("TkAgg")             # TkAgg = 用系统窗口弹出来显示图（不依赖浏览器）
import matplotlib.pyplot as plt     # 起个别名 plt，打字少

# --- 中文字体设置 ---
# matplotlib 默认字体不支持中文 → 标题/标签里的中文会变方框乱码
# 解决方案：指定一个系统里有的中文字体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 防止负号"-"也变乱码


# ============================================================
# 第 1 步：读 YOLO 训练日志
# ============================================================
# 这个文件怎么来的？
#   你跑 YOLO 训练时（model.train()），Ultralytics 自动在 runs/ 目录下生成
#   每训练完一个 epoch，追加一行数据到这文件里
#   不需要你手动写——它是训练过程的"黑匣子记录"
csv_path = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\results.csv")

# pd.read_csv() 做了什么？
#   1. 打开这个文件
#   2. 第一行当列名（epoch, train/box_loss, train/cls_loss, ...）
#   3. 后面每一行是一轮训练的数据
#   4. 返回一个 DataFrame 对象（可以想象成一个超大 Excel 表格）
df = pd.read_csv(csv_path)

# df 长这样（概念上）：
# ┌───────┬────────────┬────────────┬───┬─────────────┬────────────┐
# │ epoch │train/box   │train/cls   │...│metrics/mAP50│metrics/Rec │
# ├───────┼────────────┼────────────┼───┼─────────────┼────────────┤
# │   1   │   1.466    │   3.346    │...│    0.327    │   0.896    │
# │   2   │   1.265    │   2.676    │...│    0.615    │   0.961    │
# │  ...  │    ...     │    ...     │...│     ...     │    ...     │
# └───────┴────────────┴────────────┴───┴─────────────┴────────────┘
# 每一行 = 一个 epoch 的训练记录
# 每一列 = 某个指标在所有 epoch 的值

# dropna() = 删掉有空值（NaN=Not a Number）的行
# 比如 CSV 最后多了一行空行，读完是 NaN，直接扔掉
df = df.dropna()

# --- 从 df 里按列名取数据 ---
# df["列名"]  → 取那一列（返回 pandas Series 对象）
# .values     → 转成 numpy 数组（就是 Python 里的数字列表），方便后面画图
epochs = df["epoch"].values           # [1, 2, 3, ..., 50]  → 轮次数

# 训练时的 3 个 Loss（模型在训练集上的表现）
train_box = df["train/box_loss"].values   # [1.47, 1.26, 1.20, ...] 框位置损失
train_cls = df["train/cls_loss"].values   # [3.35, 2.68, 2.12, ...] 分类损失
train_dfl = df["train/dfl_loss"].values   # [1.58, 1.34, 1.30, ...] 边界精调损失

# 验证时的 3 个 Loss（模型在"没见过的新图"上的表现）
val_box = df["val/box_loss"].values
val_cls = df["val/cls_loss"].values
val_dfl = df["val/dfl_loss"].values

# 评估指标
precision = df["metrics/precision(B)"].values  # 精确率：框出来的框，有多少是对的
recall = df["metrics/recall(B)"].values        # 召回率：真实目标，找到了多少
mAP50 = df["metrics/mAP50(B)"].values          # mAP50：IoU=0.5 时的综合指标
mAP50_95 = df["metrics/mAP50-95(B)"].values    # mAP50-95：IoU 从 0.5→0.95 平均（更难）


# ============================================================
# 第 2 步：自动诊断（不看图也能出结论）
# ============================================================

# argmax() 返回"数组里最大值的位置（索引）"
# 比如 mAP50 = [0.32, 0.61, 0.55, 0.68, 0.63, ...]
# argmax() → 3（因为 0.68 最大，它在位置 3，即第 4 个 epoch）
best_epoch = mAP50.argmax()            # 第几轮 mAP50 最高
best_map = mAP50[best_epoch]           # 最高的 mAP50 值
last_map = mAP50[-1]                   # 最后第 1 轮的 mAP50（[-1] 是 Python 取最后一个的快捷写法）

# --- 过拟合判断 ---
# 逻辑：最佳 epoch 之后 mAP 不再提升 → 说明模型在训练集上继续好，验证集不跟了
best_map_epoch = int(epochs[best_epoch])  # 最佳 epoch 编号
if best_map_epoch < len(epochs) - 3:      # len(epochs)=总轮数，总轮数-3=倒数第3轮以后
    fit_status = f"⚠️ 可能过拟合：最佳 mAP 在第 {best_map_epoch} 轮，之后 {len(epochs)} 轮没再提升"
else:
    fit_status = "✅ 训练正常，mAP 持续提升中"

# --- 误检/漏检判断 ---
# 取最后一轮的 P 和 R
last_p = precision[-1]    # python 里 [-1] = 取最后一个元素
last_r = recall[-1]
# 阈值 0.6：P 或 R 低于 0.6 说明这个方向有明显问题
if last_p < 0.6 and last_r >= 0.6:
    error_type = "误检偏多（Precision 低 → FP 多 → 模型框了很多错的）"
elif last_r < 0.6 and last_p >= 0.6:
    error_type = "漏检偏多（Recall 低 → FN 多 → 模型漏掉了很多目标）"
elif last_p < 0.6 and last_r < 0.6:
    error_type = "误检漏检都多，模型还没学好"
else:
    error_type = "误检和漏检都在合理范围"


# ============================================================
# 第 3 步：画图 — matplotlib 版
# 这里不用 OpenCV！matplotlib 负责画统计图（折线图），OpenCV 负责处理图片像素
# ============================================================

# subplots(2, 2) = 创建 2 行 × 2 列 = 共 4 个子图
# fig：整张画布     axes：4 个子图对象，用 axes[行][列] 访问
# figsize=(14,10) = 画布 14 英寸宽 × 10 英寸高
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# suptitle = 整张图的大标题（super title）
fig.suptitle(f"训练结果诊断 | 最佳 mAP50={best_map:.3f} (epoch {best_map_epoch}) | {fit_status}",
             fontsize=13, fontweight="bold")
#  f"..." = f-string：Python 的格式化字符串，花括号里的变量会被替换成值
#  :.3f   = 保留 3 位小数（float format）


# --- 子图 1（左上）：Loss 曲线 ---
ax1 = axes[0, 0]      # 第 0 行第 0 列 → 左上角

# plot() 参数：(x轴数据, y轴数据, "颜色+线型", label="图例标签", alpha=透明度)
#   "b-"  = 蓝色实线
#   "g-"  = 绿色实线
#   "r-"  = 红色实线
#   "b--" = 蓝色虚线
#   alpha=0.7 = 70% 不透明度
ax1.plot(epochs, train_box, "b-",  label="train box_loss", alpha=0.7)
ax1.plot(epochs, train_cls, "g-",  label="train cls_loss", alpha=0.7)
ax1.plot(epochs, train_dfl, "r-",  label="train dfl_loss", alpha=0.7)
ax1.plot(epochs, val_box,   "b--", label="val box_loss",   alpha=0.7)
ax1.plot(epochs, val_cls,   "g--", label="val cls_loss",   alpha=0.7)
ax1.plot(epochs, val_dfl,   "r--", label="val dfl_loss",   alpha=0.7)
ax1.set_title("Loss 曲线（实线=训练，虚线=验证）")
ax1.set_xlabel("Epoch")   # X 轴标签
ax1.set_ylabel("Loss")    # Y 轴标签
ax1.legend(fontsize=6, ncol=2)   # legend = 图例，ncol=2 分两列显示
ax1.grid(True, alpha=0.3)        # grid = 网格线，alpha=0.3 半透明


# --- 子图 2（右上）：Precision + Recall ---
ax2 = axes[0, 1]          # 第 0 行第 1 列 → 右上角
ax2.plot(epochs, precision, "b-", label="Precision", linewidth=2)
ax2.plot(epochs, recall,    "r-", label="Recall",    linewidth=2)
# axhline() = 画一条水平参考线（horizontal line），y=0.7
# linestyle=":" = 点线，alpha=0.5 = 半透明
ax2.axhline(y=0.7, color="gray", linestyle=":", alpha=0.5)
ax2.set_title(f"Precision/Recall | 最终 P={last_p:.3f} R={last_r:.3f} | {error_type}")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("值")
ax2.legend()
ax2.grid(True, alpha=0.3)


# --- 子图 3（左下）：mAP 曲线 ---
ax3 = axes[1, 0]          # 第 1 行第 0 列 → 左下角
ax3.plot(epochs, mAP50,    "b-", label="mAP50",    linewidth=2)
ax3.plot(epochs, mAP50_95, "r-", label="mAP50-95", linewidth=2)
# axvline() = 画一条竖直的参考线（vertical line），标记最佳 epoch
ax3.axvline(x=best_map_epoch, color="green", linestyle="--", alpha=0.7)
# annotate() = 在图上加文字标注
ax3.annotate(f"best epoch {best_map_epoch}\nmAP50={best_map:.3f}",
             xy=(best_map_epoch, best_map), fontsize=9, color="green")
ax3.set_title("mAP 曲线")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("mAP")
ax3.legend()
ax3.grid(True, alpha=0.3)


# --- 子图 4（右下）：过拟合诊断 ---
ax4 = axes[1, 1]          # 第 1 行第 1 列 → 右下角

# 把三个训练 loss 加起来 → 一个"总训练 loss"
# 把三个验证 loss 加起来 → 一个"总验证 loss"
# numpy 数组直接 + ：对应位置相加（不是列表拼接）
train_total = train_box + train_cls + train_dfl
val_total = val_box + val_cls + val_dfl

ax4.plot(epochs, train_total, "b-", label="train total loss", linewidth=2)
ax4.plot(epochs, val_total,   "r-", label="val total loss",   linewidth=2)

# gap = 验证 loss - 训练 loss（越大说明过拟合越严重）
gap = val_total - train_total
# 取后 1/3 的 epoch，判断 gap 是否在"扩大"
# // = 整除（丢掉余数，向下取整）
split = len(epochs) * 2 // 3            # 例如 50 轮 → split=33
early_gap = gap[:split].mean()          # 前 33 轮 gap 的平均值
late_gap = gap[split:].mean()           # 后 17 轮 gap 的平均值
if late_gap > early_gap * 1.5:          # 后期 gap 比前期大 50% 以上 → 过拟合
    ax4.annotate("过拟合起点", xy=(split, gap[split]), fontsize=10, color="red")

ax4.set_title("过拟合诊断（train vs val loss）")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Total Loss")
ax4.legend()
ax4.grid(True, alpha=0.3)

# tight_layout() = 自动调整子图之间的间距，防止标签重叠
plt.tight_layout()
# show() = 弹出窗口显示图！等用户关了窗口，代码继续往下走
plt.show()


# ============================================================
# 第 4 步：在终端打印文字诊断报告
# ============================================================

# print() 里的 f"..." 是 f-string：花括号里的变量会替换成它的值
# 比如 f"总训练轮次: {len(epochs)}" → len(epochs)=50 → 输出 "总训练轮次: 50"
# :.4f = 保留 4 位小数

print("\n" + "=" * 60)                         # "=" * 60 = 画 60 个等号的分隔线
print("📋 训练诊断报告")
print("=" * 60)
print(f"总训练轮次: {len(epochs)}")              # len(epochs) = 总共有多少行数据
print(f"最佳 mAP50:  {best_map:.4f} (第 {best_map_epoch} 轮)")
print(f"最终 mAP50:  {last_map:.4f} (第 {len(epochs)} 轮)")
print()
print("── 过拟合判断 ──")
print(f"  {fit_status}")
print(f"  早期 train-val gap 均值: {early_gap:.3f}")
print(f"  后期 train-val gap 均值: {late_gap:.3f}")
print()
print("── 误检/漏检判断 ──")
print(f"  最终 Precision: {last_p:.3f}")
print(f"  最终 Recall:    {last_r:.3f}")
print(f"  诊断: {error_type}")
print()
print("── 三个 Loss 的趋势 ──")
# 下面这个是 Python 的"三元表达式"：条件成立取左边，不成立取右边
# 格式： 值1 if 条件 else 值2
# train_box[-1] = 最后一轮的 box_loss, train_box[0] = 第一轮的 box_loss
# 最后一轮 < 第一轮 → 在下降 → ✅
print(f"  box_loss trend: {'下降 ✅' if train_box[-1] < train_box[0] else '波动 ⚠️'} (控制框的位置)")
print(f"  cls_loss trend: {'下降 ✅' if train_cls[-1] < train_cls[0] else '波动 ⚠️'} (控制分类)")
print(f"  dfl_loss trend: {'下降 ✅' if train_dfl[-1] < train_dfl[0] else '波动 ⚠️'} (控制边界精度)")
print()
print("── 调参建议 ──")
if last_r < 0.6:
    print("  → Recall 低（漏检多）：优先提高 imgsz 或降低 conf 阈值")
if last_p < 0.6:
    print("  → Precision 低（误检多）：优先加负样本或提高 conf 阈值")
if late_gap > early_gap * 1.5:
    print("  → 过拟合迹象：加大 weight_decay 或减少 epochs")
if train_cls[-1] > 1.0:
    print("  → cls_loss 偏高：检查类别是否均衡，数据标注是否有错")
print("=" * 60)

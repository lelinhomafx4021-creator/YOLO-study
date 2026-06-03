import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt                              # ← pylab → pyplot
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]  # ← serif, SimHel
matplotlib.rcParams["axes.unicode_minus"] = False

csv_path = r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\results.csv"  # ← 补路径
df = pd.read_csv(csv_path).dropna()
epoch = df["epoch"].values
train_box = df["train/box_loss"].values                     # ← .value → .values
train_cls = df["train/cls_loss"].values                     # ←
train_dfl = df["train/dfl_loss"].values                     # ←
val_box = df["val/box_loss"].values
val_cls = df["val/cls_loss"].values
val_dfl = df["val/dfl_loss"].values
precision = df["metrics/precision(B)"].values
recall = df["metrics/recall(B)"].values
mAP50 = df["metrics/mAP50(B)"].values
mAP50_95 = df["metrics/mAP50-95(B)"].values                 # ← 横线→下划线, 90→95

best_idx = mAP50.argmax()
best_epoch = epoch[int(best_idx)]

p_final = precision[-1]
r_final = recall[-1]                                        # ← p_recall → r_final
if p_final < 0.6 and r_final >= 0.6:
    print("置信度过低")
elif p_final >= 0.6 and r_final < 0.6:
    print("漏检测过多")
elif r_final >= 0.6 and p_final >= 0.6:
    print("都比较正常")
else:
    print("两个都存在问题")

## 拟合度匹配
split = len(epoch) * 2 // 3
gap = (val_box + val_cls + val_dfl) - (train_box + train_cls + train_dfl)  # ← 去中括号
early_gap = gap[:split].mean()                               # ← .mean() + 前2/3是早期
late_gap = gap[split:].mean()                                # ← gap[split:gap]→gap[split:]
if late_gap > early_gap * 1.5:                               # ← 后期大→过拟合
    print("过拟合了")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))             # ← ((18,10)) → (18,10)
fig.suptitle(f"最好的mAP={mAP50[best_idx]:.3f}, 最好的epoch={best_epoch}",
             fontsize=14, fontweight="bold")                  # ← fonntweight → fontweight

ax = axes[0, 0]
for y, name, c in [(train_box, "train_box_loss", "blue"),
                    (train_cls, "train_cls_loss", "green"),
                    (train_dfl, "train_dfl_loss", "red"),
                    (val_box, "val_box_loss", "blue"),
                    (val_cls, "val_cls_loss", "green"),
                    (val_dfl, "val_dfl_loss", "red")]:       # ← train_box_loss→val, dlf→dfl
    ax.plot(epoch, y, color=c, label=name, linestyle="--", alpha=0.7, lw=1.5)

ax.set_title("3 种 Loss")                                     # ← zhognloss
ax.legend(fontsize=7)                                        # ← 加图例
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

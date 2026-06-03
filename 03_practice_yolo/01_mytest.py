from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def main():
    model = YOLO("yolo11n.pt")
    print(f"训练前的类别数：{len(model.names)}  (COCO 预训练 80 类)")

    # ── train ──
    # project_output 是 Path 对象，方便后面拼路径
    project_output = Path(r"D:\vision_algo_workspace\vision-bootcamp\03_practice_yolo")

    model.train(
        data=r"D:\vision_algo_workspace\vision-bootcamp\custom_data.yaml",
        project=str(project_output),   # ← Path → 字符串
        name="full_train",
        workers=0,                     # ← worker 加 s
        epochs=5,                      # ← epoch 加 s
        batch=8,
        imgsz=640,                     # ← 输入尺寸
        exist_ok=True,
    )

    # ── 训练产物 ──
    out_dir = project_output / "full_train"
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"{str(f.relative_to(out_dir)):<45} {size_kb:>8.0f} KB")

    # ── val ──
    metrics = model.val(
        data=r"D:\vision_algo_workspace\vision-bootcamp\custom_data.yaml",
        project=str(project_output),   # ← Path → 字符串
        name="full_train_val_01",
        workers=0,                     # ← worker 加 s
        batch=8,
        imgsz=640,
        exist_ok=True,
    )

    # ── 指标 ──
    # .mp  = mean Precision, .mr = mean Recall, .map50 = mAP@0.5
    print(f"\n整体 Precision: {metrics.box.mp:.4f}")
    print(f"整体 Recall:    {metrics.box.mr:.4f}")
    print(f"整体 mAP50:     {metrics.box.map50:.4f}")
    print(f"整体 mAP50-95:  {metrics.box.map:.4f}")

    # ── 每类指标 ──
    # ap_class_index 是正确的属性名
    # .p[i] = Precision, .r[i] = Recall, .ap50[i] = AP50
    print(f"\n{'类别':<15} {'Precision':>10} {'Recall':>10} {'mAP50':>10}")
    print("-" * 50)
    for i, cls_id in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(cls_id)]
        p = metrics.box.p[i]
        r = metrics.box.r[i]
        ap50 = metrics.box.ap50[i]
        print(f"{name:<15} {p:>10.4f} {r:>10.4f} {ap50:>10.4f}")
    # ── 每类诊断 ──
    for i, cls_id in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(cls_id)]
        precision = float(metrics.box.p[i])
        recall    = float(metrics.box.r[i])
        # 注意括号：float(值)>0.8，不是 float(值>0.8)
        if precision > 0.8:
            print(f"{name}: 精度正常 (P={precision:.2f})")
        else:
            print(f"{name}: 精度偏低 (P={precision:.2f})")
        if recall > 0.8:
            print(f"{name}: 召回正常 (R={recall:.2f})")
        else:
            print(f"{name}: 召回偏低 (R={recall:.2f})")

    # ── 加载验证 ──
    pt_path = out_dir / "weights" / "best.pt"
    model2 = YOLO(str(pt_path))              # ← Path → str，否则可能报错
    print(f"\n加载后类别一致: {model.names == model2.names}")

    # ═══════════════════════════════════════════════════════
    # 画图
    # matplotlib 画图三步走：
    #   ① 准备数据（X 轴是什么、Y 轴是什么）
    #   ② 画线 (plt.plot) / 画柱 (plt.bar) / 画散点 (plt.scatter)
    #   ③ plt.show() 弹出窗口
    # ═══════════════════════════════════════════════════════

    # pd.read_csv: 把 CSV 读成一个表格对象 (DataFrame)
    #   每一列就是一个指标，例如 df["epoch"] 就是第几轮
    #   .dropna(): 删掉有空值(NaN)的行（CSV 末尾可能有空行）
    df = pd.read_csv(out_dir / "results.csv").dropna()

    # df["epoch"] 返回一列（pandas Series）
    # .values 转成 numpy 数组（matplotlib 吃这个格式画图最快）
    epochs = df["epoch"].values    # [1, 2, 3, 4, 5]

    # subplots(2行, 2列): 创建一个画布和 2×2 = 4 个子图
    #   fig:   整张画布（可以设置大标题）
    #   axes:  4 个子图对象，用 axes[行][列] 访问
    #   figsize=(宽12英寸, 高9英寸): 控制画布大小
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # suptitle: 整张图的顶部大标题 (super title)
    #   fontweight="bold" = 加粗
    fig.suptitle("训练结果可视化", fontsize=14, fontweight="bold")

    # ═══════════════════════════════════════════
    # 左上子图：Loss 曲线
    # ═══════════════════════════════════════════
    # axes[0,0] = 第 0 行第 0 列 = 左上角
    ax = axes[0, 0]

    # plot(x轴数据, y轴数据, "颜色+线型", label="图例名字", lw=线宽)
    #   "b-"  = 蓝色实线    b = blue
    #   "g-"  = 绿色实线    g = green
    #   "b--" = 蓝色虚线    -- = 虚线
    #   label = 图例框里显示的名字
    ax.plot(epochs, df["train/box_loss"], "b-",  label="train box_loss")
    ax.plot(epochs, df["train/cls_loss"], "g-",  label="train cls_loss")
    ax.plot(epochs, df["val/box_loss"],   "b--", label="val box_loss")
    ax.plot(epochs, df["val/cls_loss"],   "g--", label="val cls_loss")

    # set_title / set_xlabel / set_ylabel: 设置标题和坐标轴标签
    ax.set_title("Loss 曲线 (实线=train, 虚线=val)")
    ax.set_xlabel("Epoch")

    # legend(): 显示图例框（label 里的文字出现在图里）
    #   fontsize=8 控制图例字体大小
    ax.legend(fontsize=8)

    # grid(alpha=0.3): 画网格线，alpha=透明度(0=全透明, 1=不透明)
    ax.grid(alpha=0.3)

    # ═══════════════════════════════════════════
    # 右上子图：Precision + Recall 趋势
    # ═══════════════════════════════════════════
    ax = axes[0, 1]   # 第 0 行第 1 列 = 右上角

    # lw=2 = linewidth=2，线粗一些
    ax.plot(epochs, df["metrics/precision(B)"], "b-", label="Precision", lw=2)
    ax.plot(epochs, df["metrics/recall(B)"],    "r-", label="Recall",    lw=2)

    # axhline: 画一条水平参考线 (horizontal line)
    #   y=0.7 → 在 y 轴 0.7 的位置画一条水平线
    #   ls=":" = linestyle=":"（点线），不抢眼
    ax.axhline(y=0.7, color="gray", ls=":")

    ax.set_title("P / R 趋势")
    ax.legend()
    ax.grid(alpha=0.3)

    # ═══════════════════════════════════════════
    # 左下子图：mAP 曲线
    # ═══════════════════════════════════════════
    ax = axes[1, 0]   # 第 1 行第 0 列 = 左下角

    # "g-" = 绿色实线, "orange" = 橙色（matplotlib 内置颜色名）
    ax.plot(epochs, df["metrics/mAP50(B)"],     "green",  label="mAP50",     lw=2)
    ax.plot(epochs, df["metrics/mAP50-95(B)"],  "orange", label="mAP50-95",  lw=2)

    ax.set_title("mAP 曲线")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    # ═══════════════════════════════════════════
    # 右下子图：过拟合诊断
    # ═══════════════════════════════════════════
    ax = axes[1, 1]   # 第 1 行第 1 列 = 右下角

    # 总训练 loss = box_loss + cls_loss（两个类别损失加起来）
    # pandas 的列可以直接 +：对应位置相加
    train_total = df["train/box_loss"] + df["train/cls_loss"]
    val_total   = df["val/box_loss"]   + df["val/cls_loss"]

    ax.plot(epochs, train_total, "b-", label="train total", lw=2)
    ax.plot(epochs, val_total,   "r-", label="val total",   lw=2)

    # fill_between: 填充两条线之间的区域
    #   如果 val 线在 train 线上方且 gap 在扩大 → 过拟合
    #   alpha=0.15 → 填充色几乎透明，不遮挡曲线
    ax.fill_between(epochs, train_total, val_total, alpha=0.15, color="red")

    ax.set_title("过拟合诊断 (train vs val)")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    # tight_layout(): 自动调整子图间距，防止标签重叠
    plt.tight_layout()

    # show(): 弹出桌面窗口显示整张图
    # ⚠️ 代码会暂停在这里，等你关了窗口才继续往下执行
    plt.show()


if __name__ == "__main__":
    main()

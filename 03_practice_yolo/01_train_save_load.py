# -*- coding: utf-8 -*-
"""
关卡 1 YOLO版: 完整训练 → 产物理清 → 每类指标 → 保存加载验证

跑一遍这个文件，你会得到:
  1. 训练产物目录 (weights/, results.csv, confusion_matrix.png...)
  2. 每类的 Precision / Recall / mAP50
  3. 验证 best.pt 加载后和训练完的模型完全一样

⚠️ Windows 上 multiprocessing 必须包在 if __name__ == "__main__": 里
"""

# ── 导入库 ──
# Path: 跨平台的路径处理，Windows/Linux 用 / 分隔都能拼出正确路径
#       例如 Path("a") / "b" / "c.txt" → "a/b/c.txt"
from pathlib import Path

# YOLO: Ultralytics 的高层 API，从训练到导出全封装
from ultralytics import YOLO

# pandas: Python 的 Excel，读 CSV 文件最方便
#         pd.read_csv(文件) → df（DataFrame 表格对象）
#         df["列名"] → 取一列，.iloc[i] → 取第 i 行
import pandas as pd


def main():
    """所有代码放 main() 里，确保 Windows spawn 子进程时不会重新执行"""

    # ═══════════════════════════════════════════════════════════
    # 第 1 步：训练
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)     # "=" * 60 = 画 60 个等号，终端分隔线
    print("第 1 步: 开始训练")
    print("=" * 60)

    # YOLO("yolo11n.pt"): 加载预训练模型
    #   - 如果本地没有 yolo11n.pt → 自动从 Ultralytics 服务器下载（约 5MB）
    #   - model.names: 类别字典，例如 {0: 'person', 1: 'bicycle', ..., 79: 'toothbrush'}
    model = YOLO("yolo11n.pt")
    print(f"训练前类别数: {len(model.names)} (COCO 预训练的 80 类)")

    # 为什么用绝对路径？
    #   相对路径 "03_practice_yolo" → YOLO 自动拼成 "runs/detect/03_practice_yolo"
    #   绝对路径 "d:/.../03_practice_yolo" → YOLO 直接使用，不自动加前缀
    project_dir = Path(r"d:\vision_algo_workspace\vision-bootcamp\03_practice_yolo")

    # model.train() — 启动训练
    #   每个参数的含义：
    model.train(
        # data: YOLO 格式的数据集配置文件 (.yaml)
        #   文件内容指定了 path（数据集根目录）、train/val 图片路径、类别名称
        data=r"d:\vision_algo_workspace\vision-bootcamp\01_helmet_detect\custom_data.yaml",

        # epochs: 整个训练集看多少遍
        #   5 轮 = 每张训练图被模型看 5 遍。太少欠拟合，太多可能过拟合
        epochs=5,

        # imgsz: 输入图片尺寸（像素），图片会被缩放成 imgsz × imgsz
        #   640 = YOLO 默认值。大图(1280)小目标更准但更慢，小图(320)快但精度低
        imgsz=640,

        # batch: 每次往 GPU 送几张图一起算
        #   8 = 你的 RTX 3050 Laptop 4GB 显存能承受的。设太大会 OOM（爆显存）
        batch=8,

        # project: 训练结果存到哪个目录
        #   name 会拼在 project 下面 → project/name/
        project=str(project_dir),

        # name: 本次实验的名字
        #   和 project 拼成最终路径 → 03_practice_yolo/full_train/
        name="full_train",

        # exist_ok=True: 如果目录已存在 → 直接覆盖（不创建 full_train2, full_train3...）
        # exist_ok=False(默认): 如果已存在 → 自动加后缀变成 full_train2
        exist_ok=True,

        # workers=0: DataLoader 不开子进程
        #   Windows 上用 spawn 创建子进程开销大，设 0 让主进程自己读数据
        #   设 0 数据加载会稍慢，但免去了 Windows multiprocessing 的坑
        workers=0,
    )

    # 训练后，model.names 从 80 类 COCO 变成了你自己的 3 类
    # 检测头也被替换了（80 类输出 → 3 类输出），backbone 保留预训练权重

    # 路径拼接：Path / "dir" / "file" = "parent/dir/file"
    out_dir = project_dir / "full_train"      # 训练输出目录
    pt_path = out_dir / "weights" / "best.pt" # 最佳模型文件路径

    # ═══════════════════════════════════════════════════════════
    # 第 2 步：看清训练产物（到底训出了什么文件）
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)    # \n = 前面空一行，终端更清爽
    print("第 2 步: 训练产物目录")
    print("=" * 60)

    # out_dir.rglob("*"): 递归列出目录下所有文件（**/* 的意思）
    #   rglob = recursive glob
    #   sorted(): 按名字排序，输出整齐
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():                          # 只处理文件，跳过目录
            size_kb = f.stat().st_size / 1024    # 文件大小：字节 → KB
            # .relative_to(out_dir): 去掉前缀，只显示相对于 out_dir 的路径
            # str(): WindowsPath 对象 → 普通字符串（否则格式化会报错）
            # :<45 = 左对齐占 45 列，:>8.0f = 右对齐占 8 列，0 位小数
            print(f"  {str(f.relative_to(out_dir)):<45} {size_kb:>8.0f} KB")

    # 这一步你会看到:
    #   weights/best.pt    ← mAP 最高的那轮（部署用这个）
    #   weights/last.pt    ← 最后第 1 轮（断点续训用）
    #   results.csv        ← 每轮的 Loss/P/R/mAP（数字版）
    #   results.png        ← 自动画的曲线图
    #   confusion_matrix.png ← 混淆矩阵（看哪两类容易搞混）
    #   args.yaml          ← 训练时的全部参数（方便以后复现）

    # ═══════════════════════════════════════════════════════════
    # 第 3 步：验证 — 获取毎类指标
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("第 3 步: 运行验证 — 获取每类指标")
    print("=" * 60)

    # model.val(): 在验证集上跑评测
    #   不修改模型参数（内部有 model.eval() + no_grad()）
    #   返回 metrics 对象，metrics.box 里有所有检测指标
    metrics = model.val(
        data=r"d:\vision_algo_workspace\vision-bootcamp\01_helmet_detect\custom_data.yaml",
        batch=8,
        verbose=False,  # 不打印每张图的详细信息（终端会刷屏）
        workers=0,      # Windows 兼容
    )

    # metrics.box 里的指标：
    #   metrics.box.mp:      整体 Precision（macro-averaged）
    #   metrics.box.mr:      整体 Recall
    #   metrics.box.map50:   mAP @ IoU=0.5
    #   metrics.box.map:     mAP @ IoU=0.5:0.95（mAP50-95）
    #   metrics.box.p:       每类的 Precision 数组 [P_helmet, P_no-helmet, P_vest]
    #   metrics.box.r:       每类的 Recall 数组
    #   metrics.box.ap50:    每类的 AP50 数组
    #   metrics.box.ap_class_index: 类别索引（例如 [0, 1, 2]）
    #   metrics.names:       类别字典 {0: 'helmet', 1: 'no-helmet', ...}

    # .map = .map50-95，框架里写 map 就是 mAP50-95
    print(f"\n整体: P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}  "
          f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    # 打印每类指标表格
    print(f"\n{'类别':<15} {'Precision':>10} {'Recall':>10} {'mAP50':>10}")
    print("-" * 50)
    # enumerate: 同时拿到索引 i 和值 cls_id
    #   i=0, cls_id=0 → helmet
    #   i=1, cls_id=1 → no-helmet
    #   i=2, cls_id=2 → safety-vest
    for i, cls_id in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(cls_id)]       # 类别名
        print(f"{name:<15} "                    # :<15 = 左对齐占 15 列
              f"{metrics.box.p[i]:>10.4f} "     # :>10.4f = 右对齐占 10 列，4 位小数
              f"{metrics.box.r[i]:>10.4f} "
              f"{metrics.box.ap50[i]:>10.4f}")

    # ═══════════════════════════════════════════════════════════
    # 第 4 步：诊断每类问题（自动判断误检/漏检）
    # ═══════════════════════════════════════════════════════════
    print(f"\n每类诊断:")
    for i, cls_id in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(cls_id)]
        p = float(metrics.box.p[i])    # 转 Python 原生 float（避免 tensor 比较问题）
        r = float(metrics.box.r[i])

        # 诊断逻辑:
        #   Precision 低 (P<0.6) + Recall 正常 (R≥0.6) → 误检多（框了很多错的）
        #   Recall 低    (R<0.6) + Precision 正常 (P≥0.6) → 漏检多（漏掉了很多）
        #   两个都低 → 模型还没学好
        #   两个都达标 → 正常
        if p < 0.6 and r >= 0.6:
            diag = "⚠️ 误检偏多 (把别的认成它了)"
        elif r < 0.6 and p >= 0.6:
            diag = "⚠️ 漏检偏多 (漏掉了很多)"
        elif p < 0.6 and r < 0.6:
            diag = "❌ 识别较差"
        else:
            diag = "✅ 正常"
        print(f"  {name:<15} P={p:.2f} R={r:.2f} → {diag}")

    # ═══════════════════════════════════════════════════════════
    # 第 5 步：加载 best.pt — 验证保存正确
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("第 5 步: 重新加载模型 — 验证保存正确")
    print("=" * 60)

    # YOLO(pt_path): 加载训练好的 best.pt
    #   内部: torch.load → 读 model 对象 → 挂载类别字典 → 返回 YOLO 对象
    #   不需要指定模型结构（best.pt 里什么都有，自包含）
    model2 = YOLO(str(pt_path))
    print(f"类别: {model2.names}")
    # 验证：加载后的类别字典和训练完的内存里的一样
    print(f"类别和原来一致: {model.names == model2.names}")

    # ═══════════════════════════════════════════════════════════
    # 第 6 步：分析 results.csv — 训练趋势速览
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("第 6 步: 训练趋势速览")
    print("=" * 60)

    # pd.read_csv: 把 CSV 读成 DataFrame（一个表格对象）
    # .dropna(): 删掉有空值(NaN)的行（CSV 末尾可能有空行）
    df = pd.read_csv(out_dir / "results.csv").dropna()

    print(f"训练了 {len(df)} 个 epoch")          # len(df) = 表格总行数
    # .iloc[0] = 第一行，.iloc[-1] = 最后一行（Python 里 -1 是倒数第一个）
    # .iloc 是 pandas 的"按位置取行"方法
    print(f"box_loss: {df['train/box_loss'].iloc[0]:.3f} → "
          f"{df['train/box_loss'].iloc[-1]:.3f}")
    print(f"cls_loss: {df['train/cls_loss'].iloc[0]:.3f} → "
          f"{df['train/cls_loss'].iloc[-1]:.3f}")
    print(f"mAP50:    {df['metrics/mAP50(B)'].iloc[0]:.3f} → "
          f"{df['metrics/mAP50(B)'].iloc[-1]:.3f}")

    # .argmax(): 返回数组里最大值的位置（索引）
    # 例如 mAP50 = [0.32, 0.61, 0.55, 0.68, 0.63]
    #   → argmax() = 3（0.68 最大，在第 3 位 = epoch 4）
    best_idx = df['metrics/mAP50(B)'].argmax()
    best_map  = df['metrics/mAP50(B)'].iloc[best_idx]        # 最高的 mAP50 值
    best_ep   = int(df['epoch'].iloc[best_idx])              # 那是第几轮

    print(f"\n最佳 mAP50: {best_map:.3f} (epoch {best_ep})")
    print(f"最终 P={df['metrics/precision(B)'].iloc[-1]:.3f}  "
          f"R={df['metrics/recall(B)'].iloc[-1]:.3f}")

    print("\n✅ 关卡 1 YOLO版 完成")


# ── Windows 必需的 __main__ 保护 ──
# 直接运行脚本: __name__ == "__main__" → True → 执行 main()
# 被 spawn 子进程 import: __name__ == "03_practice_yolo.01_train_save_load" → False
#   → main() 不执行，子进程只加载依赖然后去干活，不会递归创建新子进程
if __name__ == "__main__":
    main()

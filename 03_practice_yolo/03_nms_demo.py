# -*- coding: utf-8 -*-
"""
关卡 3 YOLO版: NMS IoU 阈值 + conf 阈值 双参数对比

做什么: 同一张图，用 4 组不同的 NMS 参数跑推理
       然后把 4 张结果横着拼成一张大图，一目了然
学什么: 改 iou 和 conf 参数 → 检测结果怎么变
"""

import cv2                  # 读图、画框、显示窗口
import numpy as np           # 造空白图（兜底用）
from ultralytics import YOLO
from pathlib import Path


def main():
    # ═══ 1. 加载模型 ═══
    pt = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    if not pt.exists():
        pt = "yolo11n.pt"    # 没有 best.pt → 用 COCO 预训练的兜底
    model = YOLO(str(pt))

    # ═══ 2. 准备测试图 ═══
    # cv2.imread: 从硬盘读一张图片，返回 numpy 数组 (H, W, 3)
    #   BGR 格式（不是 RGB！OpenCV 默认 BGR）
    #   如果文件不存在 → 返回 None
    img = cv2.imread("result.jpg")

    if img is None:
        # ── 兜底：没找到图 → 造一张纯黑背景 ──
        # np.zeros((高, 宽, 三通道)) → 全是 0 = 纯黑
        # dtype=np.uint8 = 每个像素值 0-255（8 位无符号整数）
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        # cv2.putText(图片, 文字, 位置(x,y), 字体, 字号, 颜色(BGR), 粗细)
        cv2.putText(img, "No Image", (200, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ═══ 3. 4 组不同参数，逐个跑 ═══
    # 列表里每个元组：(标签, iou阈值, conf阈值)
    #   iou=0.3: 重叠 >30% 就删 → 删最狠，框最少
    #   iou=0.5: 默认值，重叠 >50% 才删
    #   iou=0.7: 重叠 >70% 才删 → 最宽松，框最多
    #   conf=0.5: 置信度低于 50% 的框不显示 → 过滤更严
    configs = [
        ("iou=0.3 conf=0.25", 0.3, 0.25),
        ("iou=0.5 conf=0.25", 0.5, 0.25),
        ("iou=0.5 conf=0.50", 0.5, 0.50),
        ("iou=0.7 conf=0.25", 0.7, 0.25),
    ]

    # ── 循环跑 4 次，存结果 ──
    results_list = []        # 列表里每个元素是 (标签, 推理结果)
    for label, iou_val, conf_val in configs:
        # model.predict: 推理
        #   iou=  → NMS 的 IoU 阈值
        #   conf= → 置信度过滤阈值（低于这个的不显示）
        #   verbose=False → 不刷屏
        r = model.predict(img, conf=conf_val, iou=iou_val, verbose=False, workers=0)
        results_list.append((label, r))

    # ═══ 4. 把 4 张结果横着拼成一张大图 ═══
    # img.shape = (高度, 宽度, 通道数)
    # [:2] 取前两个 → (高度, 宽度)
    h, w = img.shape[:2]

    # np.zeros((高, 宽×4, 3)): 画布 = 原图 4 倍宽
    #   (640, 2560, 3) 的纯黑画布
    stack = np.zeros((h, w * 4, 3), dtype=np.uint8)

    # enumerate: 同时拿索引 i 和值 (label, r)
    for i, (label, r) in enumerate(results_list):
        # r[0].plot(): 把检测框画在图上，返回画好框的图片
        #   YOLO 内置方法，一行自动画框 + 写类别名 + 置信度
        annotated = r[0].plot()

        # len(r[0].boxes): 这一轮检测到几个框
        n = len(r[0].boxes)

        # cv2.putText: 在图片上写文字
        cv2.putText(annotated, f"{label}  {n}框",
                    (5, 25),                         # 左上角位置
                    cv2.FONT_HERSHEY_SIMPLEX,        # 字体
                    0.7,                             # 字号
                    (0, 255, 0),                     # 颜色 BGR = 绿色
                    2)                               # 粗细

        # 把这张结果图贴到大画布的第 i 列
        # stack[ 行范围 , 列范围 ]
        #   行: 全部 (0:h)
        #   列: i*w 到 (i+1)*w  — 第 i 列
        stack[:, i * w:(i + 1) * w] = annotated

    # cv2.imshow(窗口名, 图片): 弹出窗口显示
    #   窗口宽度 = w*4 像素，你的屏幕大概率能显示全
    cv2.imshow("NMS + Conf 双参数对比 — 同一张图 4 种参数", stack)

    # ═══ 5. 终端打印每组的检测数 ═══
    print("\n同一张图，不同参数 → 检测框数:")
    for label, r in results_list:
        # r[0].boxes.conf: 每个检测框的置信度（tensor）
        # .cpu().numpy(): GPU → CPU → numpy 数组（方便打印）
        if len(r[0].boxes) > 0:
            confs = r[0].boxes.conf.cpu().numpy()
            print(f"  {label:<22}: {len(confs):2d} 框  最高置信度 {confs.max():.3f}")
        else:
            print(f"  {label:<22}:  0 框（没有检测到任何目标）")

    # ═══ 6. 补充知识：mAP 是怎么算出来的 ═══
    print(f"""
{'─' * 55}
mAP 计算逻辑（你调 iou/conf 就是在影响这个）:

  AP（Average Precision）:
    对一个类别，改 conf 阈值从 0.1 到 0.9
    → 每个阈值算一组 (Precision, Recall)
    → 画成 P-R 曲线
    → 曲线下方的面积 = AP

  mAP（mean Average Precision）:
    所有类别的 AP 取平均值

  mAP50:
    IoU 阈值固定为 0.5（框和真值重叠 >50% 就算对）
    → 比较"宽松"，分数偏高

  mAP50-95:
    IoU 从 0.5 到 0.95，步长 0.05，取 10 个点的平均
    → 比较"严格"，分数偏低
    → mAP50-95 比 mAP50 低很多 = 框的位置还不够准

  你刚刚改的 iou 参数 → 影响推理时的 NMS
  和 mAP 计算里的 IoU 是两个不同的概念！
    NMS iou: 两个预测框之间的重叠阈值（去重用）
    mAP 的 IoU: 预测框和标注框之间的重叠阈值（打分用）
{'─' * 55}
""")

    # cv2.waitKey(0): 等待用户按任意键
    #   参数 0 = 无限等待，直到你按键盘
    #   如果传 1 = 等 1 毫秒就继续（视频循环用）
    cv2.waitKey(0)
    # cv2.destroyAllWindows(): 关闭所有 OpenCV 弹窗
    cv2.destroyAllWindows()
    print("✅ 关卡 3 YOLO版 完成")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
关卡 6 基础版: ByteTrack 多目标跟踪 — 每帧都检测，最简单版本

═══════════════════════════════════════════════════════════════════════════════
在看这段代码之前，先理解一个概念
═══════════════════════════════════════════════════════════════════════════════

  你之前跑的所有脚本，都是"每帧独立检测":
    帧1 → model.predict() → 3个框 (不知道框A和框B是不是同一人)
    帧2 → model.predict() → 3个框 (不知道框B和框C是不是同一人)
    帧3 → model.predict() → 2个框 (不知道谁走了、谁还在)

  跟踪做的事情不一样:
    帧1 → model.track()  → 3个框 (ID=1, ID=2, ID=3)
    帧2 → model.track()  → 3个框 (ID=1, ID=2, ID=3) ← 认出是同一批人！
    帧3 → model.track()  → 2个框 (ID=1, ID=3)       ← ID=2离开了

  核心区别: predict() 返回的框没有 ID，track() 返回的框带 ID。
           带 ID 意味着"我知道这个框和上一帧那个框是同一个目标"。

═══════════════════════════════════════════════════════════════════════════════
和 06_bytetrack_v2.py 的关系
═══════════════════════════════════════════════════════════════════════════════

  本文件 (基础版):
    - 每帧都跑 model.track()  → 每帧都是真检测，没有跳帧
    - 没有卡尔曼滤波          → 不维护速度、不预测位置
    - 只做一件事: 跟踪 + 记录每个 ID 的生命周期
    - 代码 ~70 行，是这个系列最简单的跟踪脚本

  v2 (升级版):
    - 跳帧: 检测帧 track()，跳过帧卡尔曼 predict() 猜位置
    - 手写 BoxKalman 类      → 维护 [cx,cy,w,h,vx,vy] 6维状态
    - 比基础版多了 ~200 行卡尔曼代码

  学习顺序: 先彻底搞懂这个基础版，再去看 v2。不要跳过这一版。

═══════════════════════════════════════════════════════════════════════════════
ByteTrack 在背后做了什么 (不需要你写，但需要知道)
═══════════════════════════════════════════════════════════════════════════════

  每帧 model.track() 内部发生的三件事:
    1. 检测: model.predict() 找出这一帧所有框
    2. 预测: 用 ByteTrack 内置卡尔曼猜"上一帧的每个 ID 这一帧会在哪"
    3. 匹配: 用匈牙利算法把"这一帧的检测框"和"预测位置"配对
       → 配对成功 = ID 延续 (还是同一个人)
       → 检测多出 = 新 ID    (新人进场)
       → ID 没配上 = 暂时保留 (可能被遮挡了)

  ByteTrack 的独特之处: 低分框 (conf 0.3~0.5) 不扔，也拿去匹配。
    其他跟踪器: 低分直接扔 → 人被遮挡时置信度下降 → ID 丢了
    ByteTrack:   低分也匹配 → 人半遮挡时还能保持 ID

  这些你不需要自己实现，ultralytics 的 model.track() 全帮你做了。

跑法: python 03_practice_yolo/06_bytetrack.py
按键: q=退出
"""

import cv2, time                     # cv2: 摄像头+画图  time: 计时算FPS
from collections import defaultdict  # 自动创建字典的默认值 (省掉 if key not in dict)
from ultralytics import YOLO         # YOLO 模型 (训练/检测/跟踪/导出)
from pathlib import Path             # 路径处理 (跨平台，Windows/Linux 都能用)


def main():
    # ═════════════════════════════════════════════════════════════════════════
    # 1. 加载模型
    # ═════════════════════════════════════════════════════════════════════════
    # 优先用自己训练的安全帽模型 (best.pt = 验证集上效果最好的权重)
    # 找不到就用官方预训练的 yolo11n.pt (能检测 COCO 的 80 类，包括人)
    pt = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    if not pt.exists():
        pt = "yolo11n.pt"
    model = YOLO(str(pt))  # 加载模型 → model 对象后续调用 track()/predict()/val()

    # ═════════════════════════════════════════════════════════════════════════
    # 2. 打开摄像头
    # ═════════════════════════════════════════════════════════════════════════
    source = 0                         # 0 = 默认摄像头 (笔记本自带那个)
    cap = cv2.VideoCapture(source)     # 创建视频捕获对象
    if not cap.isOpened():
        print("摄像头打不开")
        return                         # 打不开就直接退出

    # ═════════════════════════════════════════════════════════════════════════
    # 3. 初始化跟踪状态变量
    # ═════════════════════════════════════════════════════════════════════════

    # id_history: 记录每个 ID 在哪几帧出现过
    #   defaultdict(list) = 字典，但访问不存在的 key 时自动创建空列表
    #   不用手动写: if tid not in id_history: id_history[tid] = []
    #
    #   示例数据 (跑完后):
    #     id_history = {
    #       1: [1, 2, 3, 4, 5, ..., 100],    ← ID=1 从第1帧到第100帧都在
    #       2: [50, 51, 52, ..., 80],         ← ID=2 第50帧出现，第80帧消失
    #       3: [90, 91, 92, ..., 200],        ← ID=3 第90帧才出现
    #     }
    #   从这些数据可以算出: 总共出现了多少不同的人 (len(id_history))
    #                    每个人待了多久 (len(frames))
    #                    当前活跃的有哪些 (最近5帧还出现过的)
    id_history = defaultdict(list)

    fps_smooth = 0.0    # EMA 平滑后的 FPS (指数移动平均: 0.9×旧 + 0.1×新)
    frame_count = 0     # 全局帧计数器，从 0 开始，每帧 +1

    print("ByteTrack 跟踪 | 按 q 退出\n")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. 主循环 — 每帧做一次
    # ═════════════════════════════════════════════════════════════════════════
    while cap.isOpened():              # 摄像头还开着就一直循环
        ret, frame = cap.read()        # 从摄像头读一帧
        if not ret:                    # ret=False = 读失败了 (摄像头断了/视频放完了)
            break                      # 退出循环

        frame_count += 1               # 帧号 +1 (第一帧从1开始)

        # ── 4a. 核心: 跟踪 (不是检测!) ──────────────────────────────────
        t0 = time.time()               # 开始计时 (算这一帧用了多久)

        # model.track() vs model.predict():
        #   predict()  → 返回框的坐标 (xyxy)、置信度 (conf)、类别 (cls)
        #   track()    → 返回上面全部，外加 boxes.id (每框一个身份编号)
        #
        # 参数说明:
        #   frame       → 输入图像 (BGR格式，OpenCV 直接读出来的)
        #   persist=True → 关键!! 某帧没检测到的 ID 暂时保留，不立刻丢弃
        #                   设为 False 的话，只要一帧没检测到，ID 就永远消失了
        #                   例: 人走到树后面被挡了2帧，persist=True → ID 还在
        #                       persist=False → ID 立刻没了，再出现变成新ID
        #   tracker="bytetrack.yaml" → 使用 ByteTrack 算法 (而不是默认的 BoT-SORT)
        #   conf=0.25   → 置信度阈值: 低于 0.25 的检测直接丢弃
        #   verbose=False → 不打印 ultralytics 内部日志 (干净)
        #   workers=0    → Windows 上必须设为 0 (单进程)，否则可能报 spawn 错误
        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=0.25, verbose=False, workers=0)

        # results[0].plot(): ultralytics 内置画图函数
        #   在原始帧上画: 框 (bounding box) + ID 标签 + 类别名 + 置信度
        #   返回的是画好后的图像 (numpy array)，可以直接 imshow 显示
        annotated = results[0].plot()

        # ── 4b. 计算 FPS (带 EMA 平滑) ─────────────────────────────────
        t = time.time() - t0           # 这一帧的处理时间 (秒)
        # EMA 平滑公式: fps = α×旧值 + (1-α)×新值
        #   α=0.9 → 旧值权重很大 → 变化平缓，不会因为某帧卡一下 FPS 就猛掉
        #   max(t, 0.001) → 防止除以 0 (如果一帧快到 0 秒，就会报错)
        fps_smooth = 0.9 * fps_smooth + 0.1 / max(t, 0.001)

        # ── 4c. 记录每个 ID 在哪些帧出现过 ─────────────────────────────
        # results[0].boxes.id 是 pytorch tensor，比如:
        #   tensor([1., 2., 1., 3.])
        #   → 这一帧有 4 个框，分别属于 ID=1, ID=2, ID=1, ID=3
        #   → ID=1 有两个框 (可能是两个人都戴了安全帽)
        #
        # 注意: 如果这一帧什么都没检测到，boxes.id 是 None!
        ids = results[0].boxes.id

        if ids is not None:            # 有检测到东西
            # ids.int().tolist(): 把 tensor 转成 Python 整数列表
            #   tensor([1., 2., 1., 3.]) → [1, 2, 1, 3]
            for tid in ids.int().tolist():
                # 记录: ID=tid 在 frame_count 这一帧出现过
                # defaultdict 的特性: tid 不存在时自动创建空列表 []
                id_history[tid].append(frame_count)

            # n_tracked = 这一帧跟踪到的目标数量
            #   为什么用 set()? 因为同一个 ID 可能出现多次 (多个框同属一个ID)
            #   len(set([1, 2, 1, 3])) = 3  (ID=1 只算一次)
            #   len([1, 2, 1, 3]) = 4       (不做去重会多算)
            n_tracked = len(set(ids.int().tolist()))
        else:
            # 这一帧什么都没检测到 → 跟踪数为 0
            n_tracked = 0

        # ── 4d. 在画面上显示信息 ───────────────────────────────────────
        # 左上角显示: 当前跟踪数 + FPS
        #   putText(图像, 文字, 位置, 字体, 字号, 颜色, 线宽)
        cv2.putText(annotated, f"Tracked:{n_tracked} FPS:{fps_smooth:.0f}",
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── 4e. 每 30 帧打印一次统计信息 ──────────────────────────────
        # 不需要每帧都打印 (太刷屏了)，30帧一次刚好
        if frame_count % 30 == 0:
            # 计算活跃 ID 数: 最近 5 帧内还出现过的 ID
            #   any(f > frame_count-5 for f in frames[-5:])
            #     frames[-5:]  = 这个 ID 最近 5 次出现时的帧号
            #     只要有一次在 frame_count-5 之后 → 活跃
            #
            #   例: ID=1 的 frames = [1, 2, 3, 4, 5, ..., 100]
            #       当前 frame_count=100
            #       frames[-5:] = [96, 97, 98, 99, 100]
            #       100 > 100-5=95 → True → 活跃
            #
            #   例: ID=2 的 frames = [10, 11, 12]
            #       当前 frame_count=100
            #       frames[-5:] = [10, 11, 12]
            #       12 > 95? → False → 不活跃 (早就离开了)
            active = sum(1 for tid, frames in id_history.items()
                        if any(f > frame_count - 5 for f in frames[-5:]))

            # 打印: 帧号、当前跟踪数、FPS、活跃ID数、历史总ID数
            print(f"Frame{frame_count:5d} tracked={n_tracked:3d} "
                  f"FPS={fps_smooth:5.1f}  active={active:3d} "
                  f"total_id={len(id_history):4d}")

        # ── 4f. 显示画面 + 键盘控制 ────────────────────────────────────
        cv2.imshow("ByteTrack", annotated)  # 显示带框和 ID 的画面

        # waitKey(1): 等 1 毫秒，同时检测键盘输入
        #   & 0xFF: 取低 8 位 (某些系统返回 16 位，取低 8 位统一处理)
        #   ord('q') = 113 = 'q' 键的 ASCII 码
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break                         # 按 q 退出

    # ═════════════════════════════════════════════════════════════════════════
    # 5. 收尾: 释放资源 + 输出统计
    # ═════════════════════════════════════════════════════════════════════════
    cap.release()               # 释放摄像头 (别的程序才能用)
    cv2.destroyAllWindows()     # 关闭所有 OpenCV 窗口

    # 统计总结
    #   len(id_history) = 总共出现了多少个不同的 ID (去重后的)
    #   sum(len(v) for v in id_history.values()) = 所有 ID 出现的总帧次数
    print(f"""
统计: 总帧={frame_count}  不同ID数={len(id_history)}
每个ID平均 {sum(len(v) for v in id_history.values()) // max(len(id_history), 1)} 帧

ByteTrack: 高分框→匹配 | 低分框→不丢弃 | 卡尔曼→预测 | 匈牙利→配对
""")
    print("✅ 关卡 6 YOLO版 完成")


if __name__ == "__main__":
    main()

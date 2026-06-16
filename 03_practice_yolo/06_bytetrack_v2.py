# -*- coding: utf-8 -*-
"""
关卡 6 升级版: ByteTrack + 卡尔曼预测 = 跳帧也带滑动框

═══════════════════════════════════════════════════════════════════════════════
核心思路
═══════════════════════════════════════════════════════════════════════════════

  检测帧 (每 skip 帧跑一次):  真检测 + 纠正卡尔曼 → 画真框
  跳过帧 (其余所有帧):       不跑检测，卡尔曼纯预测 → 画预测框

  效果: 每一帧都有框，但只有检测帧跑了 model.track()，跳过帧省了算力。

  代价: 跳过帧的框是"猜"的。目标突然变速/变向时，预测框会滞后甚至漂移。
        skip 越大风险越高。

和 06_bytetrack.py 原版的区别:
  原版:  每帧 model.track() → 重，但每帧都准
  升级版: 检测帧 track → 跳过帧卡尔曼 predict() → 省算力，但跳过帧框可能滞后

═══════════════════════════════════════════════════════════════════════════════
卡尔曼滤波在这里做什么
═══════════════════════════════════════════════════════════════════════════════

  卡尔曼维护 6 个状态: [cx, cy, w, h, vx, vy]
    位置: 中心点 (cx,cy) + 宽高 (w,h)  → 4 个量描述"框在哪、多大"
    速度: (vx, vy)                     → 2 个量描述"框往哪移动"

  为什么用中心点而不是左上右下角? 因为中心点+宽高的运动更接近匀速直线，
  卡尔曼的线性假设更容易成立。角点的运动受宽高变化影响，不够稳定。

  检测帧: predict() 先外推 → correct() 用检测值纠正 → 速度被学到
  跳过帧: predict() 纯外推     → 不纠正              → 速度不变

  关键: 必须先 predict() 再 correct()!
        predict() 里 F·P·F' 更新协方差矩阵，建立位置-速度之间的协方差。
        没有这一步，卡尔曼增益 K 始终为零，速度永远学不到。

跑法: python 03_practice_yolo/06_bytetrack_v2.py
按键: q=退出  s=切换跳帧间隔(1→3→5→10→1循环)
"""

import cv2, time, numpy as np
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# BoxKalman: 手写卡尔曼滤波器，追踪一个目标的框
# ═════════════════════════════════════════════════════════════════════════════

class BoxKalman:
    """
    每个被追踪的目标 (一个 ID) 对应一个 BoxKalman 实例。

    内部维护:
      state: [cx, cy, w, h, vx, vy]  ← 6 维状态向量
      P:     6×6 协方差矩阵           ← 表示"对状态的不确定程度"
      missed: 连续多少帧没被检测到     ← 超过阈值就删掉这个 tracker

    两个核心操作:
      predict():  用速度外推位置 → "猜下一帧框在哪"
      correct():  用检测框纠正状态 → "检测告诉我校准一下"
    """

    def __init__(self, x1, y1, x2, y2):
        """
        用第一个检测框初始化卡尔曼状态。

        参数: 检测框的左上角 (x1,y1) 和右下角 (x2,y2)

        初始状态:
          cx, cy = 框的中心点
          w, h   = 框的宽高
          vx, vy = 0 → 刚开始追，不知道速度，设为 0
        """
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # 6 维状态: [中心x, 中心y, 宽, 高, x速度, y速度]
        self.state = np.array([cx, cy, w, h, 0.0, 0.0])

        # P = 协方差矩阵，对角线越大 = 对该状态越不确定
        # 初始: 位置不确定度=10, 速度不确定度=1000
        # 速度不确定度很大的原因: 只有一个框，不知道速度 → 允许快速学习
        self.P = np.eye(6) * 10.0          # 6×6 单位矩阵 × 10
        self.P[4, 4] = self.P[5, 5] = 1000.0  # 速度方差设大 → 卡尔曼增益大 → 学到快

        # 连续未检测计数
        self.missed = 0

    # ── predict: 用速度外推位置 ───────────────────────────────────────────
    def predict(self):
        """
        卡尔曼预测步骤: "按当前速度，框下一帧应该在哪?"

        做的事:
          1. cx += vx, cy += vy  → 匀速模型: 新位置 = 旧位置 + 速度
             (w,h 不变，因为假设物体大小不突变)
          2. 更新协方差 P         → 预测会引入不确定性，P 变大
          3. 返回预测框

        为什么 w,h 没有被速度更新? 因为 vx,vy 只描述中心的移动速度。
        框的大小变化不是匀速的（人走近会变大、转身会变小），
        用卡尔曼预测大小变化很难，所以只预测位置。

        返回: (x1, y1, x2, y2) 或 None（预测框无效时）
        """

        # 步骤1: 匀速模型 — 位置 += 速度
        #         state[0] = cx, state[4] = vx
        #         新 cx = 旧 cx + vx * 1帧
        self.state[0] += self.state[4]  # cx += vx
        self.state[1] += self.state[5]  # cy += vy
        # 注意: state[2](w) 和 state[3](h) 不变

        # 步骤2: 更新协方差 P = F·P·F' + Q
        #         F = 状态转移矩阵: 描述"旧状态怎么变到新状态"
        #             cx_new = 1*cx + 0*cy + 0*w + 0*h + 1*vx + 0*vy
        #             → F 的第0行: [1, 0, 0, 0, 1, 0]
        #         Q = 过程噪声: 每步预测都会引入一点误差
        F = np.eye(6)                    # 6×6 单位矩阵
        F[0, 4] = F[1, 5] = 1.0         # cx += vx, cy += vy

        # F·P·F': 把不确定性也按 F 传播
        #         这一步建立了位置和速度之间的协方差!
        #         没有这一步 → P 的对角线之外全是 0 → K 增益算不对 → 速度学不到
        self.P = F @ self.P @ F.T + np.eye(6) * 1e-2  # Q = 1e-2·I, 过程噪声

        # 步骤3: 从状态取出预测框（中心点+宽高 → 左上右下角）
        cx, cy, w, h = self.state[0:4]
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)

        # 防御: 如果预测出非法框（宽高非正），返回 None
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
        return None

    # ── correct: 用检测值纠正状态 ───────────────────────────────────────────
    def correct(self, x1, y1, x2, y2):
        """
        卡尔曼校正步骤: "检测器说框在这，综合一下"

        做的事:
          1. 算出卡尔曼增益 K → "预测和检测，我更信谁?"
          2. 用 K 加权融合预测和检测 → 更新状态
          3. 缩小 P → "校正后不确定性降低了"

        参数: 检测框 (x1, y1, x2, y2)

        卡尔曼增益 K 的决定因素:
          - P 大 (预测不确定) → K 大 → 更信检测
          - 测量噪声 R 大 (检测不准) → K 小 → 更信预测
          → 卡尔曼自动在两者之间找最优权重

        注意: correct() 必须在 predict() 之后调用!
              因为 predict() 里的 F·P·F' 建立了位置-速度协方差，
              correct() 里的 K 才能利用这个协方差去修正速度。
        """
        w, h = x2 - x1, y2 - y1
        # 测量向量 z: 检测器只能看到位置 (中心+宽高)，看不到速度
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, w, h])  # [cx, cy, w, h]

        # H: 观测矩阵 — 把 6 维状态映射到 4 维测量
        #     z = H @ state → 只取位置，不取速度（因为检测器测不到速度）
        H = np.zeros((4, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0

        # 步骤1: 计算残差协方差 S = H·P·H' + R
        #         R = 测量噪声协方差 (5.0·I) → "检测器也有误差"
        S = H @ self.P @ H.T + np.eye(4) * 5.0  # 4×4

        # 步骤2: 计算卡尔曼增益 K = P·H'·inv(S)
        #         K 是 6×4 矩阵: 告诉你每个测量对每个状态的修正量
        #         重点: K 的非零项会把测量信息"传递"到速度上!
        #         如果 predict() 没被调用(P 里位置-速度协方差=0)，
        #         K 的速度行就全是 0 → 速度永远学不到。
        K = self.P @ H.T @ np.linalg.inv(S)  # 6×4

        # 步骤3: 更新状态 state += K @ (z - H@state)
        #         (z - H@state) = 测量残差: "检测值和预测值差多少"
        #         K 把这个残差按比例分配到各个状态上
        self.state += K @ (z - H @ self.state)

        # 步骤4: 更新协方差 P = (I - K·H)·P
        #         校正后不确定性降低
        self.P = (np.eye(6) - K @ H) @ self.P

        # 重置 missed 计数（这一帧检测到了）
        self.missed = 0


# ═════════════════════════════════════════════════════════════════════════════
# 主函数
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. 加载模型 ───────────────────────────────────────────────────────
    # 优先用自己训练的安全帽模型，找不到就用官方预训练模型
    pt = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    if not pt.exists():
        pt = "yolo11n.pt"
    model = YOLO(str(pt))

    # ── 2. 打开摄像头 ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)  # 0 = 默认摄像头
    if not cap.isOpened():
        print("摄像头打不开")
        return

    # ── 3. 初始化跟踪状态 ─────────────────────────────────────────────────
    trackers = {}           # {ID: BoxKalman}   — 每个 ID 一个卡尔曼实例
    id_colors = {}          # {ID: (B,G,R)}     — 每个 ID 固定一个颜色
    id_history = defaultdict(list)  # {ID: [出现帧号列表]}  — 生命周期记录

    skip = 3                # 跳帧间隔: 每 3 帧检测一次
    frame_count = 0         # 全局帧计数器
    fps_smooth = 0.0        # EMA 平滑后的 FPS

    print(f"ByteTrack+Kalman skip={skip} | 检测帧=track | 跳过帧=预测框 | s切换 q退出\n")

    # ── 4. 主循环 ─────────────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 480))  # 统一尺寸
        t0 = time.time()

        # ── 4a. 判断: 检测帧 还是 跳过帧? ──────────────────────────────
        #       frame_count % skip == 1 → 检测帧 (第1,4,7,10...帧，当skip=3时)
        #       为什么要 == 1 而不是 == 0?
        #         第0帧太冷了(cap刚打开)，第1帧开始 track 更稳定。
        if frame_count % skip == 1:
            # ═════════════════════════════════════════════════════════════
            # 检测帧: 真检测 + 纠正
            # ═════════════════════════════════════════════════════════════

            # 第1步: 已有 tracker 先 predict()
            #        每个 tracker 用速度外推位置 → 预测框用于 ByteTrack 匹配
            #        ByteTrack 内部也会用自己的卡尔曼 predict，
            #        但我们也需要 predict → 这是为了后续 correct 能学到速度
            for t in trackers.values():
                t.predict()

            # 第2步: ByteTrack 检测 + ID 关联
            #        persist=True: 某帧没检测到的 ID 不会立刻丢弃
            #        tracker="bytetrack.yaml": 使用 ByteTrack 算法
            #        conf=0.25: 置信度阈值
            #        verbose=False: 不打印 ultralytics 日志
            #        workers=0: Windows 上用单进程（避免 spawn 问题）
            results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  conf=0.25, verbose=False, workers=0)

            # results[0].plot(): ultralytics 内置画图（画框+ID 标签）
            annotated = results[0].plot()
            boxes = results[0].boxes

            # 第3步: 用检测结果纠正卡尔曼
            seen_ids = set()  # 这一帧检测到的 ID 集合

            if boxes.id is not None:
                # boxes.id 是 tensor，比如 tensor([1., 2., 1., 3.])
                # 每个框对应一个 ID，同一目标多帧 ID 不变
                for i, tid in enumerate(boxes.id.int().tolist()):
                    seen_ids.add(tid)

                    # 从 xyxy tensor 取出框坐标
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                    # 记录生命周期: ID tid 在第 frame_count 帧出现过
                    id_history[tid].append(frame_count)

                    if tid not in trackers:
                        # 新 ID → 创建新的卡尔曼追踪器
                        trackers[tid] = BoxKalman(x1, y1, x2, y2)

                        # 给这个 ID 分配一个固定颜色（用于画框）
                        np.random.seed(tid * 7)  # 固定种子 → ID 相同颜色就相同
                        id_colors[tid] = (int(np.random.randint(0, 255)),
                                          int(np.random.randint(0, 255)),
                                          int(np.random.randint(0, 255)))
                    else:
                        # 已有 ID → 用检测框纠正卡尔曼
                        trackers[tid].correct(x1, y1, x2, y2)

            # 第4步: 处理失联的 tracker
            #        检测帧里没出现的 ID → missed += 1
            #        missed >= 30 → 删除（太久没出现，可能已离开画面）
            for tid in list(trackers.keys()):
                if tid not in seen_ids:
                    trackers[tid].missed += 1

            # 清理: 删掉 missed >= 30 的 tracker
            #        用 dict comprehension 只保留还没超时的
            trackers = {tid: t for tid, t in trackers.items() if t.missed < 30}

        else:
            # ═════════════════════════════════════════════════════════════
            # 跳过帧: 不跑检测，纯卡尔曼预测画框
            # ═════════════════════════════════════════════════════════════

            # 复制原始帧（不画 ultralytics 的图，因为没跑 track）
            annotated = frame.copy()

            # 对每个已知 ID，predict() 猜位置 → 直接画预测框
            for tid, tracker in trackers.items():
                pred = tracker.predict()  # 卡尔曼纯外推
                if pred is None:
                    continue
                x1, y1, x2, y2 = pred

                # 每个 ID 用固定颜色
                color = id_colors.get(tid, (0, 255, 0))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                # 标注 "(pred)" 表示这是预测框，不是检测框
                cv2.putText(annotated, f"ID:{tid}(pred)", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ── 4b. FPS + 状态显示 ─────────────────────────────────────────
        # EMA 平滑: fps = 0.9*旧值 + 0.1*新值
        # 好处: 不会因为某一帧抖动就剧烈跳变
        fps_smooth = 0.9 * fps_smooth + 0.1 / max(time.time() - t0, 0.001)

        # 判断当前帧模式标识
        mode = "DETECT" if frame_count % skip == 1 else "PREDICT"
        cv2.putText(annotated, f"{mode} FPS:{fps_smooth:.0f} IDs:{len(trackers)}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("ByteTrack + Kalman", annotated)

        # ── 4c. 键盘交互 ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # 退出
        if key == ord('s'):
            # 切换跳帧间隔: 1→3→5→10→1 循环
            # skip=1 意味着每帧都检测 = 无跳帧
            skip = {1: 3, 3: 5, 5: 10, 10: 1}.get(skip, 3)
            print(f"skip={skip}")

    # ── 5. 收尾 ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    # 统计输出
    print(f"总帧={frame_count}  不同ID={len(id_history)}")
    print("[OK] 关卡6升级版 完成")


if __name__ == "__main__":
    main()

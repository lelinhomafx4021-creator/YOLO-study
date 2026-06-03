# -*- coding: utf-8 -*-
"""
关卡 5 YOLO版: 视频检测 — 逐帧 vs 跳帧 实时对比
左右分屏: 左=每帧推理(真实负载), 右=跳帧(流畅度)
"""
import cv2, time, numpy as np
from ultralytics import YOLO
from pathlib import Path


def main():
    pt = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    if not pt.exists(): pt = "yolo11n.pt"
    model = YOLO(str(pt))

    source = 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("摄像头打不开"); return

    skip = 3; frame_count = 0
    fps_full, fps_skip = 0.0, 0.0
    last_annotated = None

    print(f"跳帧={skip} | 左=逐帧推理 | 右=跳帧复用 | 按 s 切换 | 按 q 退出")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))

        # 左半: 每帧推理
        t0 = time.time()
        r = model.predict(frame, conf=0.25, verbose=False, workers=0)
        t_full = time.time()-t0
        left = r[0].plot(); n_full = len(r[0].boxes)

        # 右半: 跳帧+旧框复用
        t1 = time.time()
        if frame_count % skip == 1:
            r2 = model.predict(frame, conf=0.25, verbose=False, workers=0)
            last_annotated = r2[0].plot()
        right = last_annotated if last_annotated is not None else frame
        t_skip = time.time()-t1

        fps_full = 0.9*fps_full + 0.1/max(t_full, 0.001)
        fps_skip = 0.9*fps_skip + 0.1/max(t_skip, 0.001)

        cv2.putText(left, f"FULL FPS:{fps_full:.0f} frames:{n_full}",
                    (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        cv2.putText(right, f"SKIP{skip} FPS:{fps_skip:.0f}",
                    (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.imshow("左=逐帧 | 右=跳帧(按s切换)", np.hstack([left, right]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s'):
            skip = {1:3,3:5,5:10,10:1}.get(skip,3)
            print(f"跳帧→{skip}")

    cap.release(); cv2.destroyAllWindows()
    print("✅ 关卡 5 YOLO版 完成")


if __name__ == "__main__":
    main()

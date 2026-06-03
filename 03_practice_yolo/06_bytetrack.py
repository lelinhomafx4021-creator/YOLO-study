# -*- coding: utf-8 -*-
"""
关卡 6 YOLO版: ByteTrack 多目标跟踪 + 行为分析
每个目标持续带 ID，统计活跃目标数和总 ID 数
"""
import cv2, time
from collections import defaultdict
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

    id_history = defaultdict(list)
    fps_smooth, frame_count = 0.0, 0
    print("ByteTrack 跟踪 | 按 q 退出\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        t0 = time.time()
        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=0.25, verbose=False, workers=0)
        annotated = results[0].plot()
        t = time.time()-t0
        fps_smooth = 0.9*fps_smooth + 0.1/max(t, 0.001)

        ids = results[0].boxes.id
        if ids is not None:
            for tid in ids.int().tolist():
                id_history[tid].append(frame_count)
            n_tracked = len(set(ids.int().tolist()))
        else:
            n_tracked = 0

        cv2.putText(annotated, f"Tracked:{n_tracked} FPS:{fps_smooth:.0f}",
                    (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if frame_count % 30 == 0:
            active = sum(1 for tid, frames in id_history.items()
                        if any(f > frame_count-5 for f in frames[-5:]))
            print(f"Frame{frame_count:5d} tracked={n_tracked:3d} FPS={fps_smooth:5.1f}  active={active:3d} total_id={len(id_history):4d}")

        cv2.imshow("ByteTrack", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

    print(f"""
统计: 总帧={frame_count}  不同ID数={len(id_history)}
每个ID平均 {sum(len(v) for v in id_history.values())//max(len(id_history),1)} 帧

ByteTrack: 高分框→匹配 | 低分框→不丢弃 | 卡尔曼→预测 | 匈牙利→配对
""")
    print("✅ 关卡 6 YOLO版 完成")


if __name__ == "__main__":
    main()

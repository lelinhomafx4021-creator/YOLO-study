import cv2
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path
import time
def main():
    model=YOLO("D:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    source=0
    cap =cv2.VideoCapture(source=source)
    if not cap.isOpened():
        print("摄像头打不开")
        return
    fps_smonth=0.0
    frame_count=0
    while cap.isOpened():
        ret,frame=cap.read
        if not ret:
            break
        frame_count+=1
        t0=time.time()
        results=model.track(frame,persist=True,tracker="bytetrack.yaml",conf=0.25,verbose=False,workers=0
        )
        annotated=results[0].plot
        t=time.time()-t0
        fps_smonth=0.9*fps_smonth+0.1/max(t,0.001)
        ids =results[0].boxes.id
        id_history={}
        if ids is not None:
            for tid in ids.int().tolist():
                id_history[tid].append(frame_count)
            n_tracked=len(ids.int().tolist())
        else:
            print("没有检测到任何的目标")
        cv2.putText(annotated,f"Trachked:{n_tracked},fps:{fps_smonth:.0f}",(5,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        if frame_count%30==0:
            active = sum(1 for tid, frames in id_history.items()
                        if any(f > frame_count - 5 for f in frames[-5:]))
            # 打印: 帧号、当前跟踪数、FPS、活跃ID数、历史总ID数
            print(f"Frame{frame_count:5d} tracked={n_tracked:3d} "
                  f"FPS={fps_smoth:5.1f}  active={active:3d} "
                  f"total_id={len(id_history):4d}")
        cv2.imshow("bytertrack",annotated)
        if cv2.waitKey(1)& 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main___":
    main()
            

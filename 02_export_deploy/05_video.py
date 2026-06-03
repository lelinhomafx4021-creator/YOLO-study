from pathlib import Path
from ultralytics import YOLO
import cv2
BASE_DIR=Path(__file__).resolve().parent.parent
video_url=BASE_DIR/"vedio"/"77b2903f4d0f839cf20375fa3ccfaa80.mp4"
model_path=BASE_DIR/"runs"/"safety_helmet"/"yolo11n_baseline_v1"/"weights"/"best.pt"
model=YOLO(model_path)
cap=cv2.VideoCapture(str(video_url))
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        print("视频播放结束或视频无法读取")
        break
    results=model(frame,conf=0.2,verbose=False)
    annotated_frame=results[0].plot()
    cv2.imshow("supper easy",annotated_frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

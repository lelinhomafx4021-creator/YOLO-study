from ultralytics import YOLO
import numbers as np
from pathlib import Path
import cv2
def main():
    pt=Path(r"D:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    model=YOLO(str(pt))
    img=cv2.imread("result.png")
    if img is None:
        img=np.zeros((640,540,3),dtype=np.uint8)
        cv.putText(img,"NO IMAGES",(200,200),
                   cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    config=[
        ""
    ]
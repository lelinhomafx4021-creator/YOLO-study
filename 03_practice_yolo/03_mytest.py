from ultralytics import YOLO
import numpy as np
from pathlib import Path
import cv2
def main():
    pt=Path(r"D:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    model=YOLO(str(pt))
    img=cv2.imread("result.jpg")
    if img is None:
        img=np.zeros((640,540,3),dtype=np.uint8)
        cv2.putText(img,"NO IMAGES",(200,200),
                   cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    config=[
        ("iou=0.3,conf=0.25",0.3,0.25),
        ("iou=0.5,conf=0.5",0.5,0.5),
        ("iou=0.7,conf=0.70",0.70,0.70)
    ]
    result_list=[]
    for label,iou,conf in config:
        r=model.predict(img,iou=iou,conf=conf,verbose=False,workers=0)
        result_list.append((label,r))
    h,w=img.shape[:2]
    stack=np.zeros((h,w*3,3),dtype=np.uint8)
    for i,(label,r) in enumerate(result_list):
        annotated=r[0].plot()
        n=len(r[0].boxes)
        cv2.putText(annotated,f"{label},{i}框",
                   (5,25),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0,255,0),
                   2)
        stack[:,i*w:(i+1)*w]=annotated
    cv2.imshow("参数对图检查nms的影响",stack)
    for label,r in result_list:
        if len(r[0].boxes)>0:
            print(f"最高置信度{label}:{r[0].boxes.conf.cpu().numpy().max()}")
        else:
            print(f"{label:<22}:0框")
    cv2.waitKey(0)
    cv2.destroyAllWindows(1)
if __name__=="__main__":
    main()
        


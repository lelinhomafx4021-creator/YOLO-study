import cv2
import time
from ultralytics import YOLO
from pathlib import Path
def main():
    pt_dir=Path(r"D:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    model=YOLO(pt_dir)  
    soucre=0
    cap=cv2.VideoCapture(source=soucre)
    if not cpa.isOpened():
        print("摄像头打不开")
        return
    skp=3
    fps_full,fps_skip=0.0,0.0
    last_annotated=None
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            print("视频放完了")
            break
        frame_count+=1 
        frame=cv2.resize(frame,(640,480))
        t0=time.time()
        r=model.predict(frame,verbose=False,conf=0.25,workers=0)
        t_full=time.time()-t0
        left=r[0].plot()
        n_full=len(r[0].boxes)
        if frame_count%3==1:
            last_annotated=left 
        right = last_annotated if last_annotated is not None else frame
        cv2.putText(left,f"FULL FPS",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(right,f"FULL FPS",(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imshow("左逐帧，右跳帧",np.hstack([left,right]))
        key=cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows
    print("两个跳帧完成")
if __name__=="__main__":
    print("完成了")
        



            

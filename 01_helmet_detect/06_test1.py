from pathlib import Path
from ultralytics import YOLO
import cv2
BASE_DIR=Path(__file__).resolve().parent.parent
model_path=BASE_DIR/"runs"/"satf"
test_img=BASE_DIR/""
output_dir=BASE_DIR
crop_dir=output_dir/"crops"
if not model_path:
    print(f"找不到模型文件：{model_path}")
    return
if not test_img:
    print("找不到测试文件：{test_img}")
    return
output_dir.mkdir(parents=True,exist_ok=True)
crop_dir.mkdir(parents=True,exist_ok=True)
print("开始图片推理")
model=YOLO(model_path)
results=model.predict(source=test_img,conf=0.25,save=False)
for result in results:
    for box in result.boxes:
        x1,x2,y1,y1=map(int,box.xyxy[0].tolist)
        score=box.conf.item()
        class_name=result.names[int(box.cls)]
        if class_name=="no-helmet":
            cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,0,255),thickness=3)
            label=f"no-helmet:{class_name}:{score}"
            cv2.putText(draw_img,label,(x1,y1),(x2,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),thickness=1)
            crop_y1=max(0,y1)
            crop_y2=max(h_u)
        if class_name=="helmet":
            cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,0,255),thickness=3)
            label=f"nhelmet:{class_name}:{score}"
            cv2.putText(draw_img,label,(x1,y1),(x2,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),thickness=1)


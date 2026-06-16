from pathlib import Path
from ultralytics import YOLO
import time,cv2,numpy as np,onnxruntime as ort,os,torch

# ONNX Runtime 找 CUDA 12 DLL
_dll_dir = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "onnxruntime" / "capi"
if _dll_dir.exists():
    for _f in os.listdir(str(_dll_dir)):
        if _f.endswith(".dll"):
            os.add_dll_directory(str(_dll_dir))

device = 0 if torch.cuda.is_available() else "cpu"

def main():
    pt_path=Path(r"D:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    model=YOLO(str(pt_path))
    onnx_model=model.export(format="onnx",imgsz=640,simplify=True,half=True)
    print(f"导出onnx{onnx_model}")
    img=cv2.imread("result.jpg")
    if img is None:
        img=np.random.randint(0,255,(640,640,3),dtype=np.uint8)
    runs=20
    print("原yolo测试")
    time_pt=[]
    for _ in range(runs):
        t0=time.time()
        results=model(img,workers=0,conf=0.3,verbose=False,device=0 if torch.cuda.is_available else "cpu")
        time_pt.append(time.time()-t0)
    time_mean=np.mean(time_pt)
    time_std=np.std(time_pt)
    print(f"平均测试时间为：{time_mean:.3f}s,标准差为：{time_std:.3f}s")
    print("onnx测试")
    omodel=YOLO(onnx_model)
    # 预热
    _ = omodel(img,workers=0,conf=0.3,verbose=False,device=device)
    time_onnx=[]
    for _ in range(runs):
        t1=time.time()
        omodel(img,workers=0,conf=0.3,verbose=False,device=device)
        time_onnx.append(time.time()-t1)
    time_mean1=np.mean(time_onnx)
    time_std1=np.std(time_onnx)
    print(f"平均测试时间为：{time_mean1:.3f}s,标准差为：{time_std1:.3f}s")
    ##保存图片
    out_dir=Path(r"D:\vision_algo_workspace\vision-bootcamp\output")
    out_chunk_dir=Path(r"D:\vision_algo_workspace\vision-bootcamp\output\chunks")
    annatated=results[0].plot()
    cv2.imwrite(str(out_dir / "annotated.jpg"),annatated)
    result=results[0]
    with open(out_dir / "boxes.txt","w") as f:
        for i in range(len(result.boxes)):
            x1,y1,x2,y2=map(int,result.boxes.xyxy[i].tolist())
            cls_id=int(result.boxes.cls[i].item())
            name=result.names[cls_id]
            conf=result.boxes.conf[i].item()
            f.write(f"坐标{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f},类别id{cls_id},类别名字{name},置信度{conf:.3f},图片{i}.jpg\n")

    for i in range(len(result.boxes)):
         x1,y1,x2,y2=map(int,result.boxes.xyxy[i].tolist())
         cls_id=int(result.boxes.cls[i].item())
         name=result.names[cls_id]
         conf=result.boxes.conf[i].item()
         crop=img[y1:y2, x1:x2]
         # 抠图后坐标重置为 (0,0)，标注上原图坐标
         label = f"{name} {conf:.2f}"
         coord = f"原图:({x1},{y1})-({x2},{y2})"
         cv2.putText(crop, label, (5, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
         cv2.putText(crop, coord, (5, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
         out_chunk_dir.mkdir(parents=True,exist_ok=True)
         cv2.imwrite(str(out_chunk_dir / f"crop_{i}_{name}_{conf:.2f}.jpg"),crop)
if __name__=="__main__":
    main()
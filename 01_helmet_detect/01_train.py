# -*- coding: utf-8 -*-
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载官方预训练模型
    model = YOLO("yolo11n.pt")
    
    # 2. 传入 yaml 配置文件开始训练，将结果输出到绝对路径中以防嵌套 Bug
    model.train(
        data=r"d:\vision_algo_workspace\vision-bootcamp\01_helmet_detect\custom_data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        device=0,
        workers=0,
        project=r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet",
        name="yolo11n_baseline_v1",
        exist_ok=True
    )
    print("🎉 训练全部完成！最好权重已保存在 runs/safety_helmet/yolo11n_baseline_v1/weights/best.pt")

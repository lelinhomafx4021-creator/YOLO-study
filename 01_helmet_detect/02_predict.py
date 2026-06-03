# -*- coding: utf-8 -*-
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 载入我们刚刚训练出来的 best.pt 权重
    model_path = r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt"
    model = YOLO(model_path)
    
    # 2. 对验证集（模型未见过的图片）进行批量推理，画框保存
    model.predict(
        source=r"d:\vision_algo_workspace\vision-bootcamp\datasets\safety_helmet\images\val",
        conf=0.50,
        save=True,
        project=r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet",
        name="predict_v1",
        exist_ok=True
    )
    print("🎉 批量测试完成！请去 runs/safety_helmet/predict_v1/ 目录下查看被画上正确彩色框的图片！")

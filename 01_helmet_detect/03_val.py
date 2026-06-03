# -*- coding: utf-8 -*-
"""
03_val.py - 期末成绩（验证集评估）指标评估脚本
计算模型在验证集（val）上的 Precision, Recall, mAP50 等核心指标，并输出分类别的诊断说明。
"""

from ultralytics import YOLO
import os

def main():
    model_path = r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt"
    data_yaml = r"d:\vision_algo_workspace\vision-bootcamp\01_helmet_detect\custom_data.yaml"

    if not os.path.exists(model_path):
        print(f"❌ 找不到最优模型权重: {model_path}，评估终止。")
        return

    model = YOLO(model_path)
    print("🚀 正在对验证集（val）数据进行指标评估...")
    
    metrics = model.val(
        data=data_yaml,
        project=r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet",
        name="val_v1",
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print("📋 工地安全帽检测项目 - 评估成绩单")
    print("=" * 65)
    print(f"  🏆 全网综合 mAP50      : {metrics.box.map50:.4f}")
    print(f"  🏆 综合精准率 (Precision): {metrics.box.mp:.4f}")
    print(f"  🏆 综合召回率 (Recall)   : {metrics.box.mr:.4f}")
    print("-" * 65)

    print(f"{'类别名称 (Class Name)':<20} | {'准确率 (P)':^10} | {'召回率 (R)':^10} | {'mAP50':^10}")
    print("-" * 65)
    for i, class_id in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(class_id)]
        p = metrics.box.p[i]
        r = metrics.box.r[i]
        ap50 = metrics.box.ap50[i]
        
        problem = "正常"
        if p < 0.70 and r >= 0.70:
            problem = "⚠️ 误检偏多"
        elif r < 0.70 and p >= 0.70:
            problem = "⚠️ 漏检偏多"
        elif p < 0.70 and r < 0.70:
            problem = "❌ 识别较差"
            
        print(f"{name:<20} | {p:^10.3f} | {r:^10.3f} | {ap50:^10.3f} | {problem}")
    print("=" * 65)

if __name__ == "__main__":
    main()

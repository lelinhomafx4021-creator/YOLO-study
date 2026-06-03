# -*- coding: utf-8 -*-
"""
05_pro_render.py - 工业级规范：基于 Pathlib 与 OpenCV 的自定义抠图与渲染脚本
此脚本展示了在大厂生产环境中，如何使用 pathlib 动态寻找路径，
并使用 OpenCV 和 NumPy 切片，将检测到的未戴安全帽人员头像单独抠图保存，做高危取证。
"""

from ultralytics import YOLO
from pathlib import Path
import cv2

def main():
    # ==========================================
    # 1. 🏆 工业级路径管理：用 pathlib 动态定位项目根目录
    # ==========================================
    # Path(__file__) 拿到当前脚本文件的绝对路径
    # .resolve() 转换为最真实的物理路径，防止相对路径在不同环境下报错
    # .parent 拿到 01_helmet_detect/ 文件夹
    # .parent.parent 拿到项目根目录 vision-bootcamp/
    BASE_DIR = Path(__file__).resolve().parent.parent

    # 动态拼接模型、输入图片和输出目录（使用优雅的 / 运算符，自动适应 Win/Linux 操作系统！）
    model_path = BASE_DIR / "runs" / "safety_helmet" / "yolo11n_baseline_v1" / "weights" / "best.pt"
    test_img = BASE_DIR / "datasets" / "safety_helmet" / "images" / "val" / "bf27c9e6-8c6c-4fa0-b13d-3c70a777a783_jpg.rf.LLpoEU3BxpwwdiBMAeJC.jpg"
    
    # 我们自定义的渲染和抠图输出总目录
    output_dir = BASE_DIR / "runs" / "safety_helmet" / "pro_render"
    crop_dir = output_dir / "crops"

    # 验证关键文件是否存在，提高程序的鲁棒性（Robustness）
    if not model_path.exists():
        print(f"❌ 找不到模型文件: {model_path}，请确认是否已跑完训练！")
        return
    if not test_img.exists():
        print(f"❌ 找不到测试图片: {test_img}，请确认数据集路径！")
        return

    # 🏆 使用 pathlib 优雅创建多层级输出文件夹（exist_ok=True 保证二次运行不崩溃）
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 2. 🚀 推理阶段：仅获取内存数据，不污染磁盘
    # ==========================================
    print(f"🎬 正在读取项目根目录: {BASE_DIR}")
    print("🎬 正在载入 YOLO 模型进行推理...")
    model = YOLO(model_path)
    
    # save=False 代表我们不需要 YOLO 默认的画框图，我们自己来接管一切
    results = model.predict(source=test_img, conf=0.25, save=False)
    result = results[0]

    # ==========================================
    # 3. 🎨 后处理与自定义渲染阶段
    # ==========================================
    # 从内存复制一份最干净的原始图像
    draw_img = result.orig_img.copy()
    h_img, w_img, _ = draw_img.shape

    print(f"📊 正在解析检测框数据，图片尺寸为: {w_img}x{h_img}...")

    # 遍历检测出来的每一个物体框
    for box in result.boxes:
        # 提取整型像素坐标、置信度以及类别名称
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]

        # 🚨 规则 A：如果是高危的“未戴安全帽 (no-helmet)”
        if class_name == "no-helmet":
            # 1. 用 OpenCV 画醒目的红色粗边框
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
            
            # 2. 写上醒目的白色大写警告文字
            label = f"DANGER: NO HELMET {score:.0%}"
            cv2.putText(draw_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)

            # 3. 📸 用 NumPy 切片把该人员头部抠图剪裁下来，独立存盘，用于罚单/高危取证！
            # 边界保护：防止坐标 y1, y2, x1, x2 发生越界（超出图像物理宽高）导致程序报错
            crop_y1 = max(0, y1)
            crop_y2 = min(h_img, y2)
            crop_x1 = max(0, x1)
            crop_x2 = min(w_img, x2)

            # numpy 切片语法：[y的起止范围, x的起止范围] (y在前，x在后)
            crop_head = result.orig_img[crop_y1:crop_y2, crop_x1:crop_x2]

            # 物理保存这张头像特写小图
            crop_save_path = crop_dir / f"crop_{x1}_{y1}.jpg"
            # cv2.imwrite 需要将 Path 对象转换为字符串格式才能正常写入
            cv2.imwrite(str(crop_save_path), crop_head)
            print(f"  📸 [高危取证已保存] ➡️ {crop_save_path.name}")

        # 🚨 规则 B：如果是合规的“戴了安全帽 (helmet)”或“穿了反光背心 (safety-vest)”
        elif class_name in ["helmet", "safety-vest"]:
            # 画温和的绿色细边框
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
            
            label = f"{class_name} {score:.0%}"
            cv2.putText(draw_img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), thickness=1)

    # ==========================================
    # 4. 💾 物理存盘：保存我们自定义画框的总结果图
    # ==========================================
    final_save_path = output_dir / "bf27c9e6_pro_result.jpg"
    cv2.imwrite(str(final_save_path), draw_img)

    print("\n" + "="*50)
    print("🎉 工业级后处理渲染全部完成！")
    print(f"👉 渲染后的全景图保存在: {final_save_path.name}")
    print(f"👉 未戴帽头像抠图保存在: {crop_dir.name}/ 目录下")
    print("="*50)

if __name__ == "__main__":
    main()

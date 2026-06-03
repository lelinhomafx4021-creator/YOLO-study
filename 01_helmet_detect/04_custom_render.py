# -*- coding: utf-8 -*-
"""
04_custom_render.py - OpenCV + YOLO 定制化后处理画框教学脚本
此脚本展示了如何关闭 YOLO 默认的自动画框（save=False），
在 Python 内存中提取边界框坐标、置信度，并使用 OpenCV 手动绘制不同颜色、粗细的框和警告标签。
"""

from ultralytics import YOLO
import cv2
import os

def main():
    # 1. 定义权重文件和测试图片路径
    model_path = r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt"
    # 我们拿一张验证集（val）里的图片来做测试
    test_img = r"d:\vision_algo_workspace\vision-bootcamp\datasets\safety_helmet\images\train\2a9922dc-e57c-44fb-9a85-51be1b69889b_jpg.rf.p7wSaFXZGaKvhGQryCPZ.jpg"
    output_dir = r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\custom_render"

    if not os.path.exists(model_path):
        print(f"❌ 找不到最优模型权重: {model_path}，请确保已经跑完训练！")
        return
    if not os.path.exists(test_img):
        print(f"❌ 找不到测试图片: {test_img}，请确认数据集路径！")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 2. 载入模型
    model = YOLO(model_path)
    print("🚀 正在载入模型并进行推理（已关闭 YOLO 默认自动画框存盘）...")

    # 🚨 注意：这里设置 save=False，不让 YOLO 默认画框存盘
    results = model.predict(source=test_img, conf=0.25, save=False)
    result = results[0]  # 取出第一张图的检测结果

    # 3. 复制一份原始图像，用来在上面手动画框（不污染原图）
    # result.orig_img 本身就是一个 OpenCV 格式的 numpy 数组
    draw_img = result.orig_img.copy()

    # 4. 开始遍历每一个检测出来的物体框
    box_count = len(result.boxes)
    print(f"📊 内存中共检测到 {box_count} 个目标，正在进行自定义后处理画图...")

    for box in result.boxes:
        # 4.1 提取矩形框的左上角和右下角像素坐标（转为整数）
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # 4.2 提取置信度分数（0.0 ~ 1.0）
        score = box.conf[0].item()
        
        # 4.3 提取类别整数 ID
        class_id = int(box.cls[0].item())
        
        # 4.4 从 result.names 字典里映射出真实的类别名称
        class_name = result.names[class_id]

        print(f"🔎 目标: {class_name:<12} | 置信度: {score:.2%} | 坐标: ({x1}, {y1}) -> ({x2}, {y2})")

        # 🚨 5. 编写你的自定义业务画框规则：
        # 规则 A：如果是“没戴安全帽 (no-helmet)”，这是高危情况！
        # 渲染方案：画醒目的红色粗框（厚度为3），并在上方写上大写的 WARNING 警示！
        if class_name == "no-helmet":
            color = (0, 0, 255)  # 红色 (BGR 格式)
            # 画外边框矩形
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness=3)
            
            # 定制文字标签内容
            label_text = f"DANGER: NO HELMET {score:.0%}"
            # 在框上方 10 像素写出白色警告字样，粗细为 2
            cv2.putText(draw_img, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                        color=(255, 255, 255), thickness=2)
            
            # 🚀 新增教学功能：把这个没戴安全帽的头部“框”物理剪切出来并存为独立小图片！
            # numpy 切片语法：[y的起止范围, x的起止范围] (注意：y在前，x在后)
            # 为了防止切片超出图像边界，我们可以使用 max/min 进行边界限制
            h_img, w_img, _ = draw_img.shape
            crop_y1, crop_y2 = max(0, y1), min(h_img, y2)
            crop_x1, crop_x2 = max(0, x1), min(w_img, x2)
            
            # 执行切片抠图
            crop_head = result.orig_img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 保存这个头盔缺失的高危头像小图
            crop_dir = os.path.join(output_dir, "crops")
            os.makedirs(crop_dir, exist_ok=True)
            crop_path = os.path.join(crop_dir, f"no_helmet_crop_{x1}_{y1}.jpg")
            cv2.imwrite(crop_path, crop_head)
            print(f"  📸 [高危取证] 成功把未戴帽人头抠图并存入: {os.path.basename(crop_path)}")

        # 规则 B：如果是“正常戴帽 (helmet)”或者是“反光背心 (safety-vest)”，这是合规情况！
        # 渲染方案：画温和的绿色细框（厚度为1），文字字号小一点
        elif class_name in ["helmet", "safety-vest"]:
            color = (0, 255, 0)  # 绿色 (BGR 格式)
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness=1)
            
            label_text = f"{class_name} {score:.0%}"
            cv2.putText(draw_img, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
                        color=color, thickness=1)

    # 6. 保存由我们完全接管并亲手绘制的图片
    save_path = os.path.join(output_dir, "bf27c9e6_custom.jpg")
    cv2.imwrite(save_path, draw_img)
    print(f"\n🎉 自定义画框完成！渲染好的结果已成功保存至:\n👉 {save_path}")

if __name__ == "__main__":
    main()

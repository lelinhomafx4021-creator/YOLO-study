# -*- coding: utf-8 -*-
"""
01_opencv_basics.py - 基础 OpenCV 图像处理
此脚本整合了最基础的 OpenCV 操作，包括读取、尺寸缩放、绘制几何检测框、添加标签以及保存。
这是理解目标检测后处理（在图片上画框）的关键基础。
"""

import cv2
import os

def main():
    # 1. 尝试读取图片（以项目根目录下的 result.jpg 为例，若不存在则提示）
    img_path = "result.jpg"
    if not os.path.exists(img_path):
        print(f"❌ 找不到测试图片 {img_path}，请确保路径正确。")
        return

    # 读取图片
    img = cv2.imread(img_path)
    print("📋 图片读取成功！")
    print(f"  - 图像形状 (高, 宽, 通道): {img.shape}")
    print(f"  - 数据类型: {img.dtype}")

    # 2. 缩放图片 (Resize)
    # 我们把图片缩放到 640 x 480 尺寸
    resized_img = cv2.resize(img, (640, 480))
    print(f"  - 缩放后形状: {resized_img.shape}")

    # 3. 在图片上绘制图形 (模拟目标检测画框)
    # 参数含义: 图像, 左上角坐标(x1, y1), 右下角坐标(x2, y2), 颜色BGR格式(这里是青色), 线条粗细
    cv2.rectangle(resized_img, (100, 80), (300, 350), (255, 255, 0), 2)

    # 4. 在图片上写字 (模拟标注类别与置信度)
    # 参数含义: 图像, 字符串内容, 文字起始坐标(左下角), 字体, 字体缩放大小, 颜色BGR格式, 线条粗细
    text = "helmet: 0.95"
    cv2.putText(resized_img, text, (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 5. 保存处理后的图片
    save_path = "00_basics/opencv_test_result.jpg"
    # 确保文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, resized_img)
    print(f"✅ 处理后的图片已安全保存至: {save_path}")

    # 6. 显示图片 (在本地桌面环境下会弹窗，Windows按任意键关闭窗口)
    # 注意: 如果是在无桌面环境或远程服务器上运行，请注释掉这两行以防挂起
    print("📺 弹窗显示图片中，请在弹出的窗口上按任意键退出...")
    cv2.imshow("OpenCV Basics Demo", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

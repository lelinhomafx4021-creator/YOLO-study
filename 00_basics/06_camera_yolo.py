# -*- coding: utf-8 -*-
"""
06_camera_yolo.py - YOLO 结合 OpenCV 调用摄像头/视频实时检测
此脚本整合了原先 week2_pytorch/ 下 05_opecvandyolo.py 到 08_multtest.py 中有关实时图像捕获和后处理手画框的逻辑。
展示了如何在每秒几十帧的视频流中，捕获图像像素、送入 YOLO 推理、手动画出检测框和类别标签，最后推流到窗口显示。
"""

import cv2
from ultralytics import YOLO
import os

def main():
    # 1. 加载官方预训练的 YOLO11n 权重模型（作为测试底座）
    model_path = "yolo11n.pt"
    if not os.path.exists(model_path):
        print(f"🔄 未检测到本地权重 {model_path}，正在自动下载官方预训练权重...")
    
    model = YOLO(model_path)

    # 2. 选择视频输入源
    # 如果你想调用电脑自带的摄像头，请把下方的 source 设为数字 0 (例如: source = 0)
    # 如果读取静态图片，可以写图片路径；这里为了安全演示，先尝试读取默认的图片。
    source = 0  # 默认设为 0（激活笔记本摄像头），如果在没有摄像头的服务器上，请改成图片或视频路径。
    
    print("==============================================================")
    print("📺 正在启动实时目标检测...")
    print("💡 提示: 如果你是在有桌面的 Windows 电脑上运行，窗口会实时弹出。")
    print("💡 操作: 选中视频窗口，按键盘上的 'q' 键即可安全退出实时检测。")
    print("==============================================================")

    # 打开摄像头或视频
    cap = cv2.VideoCapture(source)
    
    # 检查输入源是否正常打开
    if not cap.isOpened():
        print(f"❌ 无法打开输入源: {source}。")
        print("   如果你没有连接摄像头，请在代码里将 source 改为一张静态图片或视频文件的路径。")
        # 降级读取 result.jpg 静态图片，保证代码在无摄像头状态下也能演示
        source = "result.jpg"
        if os.path.exists(source):
            print(f"🔄 自动降级为静态图片检测: {source}")
            img = cv2.imread(source)
            # 推理
            results = model(img)
            result = results[0]
            # 手画框
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = box.conf[0].item()
                label = result.names[cls_id]
                
                # 画矩形框 (黄色)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # 写类别与置信度标签
                cv2.putText(img, f"{label}: {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Static Image YOLO Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 3. 实时视频帧循环读取
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法获取视频帧，或者视频播放结束。")
            break

        # ① 使用 YOLO 进行推理识别 (不自动保存，只在内存中计算)
        results = model(frame, verbose=False) # verbose=False 可以让控制台安静一点，不刷屏幕
        result = results[0]

        # ② 手动解析结果并在帧画面上画出彩色检测框
        for box in result.boxes:
            # 拿到预测边界框左上角和右下角的像素坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 类别编码（整数）
            cls_id = int(box.cls[0])
            # 置信度分数（0.0 ~ 1.0）
            conf = box.conf[0].item()
            # 类别名称
            label = result.names[cls_id]

            # 过滤掉低于 25% 置信度的杂碎框
            if conf < 0.25:
                continue

            # 在画面上画框 (亮黄色)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # 写字
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # ③ 显示实时渲染后的视频帧
        cv2.imshow("YOLO Live Video Detection", frame)

        # ④ 退出监听：按键盘上的 'q' 键中断循环
        # cv2.waitKey(1) 会等待 1 毫秒来处理窗口事件，并返回按键的 ASCII 码
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 检测到 'q' 键按下，正在关闭实时流...")
            break

    # 4. 清理垃圾，释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 摄像头已成功关闭，程序退出。")

if __name__ == "__main__":
    main()

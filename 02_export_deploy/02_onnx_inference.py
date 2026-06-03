# -*- coding: utf-8 -*-
"""
02_onnx_inference.py - 纯 ONNX Runtime + OpenCV 的独立推理与后处理脚本
本脚本不导入任何 PyTorch 和 Ultralytics 库，实现完全独立的部署计算。
包含：等比例缩放（Letterbox）、矩阵转置、坐标解码还原、以及 OpenCV 极速 NMS 过滤。
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    保持长宽比缩放图片，剩余区域用灰色填充（Letterbox）
    """
    shape = im.shape[:2]  # 原始图片的宽高 [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 缩放比例
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 剩余填充像素
    
    # 左右、上下均分填充像素
    dw /= 2
    dh /= 2

    # 如果需要缩放，则缩放
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    # 对图像四周进行灰色边界填充，拼满 640x640
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im, r, (left, top)

def main():
    # 1. 动态获取项目根目录并定位路径
    BASE_DIR = Path(__file__).resolve().parent.parent
    onnx_path = BASE_DIR / "runs" / "safety_helmet" / "yolo11n_baseline_v1" / "weights" / "best.onnx"
    test_img_path = BASE_DIR / "datasets" / "safety_helmet" / "images" / "val" / "bf27c9e6-8c6c-4fa0-b13d-3c70a777a783_jpg.rf.LLpoEU3BxpwwdiBMAeJC.jpg"
    output_dir = BASE_DIR / "runs" / "safety_helmet" / "onnx_deploy_results"
    crop_dir = output_dir / "crops"

    if not onnx_path.exists():
        print(f"❌ 找不到 ONNX 模型: {onnx_path}，请先运行 01_export.py 进行导出！")
        return
    if not test_img_path.exists():
        print(f"❌ 找不到测试图片: {test_img_path}，请确认数据集路径！")
        return

    # 创建输出文件夹
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    # 2. 初始化 ONNX Runtime 推理会话 (Session)
    # 优先指定使用 GPU (CUDA) 加速，如果无 CUDA 环境则平滑自动降级到 CPU
    print("🎬 正在初始化 ONNX Runtime 推理会话 (优先使用 GPU/CUDA)...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    
    # 3. 图像读取与预处理
    original_img = cv2.imread(str(test_img_path))
    h_orig, w_orig, _ = original_img.shape
    
    # 3.1 进行等比例缩放填充 (Letterbox)
    pad_img, r, pad = letterbox(original_img, (640, 640))
    
    # 3.2 BGR 转为 RGB 通道对齐
    rgb_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)
    
    # 3.3 归一化且维度重排：(H,W,C) ➡️ (C,H,W)，并增加 Batch 维度 ➡️ (1,C,H,W)
    blob = rgb_img.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0

    # 4. 执行推理前向传播
    print("🎬 正在使用 ONNX 引擎执行前向推理...")
    raw_outputs = session.run(None, {input_name: blob})
    predictions = raw_outputs[0]  # 原始输出形状: (1, 7, 8400)
    
    # 4.1 挤压并转置：将 (1, 7, 8400) 变形成 (8400, 7)
    predictions = np.squeeze(predictions, axis=0) # 变为 (7, 8400)
    predictions = predictions.T                   # 变为 (8400, 7)

    # 5. 后处理逻辑：解码预测框与得分过滤
    bboxes = []
    confidences = []
    class_ids = []
    
    for row in predictions:
        # 取第 4 到第 6 列作为类别预测分数
        scores = row[4:]
        class_id = np.argmax(scores)
        max_score = scores[class_id]
        
        # 只要把握（置信度）大于 25% 的框
        if max_score > 0.25:
            # 拿到 640x640 大图下的中心点坐标和宽高 (cx, cy, w, h)
            cx, cy, w, h = row[0:4]
            
            # 还原 Letterbox 的灰色黑边 Padding 偏移量
            cx -= pad[0]
            cy -= pad[1]
            
            # 还原缩放比例，计算在原始 1080P 图中的真实坐标
            cx /= r
            cy /= r
            w /= r
            h /= r
            
            # 从中心点坐标算出左上角起点 (x1, y1)
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            
            bboxes.append([x1, y1, int(w), int(h)])
            confidences.append(float(max_score))
            class_ids.append(int(class_id))

    # 6. 调用 OpenCV 底层 C++ 极速 NMS 引擎过滤多余的重合框
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold=0.25, nms_threshold=0.45)
    
    # 3分类名字字典映射
    names = {0: "helmet", 1: "no-helmet", 2: "safety-vest"}
    draw_img = original_img.copy()

    print(f"📊 NMS 过滤完成，最终保留了 {len(indices)} 个最精确的目标框。正在手画渲染并抠图...")
    for idx in indices:
        i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
        x1, y1, w, h = bboxes[i]
        score = confidences[i]
        class_id = class_ids[i]
        class_name = names.get(class_id, "unknown")
        
        # 计算右下角终点坐标 (x2, y2)
        x2, y2 = x1 + w, y1 + h

        # 规则 A：如果是高危的“未戴安全帽 (no-helmet)”
        if class_name == "no-helmet":
            # 用 OpenCV 画大红色框并写上危险警告
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(draw_img, f"DANGER: {class_name} {score:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 边界保护截取未戴帽的人头区域
            crop_y1 = max(0, y1)
            crop_y2 = min(h_orig, y2)
            crop_x1 = max(0, x1)
            crop_x2 = min(w_orig, x2)
            crop_head = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 物理保存独立头像
            crop_save_path = crop_dir / f"onnx_crop_{x1}_{y1}.jpg"
            cv2.imwrite(str(crop_save_path), crop_head)
            print(f"  📸 [高危抓拍取证] 成功截取头像 ➡️ {crop_save_path.name}")
            
        # 规则 B：如果是安全的“带了安全帽”或“穿了背心”
        elif class_name in ["helmet", "safety-vest"]:
            # 画温和的绿色框
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(draw_img, f"{class_name} {score:.0%}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 7. 物理存盘渲染好的大图结果
    final_save_path = output_dir / "onnx_inference_result.jpg"
    cv2.imwrite(str(final_save_path), draw_img)
    
    print("\n" + "="*50)
    print("🎉 ONNX 独立推理与渲染抠图全部完成！")
    print(f"👉 结果全景图保存在: {final_save_path.name}")
    print(f"👉 高危人脸抠图保存在: {crop_dir.name}/ 目录下")
    print("="*50)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
04_onnx_video_inference.py - 纯 ONNX Runtime + OpenCV 的视频侦测实战
【基础版：单线程 + 隔帧检测（旧框复用）】
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """保持长宽比缩放图片，剩余区域用灰色填充"""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def run_onnx_inference(session, input_name, original_img):
    """封装原本的预处理、推理、和后处理逻辑，返回当前帧检测到的所有框及其属性"""
    pad_img, r, pad = letterbox(original_img, (640, 640))
    rgb_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)
    blob = rgb_img.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0

    raw_outputs = session.run(None, {input_name: blob})
    predictions = np.squeeze(raw_outputs[0], axis=0).T

    bboxes, confidences, class_ids = [], [], []
    for row in predictions:
        scores = row[4:]
        class_id = np.argmax(scores)
        max_score = scores[class_id]
        if max_score > 0.25:
            cx, cy, w, h = row[0:4]
            cx, cy = cx - pad[0], cy - pad[1]
            cx, cy, w, h = cx / r, cy / r, w / r, h / r
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            bboxes.append([x1, y1, int(w), int(h)])
            confidences.append(float(max_score))
            class_ids.append(int(class_id))

    # NMS 过滤
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold=0.25, nms_threshold=0.45)
    
    # 组装返回结果列表，格式为 [{ 'box': [x1, y1, w, h], 'score': conf, 'class_id': id }]
    final_results = []
    if len(indices) > 0:
        for idx in indices:
            i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
            final_results.append({
                'box': bboxes[i],
                'score': confidences[i],
                'class_id': class_ids[i]
            })
    return final_results

def main():
    # ================= 1. 环境与路径准备 =================
    BASE_DIR = Path(__file__).resolve().parent.parent
    onnx_path = BASE_DIR / "runs" / "safety_helmet" / "yolo11n_baseline_v1" / "weights" / "best.onnx"
    
    output_dir = BASE_DIR / "runs" / "safety_helmet" / "onnx_video_results"
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    if not onnx_path.exists():
        print(f"❌ 找不到模型: {onnx_path}")
        return

    # 初始化 ONNX Session (放在循环外！)
    print("🎬 正在初始化 ONNX Runtime 推理引擎...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    # ================= 2. 视频源与保存器配置 =================
    # 【修改这里】如果电脑有摄像头，填 0 即可开启实时摄像头。如果有测试视频，填绝对路径。
    video_source = 0  
    # 示例: video_source = str(BASE_DIR / "datasets" / "safety_helmet" / "test_video.mp4")
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频源: {video_source}")
        return
        
    # 获取视频属性用于保存
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0 # 默认30
    
    # 初始化 VideoWriter（将加工好的相册页面装订成新视频）
    out_video_path = str(output_dir / "result_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(out_video_path, fourcc, fps_video, (width, height))

    # ================= 3. 隔帧与平滑 FPS 状态变量 =================
    names = {0: "helmet", 1: "no-helmet", 2: "safety-vest"}
    
    skip_interval = 3    # 每 3 帧让 ONNX 工作 1 次
    frame_id = 0         # 帧计数器
    latest_results = []  # 【旧框复用】的核心大脑，记住最近一次算出框的位置
    
    # FPS 计算器用
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    print("🚀 准备就绪，开始视频流水线 (按 'q' 键退出)...")
    
    # ================= 4. 视频流水线核心 =================
    while True:
        ret, frame = cap.read() # 动作1：从原视频复印一张图片（撕一页）
        if not ret:
            print("✅ 视频读取完毕！")
            break
            
        h_orig, w_orig = frame.shape[:2]
        
        # 动作2：决定是否要画新框，还是旧框复用
        if frame_id % skip_interval == 0:
            # 刚好遇到工作帧，调用 ONNX 进行真正的推理
            latest_results = run_onnx_inference(session, input_name, frame)
        else:
            # 遇到偷懒帧，什么也不做，完美复用上一帧留下来的 latest_results
            pass 
            
        # 动作3：无论怎样，在这个 frame 副本上，把框涂鸦上去
        for res in latest_results:
            x1, y1, w, h = res['box']
            x2, y2 = x1 + w, y1 + h
            score = res['score']
            class_name = names.get(res['class_id'], "unknown")
            
            # 画框与抓拍逻辑
            if class_name == "no-helmet":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # 粗红框
                cv2.putText(frame, f"DANGER: no-helmet {score:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 如果是工作帧，顺便保存高危抓拍（偷懒帧就不重复存图了）
                if frame_id % skip_interval == 0:
                    crop_y1, crop_y2 = max(0, y1), min(h_orig, y2)
                    crop_x1, crop_x2 = max(0, x1), min(w_orig, x2)
                    crop_head = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    # 用时间戳作为唯一文件名
                    crop_name = f"no_helmet_{int(time.time()*1000)}.jpg"
                    cv2.imwrite(str(crop_dir / crop_name), crop_head)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1) # 细绿框
                cv2.putText(frame, f"{class_name} {score:.0%}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ====== 涂鸦上平滑的实时 FPS ======
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0: # 每隔 1 秒更新一次显示数字
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
            
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # 动作4：把画好框的这页纸，拍在屏幕上展示给观众看
        cv2.imshow("ONNX Video Inference", frame)
        
        # 动作5：把这页纸收进新的视频相册里
        out.write(frame)
        
        frame_id += 1
        
        # 窗口心脏起搏器 + 按键限速/侦听器 (1 毫秒等待)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 用户按下了 'q' 键，中断检测流水线。")
            break

    # ================= 5. 优雅释放资源 =================
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("="*50)
    print(f"🎉 视频检测全流程完成！")
    print(f"👉 带框合成新视频已保存在: {out_video_path}")
    print(f"👉 违规抓拍截图保存在: {crop_dir.name}/")
    print("="*50)

if __name__ == "__main__":
    main()

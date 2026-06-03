# -*- coding: utf-8 -*-
"""
07_onnx_video_homework.py - 纯 ONNX Runtime + OpenCV 视频检测（完整参数详解版）
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    保持长宽比缩放图片，剩余区域用灰色填充。
    这是前处理的步骤，目的是把视频每一帧（比如1920x1080）转换成模型要求的 640x640 正方形。
    """
    shape = im.shape[:2]  # 获取原图的 (高度, 宽度)
    
    # 计算缩放比例：选宽度缩放比和高度缩放比中最小的那个，防止图片变形
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # 计算缩放后的实际长宽
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # 计算需要填充多少灰色像素（dw是左右填充，dh是上下填充）
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2  # 平分到两边
    
    # 如果尺寸不一致，就缩放图片
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    # 计算贴灰边的边界厚度
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 往缩放后的图片四周贴上灰色背景板
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # 返回：贴边后的图 (640, 640, 3)、缩放比例 r、填充的偏移量 (左边填充宽度, 顶部填充高度)
    return im, r, (left, top)


def run_onnx_inference(session, input_name, original_img):
    """
    运行 ONNX 推理的主函数：前处理 -> 推理 -> 后处理还原
    """
    # 1. 缩放并用灰色贴边，把原始高清帧拼成 640x640
    pad_img, r, pad = letterbox(original_img, (640, 640))
    
    # ====================================================
    # 【已填补】TODO 1: 图像前处理三步走
    # ====================================================
    
    # 步骤 A：把 OpenCV 默认的 BGR 格式（蓝绿红）转为 RGB（红绿蓝）
    # 参数详解：
    #   - pad_img: 刚刚缩放填充好的 640x640 图像
    #   - cv2.COLOR_BGR2RGB: 告诉 OpenCV 把通道顺序从 BGR 调换成 RGB
    rgb_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)
    
    # 步骤 B：把图像维度从 HWC (高, 宽, 通道) 转置为 CHW (通道, 高, 宽)
    # 参数详解：
    #   - (2, 0, 1): 原本排在最后的通道（索引2）挪到最前面，高（索引0）挪到中间，宽（索引1）挪到最后
    chw_img = rgb_img.transpose(2, 0, 1)
    
    # 步骤 C：增加批次(Batch)维度，数值归一化，并强转为 float32 格式
    # 参数详解：
    #   - np.expand_dims(chw_img, axis=0): 在第0个维度（最前面）挤出一个维度，形状由 (3, 640, 640) 变成 (1, 3, 640, 640)
    #   - .astype(np.float32): 把默认的 uint8 整数类型强转为 32位浮点数，方便模型计算
    #   - / 255.0: 将像素值从 0~255 缩放到 0.0~1.0 之间（归一化）
    blob = np.expand_dims(chw_img, axis=0).astype(np.float32) / 255.0
    
    
    # ====================================================
    # 【已填补】TODO 2: 启动 ONNX 推理（前向传播）
    # ====================================================
    
    # 调用 ONNX Runtime 运行模型推理
    # 参数详解：
    #   - None: 表示返回模型所有的输出节点（通常只有一个输出）
    #   - {input_name: blob}: 这是一个字典，把模型输入节点的名字（input_name）和我们刚刚前处理好的 blob 矩阵绑定送进去
    raw_outputs = session.run(None, {input_name: blob})
    
    
    # ====================================================
    # 后处理阶段（解析坐标，过滤重叠的框）
    # ====================================================
    if raw_outputs is None:
        return []

    # 剔除 Batch 维度，并转置矩阵，得到 (每行对应一个预测框) 的结构
    predictions = np.squeeze(raw_outputs[0], axis=0).T

    bboxes, confidences, class_ids = [], [], []
    for row in predictions:
        scores = row[4:]  # row的前4个元素是坐标，从第5个元素开始是每个类别的置信度分数
        class_id = np.argmax(scores)  # 找出分数最高的那个类别索引
        max_score = scores[class_id]  # 获取这个类别的置信度分值
        
        # 只有置信度大于 0.25 的框才保留，过滤掉背景和无用噪点
        if max_score > 0.25:
            cx, cy, w, h = row[0:4]  # 提取 640x640 尺度下的预测中心坐标和宽高
            
            # 【核心还原逻辑】把坐标从 640 尺度中还原回 1920x1080 的高清尺寸
            cx, cy = cx - pad[0], cy - pad[1]  # 减去灰边宽度
            cx, cy, w, h = cx / r, cy / r, w / r, h / r  # 除以缩放比例 r，等比例放大
            
            # 算出矩形框左上角的坐标 (x1, y1)
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            
            bboxes.append([x1, y1, int(w), int(h)])
            confidences.append(float(max_score))
            class_ids.append(int(class_id))

    # 应用非极大值抑制（NMS），消除同一个目标上层层叠叠的重叠框
    # 参数详解：
    #   - score_threshold=0.25: 再次确认置信度下限
    #   - nms_threshold=0.45: 重叠度（IoU）大于 45% 的多余框会被删掉，只保留分数最高的那个
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold=0.25, nms_threshold=0.45)
    
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
    # 基础路径配置
    BASE_DIR = Path(__file__).resolve().parent.parent
    onnx_path = BASE_DIR / "runs" / "safety_helmet" / "yolo11n_baseline_v1" / "weights" / "best.onnx"
    video_url = BASE_DIR / "vedio" / "77b2903f4d0f839cf20375fa3ccfaa80.mp4"
    
    # 创建保存输出文件的目录
    output_dir = BASE_DIR / "runs" / "safety_helmet" / "onnx_homework_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 初始化 ONNX 引擎
    print("🎬 正在初始化 ONNX 推理引擎...")
    providers = ["CPUExecutionProvider"]  # 默认在CPU上运行推理，防止部分电脑没有安装GPU驱动
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name  # 自动获取模型输入节点的名字（通常叫 "images"）

    # 2. 打开视频文件
    cap = cv2.VideoCapture(str(video_url))
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_url}")
        return
        
    # 获取视频原始参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频画面宽度（如 1920）
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频画面高度（如 1080）
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # 视频原始帧率
    
    # 初始化视频写入器（VideoWriter）用于保存最终结果
    out_video_path = str(output_dir / "homework_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设定保存视频的格式为 MP4
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # 类别字典映射
    class_names = {0: "helmet", 1: "no-helmet", 2: "safety-vest"}
    
    # 隔帧推理计数器
    frame_id = 0
    skip_interval = 3      # 每隔3帧推理一次，其余帧直接复用上一次的结果，这在低配电脑上可以极大减少卡顿
    latest_results = []    # 缓存上一次推理得到的目标框数据
    
    print("🚀 视频检测流水线已经就绪！")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🎉 视频处理完毕并成功保存！")
            break
            
        # 跳帧控制：只有当帧数能被 3 整除时，才执行耗电的模型推理
        if frame_id % skip_interval == 0:
            latest_results = run_onnx_inference(session, input_name, frame)

        # ====================================================
        # 【已填补】TODO 3: 手动用 OpenCV 在高清原图上画框和写字
        # ====================================================
        for res in latest_results:
            x1, y1, w, h = res['box']  # 拿到已经还原到 1920x1080 高清尺寸的坐标
            score = res['score']  # 拿到置信度分数
            cid = res['class_id']  # 类别ID（0或1）
            
            # 算出矩形框的右下角坐标 (x2, y2)
            x2 = x1 + w
            y2 = y1 + h
            
            # 从类别字典里获取这个框的名字（例如 "helmet" 或 "no-helmet"）
            name = class_names.get(cid, "unknown")
            
            # 根据类别，设定不同的画框颜色和警告文字
            if cid == 0:  # 0代表戴了安全帽
                color = (0, 255, 0)  # BGR 格式：纯绿色（安全）
                text = f"helmet {score:.0%}"
                thickness = 2  # 细线
            elif cid == 1:  # 1代表没戴安全帽（危险违规！）
                color = (0, 0, 255)  # BGR 格式：纯红色（危险警告）
                text = f"DANGER: no-helmet {score:.0%}"
                thickness = 3  # 粗线，引起注意
            else:  # 其他类型（如反光衣）
                color = (255, 255, 0)  # 青黄色
                text = f"{name} {score:.0%}"
                thickness = 2
                
            # A. 用 cv2.rectangle 在当前高清帧(frame)上画出矩形框
            # 参数详解：
            #   - frame: 要画框的画纸（原图）
            #   - (x1, y1): 矩形左上角坐标
            #   - (x2, y2): 矩形右下角坐标
            #   - color: 边框的颜色（BGR）
            #   - thickness: 边框线条的粗细（像素级）
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # B. 用 cv2.putText 在矩形框顶部写上分类名字和置信度分数
            # 参数详解：
            #   - frame: 要写字的画纸
            #   - text: 写在屏幕上的字符串内容
            #   - (x1, y1 - 10): 文字的起笔位置（稍微往上挪10个像素，不压住框的线条）
            #   - cv2.FONT_HERSHEY_SIMPLEX: OpenCV自带的标准无衬线字体
            #   - 0.5: 字体大小比例（缩放系数）
            #   - color: 文字的颜色（为了看着统一，和框同色）
            #   - 2: 文字线条的粗细
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            
        # 在弹窗窗口中实时播放画好框的画面
        cv2.imshow("ONNX Homework Show", frame)


        # ====================================================
        # 【已填补】TODO 4: 把画好框的这一帧写入到保存视频里
        # ====================================================
        
        # 参数详解：
        #   - out: 我们在开头创建的 VideoWriter 视频写入器
        #   - frame: 当前已经手工画好各种红绿边框和警告字体的 1920x1080 图像
        out.write(frame)
        

        # 按 'q' 键手动退出播放
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 用户主动中断了程序。")
            break
            
        frame_id += 1

    # 释放所有硬软件资源，封包视频
    cap.release()
    out.release()  # 这一步会彻底把数据封包进 mp4 文件并写入硬盘
    cv2.destroyAllWindows()
    
    print("="*50)
    print(f"🎉 处理完成！对比视频已生成在: {out_video_path}")
    print("="*50)

if __name__ == "__main__":
    main()

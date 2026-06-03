import cv2
from pathlib import Path
from ultralytics import YOLO
import os

# 1. 获取项目根目录，并正确定位权重路径和验证集图片路径
# 注意 resolve() 是方法，需要加括号
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "runs" / "safety_helmet" / "yolo11n_baseline_v1" / "weights" / "best.onnx"
val_img_dir = BASE_DIR / "datasets" / "safety_helmet" / "images" / "val"

# 定义结果输出路径（ renders存渲染图，crops存未戴帽抠图，labels存日志txt ）
output_dir = BASE_DIR / "runs" / "safety_helmet" / "onnx_deploy_results"
render_dir = output_dir / "renders"
crop_dir = output_dir / "crops"
label_dir = output_dir / "labels"

# 2. 自动创建输出的文件夹目录
for d in [render_dir, crop_dir, label_dir]:
    d.mkdir(parents=True, exist_ok=True)

# 3. 初始化 YOLO 实例对象加载模型
model = YOLO(model_path)
print("🚀 模型加载成功，准备开始批量验证集检测...")

# 定义画框的类别颜色（ BGR格式，OpenCV 默认是蓝、绿、红 ）
class_colors = {
    "helmet": (0, 255, 0),       # 绿色 - 合规
    "no-helmet": (0, 0, 255),    # 红色 - 高危
    "safety-vest": (255, 165, 0) # 橙色 - 合规
}

# 4. 获取验证集下所有的图片路径
img_paths = sorted(val_img_dir.glob("*.jpg"))

if not img_paths:
    print(f"❌ error: 在 {val_img_dir} 下没有找到图片文件")
    exit(1)

total_boxes = 0   
total_alarms = 0  

# 5. 循环处理每一张图片
for idx, img_path in enumerate(img_paths):
    # 读取原始图片（BGR 格式的 ndarray 矩阵）
    original_image = cv2.imread(str(img_path))
    h, w = original_image.shape[0:2]
    
    # 复制一张干净的图用于画框，防止原图被“框线污染”影响后面抠图
    draw_img = original_image.copy()
    
    # 执行模型前向推理，指定使用 CPU 以防 GPU-CPU 数据拷贝绑定报错
    results = model(original_image, verbose=False, conf=0.25, save=False, device="cpu")
    boxes = results[0].boxes
    
    # 准备当前图片的标签文本行列表
    current_labels = []
    box_idx = 0  # 单张图里的目标计数器，用来区分抠图文件名
    
    # 6. 遍历这张图里检测到的每一个目标框
    for box in boxes:
        # 降维并转为 Python 列表，然后将每个元素强转为 int 像素点整数
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # 提取类别 ID 和置信度
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        
        # 映射出类别名称
        cls_name = results[0].names[cls_id]
        
        # 获取颜色，默认使用白色
        color = class_colors.get(cls_name, (255, 255, 255))
        
        # 如果是高危的未戴帽行为，将框线加粗，并记录报警总数
        thickness = 3 if cls_name == "no-helmet" else 1
        if cls_name == "no-helmet":
            total_alarms += 1
            
        # ====== 绘制矩形框和类别标签 ======
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)
        
        # 在框上方绘制标注文本
        text_label = f"{cls_name} {conf:.0%}"
        cv2.putText(draw_img, text_label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # ====== NumPy 图像切片完成抠图 ======
        # 做边界范围保护，防止切片坐标超出图像尺寸报错
        crop_y1, crop_y2 = max(0, y1), min(h, y2)
        crop_x1, crop_x2 = max(0, x1), min(w, x2)
        crop_img = original_image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # 生成抠图保存的文件名并存盘
        crop_filename = f"{img_path.stem}_{cls_name}_{box_idx}.jpg"
        cv2.imwrite(str(crop_dir / crop_filename), crop_img)
        
        # ====== 收集标签数据 ======
        # 格式：类别 置信度 x1 y1 x2 y2
        current_labels.append(f"{cls_name} {conf:.4f} {x1} {y1} {x2} {y2}")
        
        box_idx += 1
        total_boxes += 1
        
    # ====== 保存当前图的渲染标注图 ======
    render_filename = f"{img_path.stem}_render.jpg"
    cv2.imwrite(str(render_dir / render_filename), draw_img)
    
    # ====== 保存当前图的文本检测日志 ======
    label_filename = f"{img_path.stem}.txt"
    with open(label_dir / label_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(current_labels))
        
    print(f"📸 [{idx + 1}/{len(img_paths)}] 成功处理: {img_path.name} (目标数: {box_idx})")

# 7. 推理结束后输出汇总报告
print("\n" + "=" * 55)
print("🎉 批量验证集推理与标签生成任务全部完成！")
print(f"📊 总共检测图片: {len(img_paths)} 张")
print(f"🏷️ 共捕获检测框: {total_boxes} 个")
print(f"🚨 高危未戴帽报警: {total_alarms} 次")
print(f"📂 渲染标注大图已保存在: {render_dir}")
print(f"📂 未戴帽等目标抠图保存在: {crop_dir}")
print(f"📂 自定义文本日志已保存在: {label_dir}")
print("=" * 55)

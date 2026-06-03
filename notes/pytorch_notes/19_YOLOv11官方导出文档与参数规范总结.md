# 19_YOLOv11 官方导出与参数规范

基于 YOLOv11（Ultralytics）官方最新规范整理，为你总结一键导出的代码和超参数避坑指南。

---

## 🛠️ 一、 一键导出命令与代码

### 1. 命令行快速导出 (推荐)
```powershell
uv run yolo export model=runs/safety_helmet/yolo11n_baseline_v1/weights/best.pt format=onnx imgsz=640 simplify=True
```

### 2. Python 代码导出
```python
from ultralytics import YOLO

# 1. 加载训练好的 PyTorch 模型
model = YOLO("runs/safety_helmet/yolo11n_baseline_v1/weights/best.pt")

# 2. 导出为高性能 ONNX 格式
model.export(
    format="onnx",      # 目标格式
    imgsz=640,          # 输入分辨率 (静态 640x640)
    simplify=True,      # 开启计算图简化 (强推)
    half=False,         # CPU 部署建议保持 False; GPU 部署可以开 True
    opset=17            # ONNX 算子版本 (默认 17 兼容性好)
)
```

---

## ⚙️ 二、 核心参数深度避坑（面试与工程落地）

*   **`simplify=True` (图简化) —— 必须开启**
    *   *原理*：调用 `onnxslim` 合并网络里重复的节点和常量。
    *   *作用*：不仅能缩小文件体积，还能防止在老旧硬件设备上推理时因“不支持某些算子”而导致程序崩溃。
*   **`half=True` (FP16量化) —— 看硬件决定**
    *   *⚠️ 避坑*：**只在有英伟达 GPU（显卡）部署时开启**。如果你的部署端是普通的 CPU，开启 `half` 反而会因为 CPU 不支持 FP16 硬件加速，而需要频繁在内存中做“数据类型转换”，导致推理速度**变慢**！CPU 部署请老老实实设为 `False`。
*   **`dynamic=True` (动态尺寸输入) —— 建议关闭**
    *   *⚠️ 避坑*：开启它允许输入任意分辨率的图片，但会阻止推理引擎进行“静态计算图内存预分配”，速度会慢 **10% ~ 20%**。在工业界，优先推荐用 OpenCV Letterbox 把图片都缩放到 640x640 静态尺寸再喂给模型。

---

## 🏆 三、 硬件平台部署“黄金路线”（面试必背）

面试官问：*“对于 XX 硬件，你推荐用什么格式部署？”*

1.  **工控机 / 普通 CPU 平台**：
    *   `YOLOv11 (.pt)` ➡️ 导出为 **`ONNX`** ➡️ 使用 **ONNX Runtime (CPU)** 推理。
    *   *优化*：`simplify=True`，关闭半精度（`half=False`）。
2.  **英伟达 GPU 显卡平台**：
    *   `YOLOv11 (.pt)` ➡️ 导出为 **`TensorRT (engine)`** ➡️ 使用 **TensorRT** 驱动推理。
    *   *优化*：开启 `half=True`（FP16量化），延迟能直接降到几毫秒级别。
3.  **Intel 处理器 (工控机/普通PC)**：
    *   `YOLOv11 (.pt)` ➡️ 导出为 **`OpenVINO`** ➡️ 使用 **OpenVINO Runtime** 推理。
4.  **手机等极低算力移动端**：
    *   **Android**：导出为 **`TFLite`** 或 **`NCNN`**。
    *   **iOS**：导出为 **`CoreML`**。
    *   *优化*：必须开启 INT8 量化，防止占满手机内存。

---

## 🔧 四、你的实际导出代码对照

你已有 `02_export_deploy/01_export.py`：

```python
from pathlib import Path
from ultralytics import YOLO
BASE_DIR = Path(__file__).resolve().parent.parent
pt_path = BASE_DIR/"runs"/"safety_helmet"/"yolo11n_baseline_v1"/"weights"/"best.pt"
model = YOLO(pt_path)
onnx_path = model.export(format="onnx", imgsz=640, simplify=True, half=False)
```

### 导出后验证

```python
import onnx
onnx_model = onnx.load("best.onnx")
onnx.checker.check_model(onnx_model)   # 检查结构合法性
print("输入:", [d.name for d in onnx_model.graph.input])
print("输出:", [d.name for d in onnx_model.graph.output])
```

### 各格式导出 + 大小对比

```python
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, simplify=True, half=True)    # FP16, ~3MB
model.export(format="engine", imgsz=640, half=True)                 # TensorRT
model.export(format="openvino", imgsz=640)                          # OpenVINO

# 对比:
# best.pt      ≈ 5-6 MB  → 需要 PyTorch (3GB)
# best.onnx    ≈ 5-10 MB → 需要 onnxruntime (100MB)
# best_fp16.onnx ≈ 2.5-5 MB → FP16 半精度
# best.engine  ≈ 8-20 MB → TensorRT (NVIDIA 专用,最快)
```

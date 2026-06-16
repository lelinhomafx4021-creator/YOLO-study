# 18_YOLO 模型导出与高性能部署解密

在大厂和实际项目落地中，**“会训练模型”只能拿50分，“会部署和优化”才能拿满分**。本篇笔记用最直接的大白话和面试考点，帮你快速打通“算法部署最后一公里”。

---

## 💡 一、 为什么生产环境不直接部署 `.pt` 模型？（面试高频）
*   **依赖过重（“身体臃肿”）**：PyTorch + GPU 运行库动辄几个 G，边缘端设备（如工控机、树莓派）内存小，根本装不下。
*   **Python 的速度瓶颈**：Python 的全局解释器锁（GIL）导致其无法实现真正的高并发多线程推理。
*   **跨平台困难**：工业控制、摄像头管理系统一般用 C++、C# 或 Java 编写，强行调用 Python 脚本不仅慢，而且极易崩溃。

> **🌟 工业界金科玉律**：**“PyTorch 训练（灵活好用） ➡️ 导出为 ONNX 中间格式 ➡️ 部署端使用 C++/Python 轻量化引擎跑推理（追求极致速度）”**。

---

## 🌎 二、 ONNX：AI 界的“PDF 格式”
*   **ONNX (Open Neural Network Exchange)** 是一种跨平台的计算图表示规范。
*   **大白话比喻**：用 PyTorch 训练好模型就像用 Word 写的 `.docx` 文档（没装 PyTorch 的机器打不开）。把它导出为 `.onnx` 就像存为 `.pdf` 格式。别人只需要安装一个体积只有几十MB的 **ONNX Runtime (PDF 阅读器)**，就能轻松跑推理。

---

## ⚡ 三、 精度量化加速：FP32 vs FP16 vs INT8（面试必背）

### 3.1 一张表看懂

| 精度类型 | 占用位数 | 大小对比 | 速度 | 精度 | 用在哪 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** (单精度) | 32位=4字节 | 100% | 1× | 无损 | 训练、CPU部署 |
| **FP16** (半精度) | 16位=2字节 | 50% | ~2× | 极小 | GPU推理(Tensor Cores) |
| **INT8** (8位整数) | 8位=1字节 | 25% | 3-4× | 需校准 | 手机、嵌入式 |

### 3.2 举个具体例子

```
一个 268,650 参数的 SimpleCNN：

FP32（默认）：
  268,650 × 4 字节 ≈ 1.07 MB
  每个参数是一个 32 位浮点数，精度很高

FP16：
  268,650 × 2 字节 ≈ 0.54 MB
  每个参数压缩到 16 位，大小砍半
  精度损失 ≈ 0.1-0.3%，肉眼看不出来

INT8：
  268,650 × 1 字节 ≈ 0.27 MB
  每个参数压缩到 8 位整数
  精度可能掉 1-3%，需要校准（给一批图让模型看，找最佳缩放范围）
```

### 3.3 FP16 为什么能加速

```
一个矩阵乘法 W × X：

FP32：每个数 4 字节 → 从显存读到计算单元 → 4 字节走一遍 → 算
FP16：每个数 2 字节 → 显存传输量直接减半
     → NVIDIA Tensor Cores 硬件专为 FP16 设计 → 吞吐量翻倍
     → 你的 RTX 3050 就有 Tensor Cores
```

**类比：** 搬砖，FP32 是一个个搬（每次 1 块），FP16 是一次搬两块（但砖稍微轻一点）。

### 3.4 YOLO 里怎么用

```python
# 导出时指定精度
model.export(format=”onnx”, half=True)   # half=True = FP16
model.export(format=”onnx”, int8=True)   # INT8 量化（需要额外安装库）

# 训练时也可以用混合精度（自动切换 FP32/FP16）
model.train(data=”data.yaml”, epochs=50, amp=True)
# amp=True = Automatic Mixed Precision
# → 大部分计算用 FP16 加速，关键步骤（如 loss）保留 FP32 保证精度
```

---

## ⚡ 四、 完整 ONNX 导出 + 推理实战（手把手）

### 4.1 第一步：导出 .pt → .onnx

```python
from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO(“runs/safety_helmet/yolo11n_baseline_v1/weights/best.pt”)

# 导出 ONNX（最简单方式）
model.export(format=”onnx”)
# 生成: best.onnx（和 best.pt 同目录）

# 带参数的导出
model.export(
    format=”onnx”,
    imgsz=640,        # 输入尺寸
    half=True,        # FP16（文件更小，推理更快）
    opset=12,         # ONNX 算子集版本
    simplify=True,    # 精简计算图（去掉冗余操作）
)
```

### 4.2 第二步：核实 ONNX 模型

```python
import onnx

# 加载 ONNX 模型
onnx_model = onnx.load(“best.onnx”)

# 检查模型是否合法
onnx.checker.check_model(onnx_model)
print(“ONNX 模型结构合法 ✅”)

# 看输入输出
print(f”输入: {onnx_model.graph.input[0].name} — 形状 {onnx_model.graph.input[0].type}”)
print(f”输出: {onnx_model.graph.output[0].name} — 形状 {onnx_model.graph.output[0].type}”)
```

### 4.3 第三步：ONNX Runtime 推理（完整代码）

这是笔记第四节”四部曲”的完整可跑代码：

```python
import cv2
import numpy as np
import onnxruntime as ort
import time

# ── 1. 加载 ONNX 模型 ──
session = ort.InferenceSession(
    “best.onnx”,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先 GPU
)

# 看模型期望的输入形状
input_info = session.get_inputs()[0]
print(f”模型输入名: {input_info.name}”)
print(f”模型输入形状: {input_info.shape}”)  # 如 [1, 3, 640, 640]

# ── 2. 读图 + 预处理（必须和训练时完全一致！）──
img = cv2.imread(“test.jpg”)                    # OpenCV 读 → BGR 格式
h, w = img.shape[:2]                            # 记录原始尺寸，后面还原框坐标用

# Letterbox：等比例缩放，保持长宽比，用灰色填充
scale = min(640/w, 640/h)                       # 计算缩放比例
new_w, new_h = int(w * scale), int(h * scale)
img_resized = cv2.resize(img, (new_w, new_h))

# 创建 640×640 的灰色画布，把缩放后的图贴中间
canvas = np.full((640, 640, 3), 114, dtype=np.uint8)  # 114=YOLO 默认填充色
canvas[(640-new_h)//2:(640-new_h)//2+new_h,
       (640-new_w)//2:(640-new_w)//2+new_w] = img_resized

# 通道 + 归一化 + 维度
img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)       # BGR → RGB
img_norm = img_rgb.astype(np.float32) / 255.0             # [0,255] → [0,1]
img_chw = np.transpose(img_norm, (2, 0, 1))               # HWC → CHW
input_tensor = img_chw[np.newaxis, ...]                   # 加 batch 维 → [1,3,640,640]

# ── 3. 推理 ──
t0 = time.time()
outputs = session.run(None, {input_info.name: input_tensor})
elapsed = time.time() - t0

# outputs[0] 形状: [1, 8400, 84]
#   8400 个候选位置
#   84 = 4个框坐标 + 1个置信度 + 79个类别分数(YOLO预训练)
#   如果用你自己的模型(3类)，需要去 Ultralytics 文档确认输出格式

print(f”推理耗时: {elapsed*1000:.1f} ms”)
print(f”输出形状: {outputs[0].shape}”)
```

### 4.4 第四步：后处理（NMS + 坐标解码）

```python
def nms(boxes, scores, iou_threshold=0.5):
    “””简单的 NMS 实现”””
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        keep.append(order[0])
        if len(order) == 1:
            break
        ious = compute_iou_matrix(boxes[order[0]], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return keep

# 实际使用时 YOLO 输出格式比较复杂，建议直接看你的
# 02_export_deploy/02_onnx_inference.py — 里面有完整的手写 NMS 和坐标解码
```

### 4.5 预处理为什么必须对齐

```
训练时（YOLO 内部做的）：
  读图 → BGR→RGB → /255 → letterbox → [1,3,640,640] → 喂给模型

推理时（你手写的）：
  读图 → BGR→RGB → /255 → letterbox → [1,3,640,640] → 喂给 ONNX

如果任何一步不一样（比如忘了 BGR → RGB，或者先 resize 再 letterbox）
→ 模型看到的图和训练时的不一样 → 预测结果漂移 → 精度掉
```

---

## ⚡ 五、 导出 PyTorch 原生模型为 ONNX（不用 YOLO）

```python
import torch

# 你自己的 SimpleCNN
model = SimpleCNN()
model.load_state_dict(torch.load(“my_cnn.pth”, map_location=”cpu”))
model.eval()

# 导出 ONNX
dummy_input = torch.randn(1, 3, 32, 32)  # 假输入，让 PyTorch”追踪”一遍模型

torch.onnx.export(
    model,                          # 模型
    dummy_input,                    # 示例输入（决定输入形状）
    “my_cnn.onnx”,                  # 保存路径
    input_names=[“input”],           # 输入节点名字
    output_names=[“output”],         # 输出节点名字
    dynamic_axes={                   # 可变维度（batch）
        “input”: {0: “batch”},
        “output”: {0: “batch”},
    },
    opset_version=17,               # ONNX 算子集版本
)
print(“导出完成: my_cnn.onnx”)
```

**和 YOLO 导出的区别：**
- YOLO 导出的 ONNX 包含预处理后处理（letterbox、NMS）
- 你手写的 ONNX 只包含网络本身 → 预处理后处理要自己写

---

## ⚡ 六、在本机跑推理：YOLO(.pt) vs ONNX(.onnx) 实际对比

### 6.1 依赖大小对比

```bash
# YOLO 推理 (.pt) 需要安装 PyTorch
pip install torch torchvision ultralytics
# → torch 约 2.5 GB（含 CUDA 运行时）
# → ultralytics 约 0.5 GB
# → 总依赖 ≈ 3 GB

# ONNX Runtime 推理 (.onnx) 不需要 PyTorch
pip install onnxruntime-gpu    # 有 GPU
pip install onnxruntime        # 纯 CPU
# → onnxruntime 约 100 MB
# → 总依赖 ≈ 100 MB
```

**在本机上：ONNX Runtime 推理完全不依赖 torch 包。** 你把 `torch` 卸载了照样跑。

### 6.2 代码对比

| | YOLO (.pt) | ONNX Runtime (.onnx) |
|---|---|---|
| 代码量 | 1-2 行 | 20-30 行 |
| 需要装 PyTorch | **需要（3 GB）** | **不需要（100 MB）** |
| 速度 | 正常 | 更快 |
| 预处理/NMS | 自动 | 手写 |
| 适合什么 | 开发、实验 | 部署、嵌入应用 |

### 6.3 本机练习

```bash
pip install onnxruntime-gpu

python 02_export_deploy/01_export.py          # 导出 .onnx
python 02_export_deploy/02_onnx_inference.py  # 用 .onnx 推理
python 02_export_deploy/04_onnx_video_inference.py  # 视频版
```

---

## 📦 七、YOLO 的 ONNX 导出内部干了什么

```python
model.export(format=”onnx”)
# 内部调用了：
# 1. torch.onnx.export() → PyTorch 模型 → ONNX 计算图
# 2. 把预处理（letterbox、归一化、BGR→RGB）写进 ONNX 图
# 3. 把 NMS 后处理写进 ONNX 图（可选）
# 4. 保存为 .onnx 文件
```

**YOLO 导出的 ONNX 比你自己 `torch.onnx.export` 的更”完整”。** 所以 YOLO 的 ONNX 推理比纯手写简单——预处理和 NMS 都打包了。

---

## 🔑 关键结论

```
训练完的产物：
  best.pt      → 自己用 YOLO 跑（要装 PyTorch 3GB）
  best.onnx    → 给别人用 / 轻量化部署（只要 onnxruntime 100MB）

PyTorch 重量级是为了训练（需要 autograd、优化器、backward）
ONNX 轻量级是只做推理（只有前向传播，不需要梯度）

所以：”训练用 PyTorch，部署用 ONNX” — 各取所长。
```

---

## 🚚 四、 ONNX Runtime 高性能推理“四部曲”
用 ONNX 推理时，我们必须手动在代码里写出这四步（即**预处理对齐**）：
1. **图像读入**：使用 OpenCV 读图（默认 BGR 格式）。
2. **预处理（最容易踩精度不对齐的坑）**：
   * **等比例缩放（Letterbox）**：将图缩放到 `640x640`，多余地方用灰色填充，保持长宽比不变。
   * **通道对调**：将 BGR 转成 **RGB**。
   * **归一化**：像素值除以 255.0，转为 `0.0 ~ 1.0` 浮点数。
   * **维度重排**：转为 `(1, 3, 640, 640)`，即 `(batch, channel, height, width)`。
3. **引擎推理**：调用 `session.run()` 送入 ONNX 引擎计算。
4. **后处理 (NMS)**：调用**非极大值抑制（NMS）**去除重合多余的候选框，过滤出最终的检测框。

---

## 🔧 五、在本机跑推理：YOLO(.pt) vs ONNX(.onnx) 实际对比

### 5.1 依赖大小对比

```bash
# YOLO 推理 (.pt) 需要安装 PyTorch
pip install torch torchvision ultralytics
# → torch 约 2.5 GB（含 CUDA 运行时）
# → ultralytics 约 0.5 GB
# → 总依赖 ≈ 3 GB

# ONNX Runtime 推理 (.onnx) 不需要 PyTorch
pip install onnxruntime-gpu    # 有 GPU
pip install onnxruntime        # 纯 CPU
# → onnxruntime 约 100 MB
# → 总依赖 ≈ 100 MB
```

**在本机上：ONNX Runtime 推理完全不依赖 torch 包。** 你把 `torch` 卸载了照样跑。

### 5.2 代码对比：同一张图，两种推理方式

```python
# ========== 方式 A：YOLO 推理 (.pt) ==========
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source="test.jpg", save=True)
# 一行搞定，预处理/NMS/画框全自动
```

```python
# ========== 方式 B：ONNX Runtime 推理 (.onnx) ==========
import cv2, numpy as np, onnxruntime as ort

# 1. 手写预处理
img = cv2.imread("test.jpg")
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # [1, 3, 640, 640]

# 2. 推理
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"images": img})

# 3. 手写后处理（NMS + 画框 + 解码坐标）
# ... 需要自己写 NMS 和坐标解码逻辑 ...
```

| | YOLO (.pt) | ONNX Runtime (.onnx) |
|---|---|---|
| 代码量 | 1-2 行 | 20-30 行 |
| 需要装 PyTorch | **需要（3 GB）** | **不需要（100 MB）** |
| 速度 | 正常 | 更快（无 Python 开销） |
| 预处理/NMS | 自动 | 手写 |
| 适合什么 | 开发阶段、快速实验 | 部署给别人、嵌入应用、轻量化 |

### 5.3 什么时候用哪种

```
你和你的电脑（开发阶段）：
  → YOLO .pt 推理就够了，torch 已经装了
  → 没必要折腾 ONNX

你要部署给别人/服务器/工控机：
  → 转成 ONNX → 他人只需装 onnxruntime（100 MB）
  → 不用要求别人装 3 GB 的 torch

你要嵌入到 C++/C# 应用里：
  → ONNX Runtime 有 C/C++/C# API
  → PyTorch 只有 Python API 和 C++(LibTorch)
```

### 5.4 本机练习 ONNX 推理（不装额外依赖）

你的项目里已经有完整的 ONNX 推理脚本：

```
02_export_deploy/01_export.py          ← YOLO → ONNX 导出
02_export_deploy/02_onnx_inference.py  ← ONNX Runtime 推理（手动预处理 + NMS）
02_export_deploy/04_onnx_video_inference.py ← 视频版
```

```bash
# 如果你还没装 onnxruntime：
pip install onnxruntime-gpu    # GPU 版（推荐）
pip install onnxruntime        # CPU 版

# 跑一遍：
python 02_export_deploy/01_export.py          # 导出 .onnx
python 02_export_deploy/02_onnx_inference.py  # 用 .onnx 推理
```

### 5.5 一个实验验证

```python
# 在终端跑这个，证明 ONNX 不依赖 torch
import onnxruntime as ort
import sys

print(f"Python 路径里有 torch 吗？{'torch' in sys.modules}")

session = ort.InferenceSession("02_export_deploy/model.onnx")
print("ONNX 推理成功！完全没用到 torch。")
```

---

## 📦 六、YOLO 的 ONNX 导出到底干了什么

```python
model.export(format="onnx")
# 内部做的事：
# 1. torch.onnx.export() → 把 PyTorch 模型转成 ONNX 计算图
# 2. 把预处理（letterbox、归一化）也写进 ONNX 图里（YOLOv8+ 的特性）
# 3. 把 NMS 后处理也写进 ONNX 图里（可选）
# 4. 保存为 .onnx 文件
```

**YOLO 导出的 ONNX 比你自己 `torch.onnx.export` 的更"完整"——预处理和 NMS 都打包进去了。** 所以 YOLO 的 ONNX 推理比自己手写的简单。

---

## 🔑 关键结论

```
训练完的产物：
  best.pt      → 自己用 YOLO 跑（要装 PyTorch 3GB）
  best.onnx    → 给别人用 / 轻量化部署（只要 onnxruntime 100MB）

PyTorch 重量级是为了训练（需要 autograd、优化器、backward）
ONNX 轻量级是只做推理（只有前向传播，不需要梯度）
```

这就是为什么"训练用 PyTorch，部署用 ONNX"——各取所长。

---

## 复习速答

- `PyTorch`：适合训练。
- `ONNX`：适合部署。
- `TensorRT`：NVIDIA 平台高性能推理引擎。
- `OpenVINO`：Intel 平台常用推理方案。
- `导出目标`：把训练模型变成更轻量、更通用的部署格式。

# 📊 ONNX Runtime API 与 YOLO Model API 对比速查笔记

在计算机视觉项目中，我们最常打交道的就是两套推理 API：
1. **YOLO Model API (高层保姆式封装)**：使用 PyTorch 或 Ultralytics 库，适合快速开发、实验。
2. **ONNX Runtime API (底层工业部署引擎)**：使用 ONNX 引擎，适合脱离大框架后在各类端侧设备上高效运行。

这篇笔记用最通俗的语言和对照表，帮你理清它们的 API 怎么用，以及各自对应的步骤。

---

## 🆚 核心流程对照表

| 步骤 | YOLO Model (高层封装) | ONNX Runtime (底层引擎) |
| :--- | :--- | :--- |
| **导入库** | `from ultralytics import YOLO` | `import onnxruntime as ort` |
| **加载模型** | `model = YOLO("best.pt")` | `session = ort.InferenceSession("best.onnx")` |
| **前处理** | ❌ 自动帮你做完 | 🛠️ 需要自己写 `letterbox`、转 RGB、转 CHW、归一化 |
| **送入推理** | `results = model(img)` | `outputs = session.run(None, {input_name: blob})` |
| **输出格式** | 包装好的 Python 对象列表 | 原始的 Numpy 矩阵（一堆数字坐标和分数） |
| **后处理/NMS**| ❌ 自动帮你做完 | 🛠️ 需要自己写 `cv2.dnn.NMSBoxes` 过滤重叠框 |
| **画框标注** | `annotated_img = results[0].plot()` | 🛠️ 需要自己写 `cv2.rectangle` 和 `cv2.putText` |

---

## 🛠️ 详细 API 拆解与参数解释

### 一、 YOLO Model API (以 Ultralytics 为例)

高层 API 的特点就是**“把复杂留给库，把简单留给用户”**。

#### 1. 加载模型
```python
model = YOLO("best.pt")
```
* **作用**：读取训练好的 PyTorch 权重文件，在内存中构建好神经网络。

#### 2. 前向传播（推理）
```python
results = model(frame, conf=0.25, verbose=False)
```
* **常用参数详解**：
  * `frame`：输入的图片（可以是 OpenCV 读进来的 BGR 矩阵、图片路径字符串、甚至是视频流）。
  * `conf=0.25`：置信度阈值。只有模型认为概率大于 25% 的目标才会被返回。
  * `verbose=False`：是否在控制台打印 YOLO 的每一帧检测耗时。在视频流中通常设为 `False` 保持控制台干净。
  * `device='cuda'`：指定在 GPU（显卡）上跑还是在 CPU 上跑。

#### 3. 获取输出结果
```python
results[0].boxes      # 拿到所有的检测框坐标、置信度和类别ID
results[0].plot()     # 自动在图上画好各种框，并返回画好的图片
results[0].save()     # 自动把画好框的图保存到硬盘上
```

---

### 二、 ONNX Runtime API (底层部署)

ONNX Runtime 不认识任何图片格式，它只认识 **NumPy 多维矩阵 (Tensor)**。它只负责最核心的“黑盒数学矩阵计算”。

#### 1. 加载模型（创建 Session）
```python
providers = ["CPUExecutionProvider"] # 或者 "CUDAExecutionProvider" (GPU加速)
session = ort.InferenceSession("best.onnx", providers=providers)
```
* **InferenceSession**：ONNX Runtime 的核心会话对象。相当于我们的“推理计算器”。
* **providers**：指定用什么硬件加速。`CPUExecutionProvider` 代表在 CPU 上算，`CUDAExecutionProvider` 代表用英伟达显卡算。

#### 2. 获取输入节点信息
由于我们要手动喂数据，我们需要先问问模型：“你的输入接口叫什么名字？它要什么形状的矩阵？”
```python
input_name = session.get_inputs()[0].name
```
* **session.get_inputs()**：获取模型所有输入接口的列表。 YOLOV11 通常只有一个输入节点。
* **.name**：通常返回一个字符串，比如 `"images"`。推理时我们需要把这个名字和我们的图片矩阵绑定。

#### 3. 执行推理前向传播
```python
raw_outputs = session.run(None, {input_name: blob})
```
* **session.run()**：开始启动模型内部的矩阵乘法计算。
* **第一个参数 `None`**：代表我们需要模型所有的输出结果。
* **第二个参数 `{input_name: blob}`**：这是一个 Python 字典。
  * `input_name`：就是刚才拿到的输入接口名字（例如 `"images"`）。
  * `blob`：必须是经过转置、归一化、增加了 Batch 维度的四维 NumPy 矩阵，形状为 `(1, 3, 640, 640)`，数据类型必须是 `np.float32`。
* **返回值 `raw_outputs`**：返回一个 NumPy 数组的列表（包含模型吐出来的原始数据）。

---

## 💡 面试加分小知识

面试官常问：“**既然高层 API（如 PyTorch/YOLO）这么方便，为什么我们在实际的工业相机、路边监控摄像头落地时，一定要导出成 ONNX 来用底层 API 跑？**”

* **答案思路**：
  1. **脱离大框架依赖**：PyTorch 框架非常臃肿（好几个G大小），不适合安装在资源紧张的嵌入式设备（如工地摄像头盒子、行车记录仪）上。而 ONNX Runtime 库非常小巧，只有几十MB。
  2. **跨语言支持**：PyTorch 主要是 Python 语言使用。但工业界很多的硬件控制程序是用 **C++** 写的。ONNX 提供了完美的 C++ 接口，可以直接被 C++ 载入和运行。
  3. **极致优化**：ONNX 格式模型是经过静态图优化的，去除了训练时的冗余信息，运行速度比 PyTorch 原生 `.pt` 快得多。

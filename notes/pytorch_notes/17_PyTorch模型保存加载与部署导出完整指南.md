# PyTorch 模型保存、加载与部署导出完整指南

从底层的 `torch.save` 到线上的 ONNX 推理，一篇文章串起来。

---

## 先看清两条路线

```
PyTorch 底层路线：                   YOLO 框架路线：
torch.save(state_dict)               model.train() → best.pt 自动生成
torch.load + load_state_dict         YOLO("best.pt") 一行加载
torch.onnx.export(...)               model.export(format="onnx")
手写预处理 + ONNX Runtime 推理        自动预处理，不用你管
```

**YOLO 是对 PyTorch 底层的封装。** 你把底层搞懂了，YOLO 的内部就知道在干嘛。

---

## 第 1 步：PyTorch 底层的保存和加载

### 1.1 state_dict 是什么

```python
model = SimpleCNN()

# state_dict 是一个普通 Python 字典
# key = 层的名字，value = 那层的参数（Tensor）
state = model.state_dict()

print(state.keys())
# odict_keys([
#   'features.0.weight',     ← Conv1 的权重  [16, 3, 3, 3]
#   'features.0.bias',       ← Conv1 的偏置  [16]
#   'features.3.weight',     ← Conv2 的权重  [32, 16, 3, 3]
#   'features.3.bias',       ← Conv2 的偏置  [32]
#   'classifier.1.weight',   ← Linear1 的权重 [128, 2048]
#   'classifier.1.bias',     ← Linear1 的偏置 [128]
#   'classifier.3.weight',   ← Linear2 的权重 [10, 128]
#   'classifier.3.bias',     ← Linear2 的偏置 [10]
# ])
```

**state_dict 只存参数值，不存模型结构。** 就像一份"体检报告单"——只有数值，没有骨架信息。

### 1.2 推荐方式：存 state_dict（轻量、灵活）

```python
# ========== 保存 ==========
torch.save(model.state_dict(), "my_cnn.pth")

# ========== 加载 ==========
model = SimpleCNN()                              # 先搭好结构（必须和保存时一样）
model.load_state_dict(torch.load("my_cnn.pth"))  # 把参数灌进去
model.eval()                                     # 切到推理模式
```

**这是官方推荐的方式。** 文件小，跨设备方便。

### 1.3 不推荐但省事：存整个模型

```python
# ========== 保存 ==========
torch.save(model, "my_cnn_full.pth")   # 整个对象序列化

# ========== 加载 ==========
model = torch.load("my_cnn_full.pth")  # 结构和参数一起还原
model.eval()
```

**缺点：** 文件大、依赖 pickle（安全风险）、换个 Python 环境可能加载失败。

### 1.4 训练中断点续训（Checkpoint）

```python
# ========== 保存检查点 ==========
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}
torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")

# ========== 从检查点恢复 ==========
ckpt = torch.load("checkpoint_epoch_25.pth")
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch'] + 1

# 接着训...
for epoch in range(start_epoch, total_epochs):
    ...
```

---

## 第 2 步：GPU/CPU 之间的模型搬运

```python
# 场景 1：在 GPU 上训，保存到 CPU
model = SimpleCNN().cuda()
# ... 训练 ...
torch.save(model.state_dict(), "model_cpu.pth")  # 自动保存到 CPU

# 场景 2：加载到 GPU
model = SimpleCNN()
model.load_state_dict(torch.load("model_cpu.pth"))
model.cuda()

# 场景 3：加载到 CPU（只有 CPU 的机器上）
model = SimpleCNN()
model.load_state_dict(torch.load("model_cpu.pth", map_location="cpu"))
#                                              ↑ 强制加载到 CPU

# 场景 4：GPU 环境间迁移
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model.pth", map_location=device))
```

---

## 第 3 步：推理时怎么用（PyTorch 原生）

```python
import torch
from PIL import Image
from torchvision import transforms

# 1. 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load("my_cnn.pth", map_location="cpu"))
model.eval()

# 2. 读图 + 预处理（和训练时一模一样）
img = Image.open("test.jpg")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
x = transform(img).unsqueeze(0)   # [1, 3, 32, 32] — 加 batch 维

# 3. 推理
with torch.no_grad():
    output = model(x)
    _, predicted = torch.max(output, 1)
    print(f"预测类别: {predicted.item()}")
```

**和训练时唯一区别：** 加了 `model.eval()` + `torch.no_grad()`，丢了 `backward()` 和 `step()`。

---

## 第 4 步：导出到 ONNX（脱离 PyTorch 运行）

### 4.1 为什么需要 ONNX

```
PyTorch 推理：
  必须装 torch + CUDA + 所有依赖 → 部署环境笨重

ONNX Runtime 推理：
  只装 onnxruntime → 轻量、跨平台、可在 C++/手机/网页跑
```

### 4.2 PyTorch 原生 ONNX 导出

```python
import torch.onnx

model = SimpleCNN()
model.load_state_dict(torch.load("my_cnn.pth", map_location="cpu"))
model.eval()

# torch.onnx.export(模型, 示例输入, 保存路径, ...)
dummy_input = torch.randn(1, 3, 32, 32)   # 一个假的输入，用来"追踪"模型结构

torch.onnx.export(
    model,
    dummy_input,
    "my_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},     # batch 维度可变
        "output": {0: "batch"},
    },
    opset_version=17,              # ONNX 算子集版本
)
print("导出完成: my_cnn.onnx")
```

### 4.3 ONNX Runtime 推理（不用 PyTorch！）

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# 1. 加载 ONNX 模型（不需要 torch！）
session = ort.InferenceSession("my_cnn.onnx")

# 2. 读图 + 预处理（和训练时一模一样，但用 numpy 而不是 tensor）
img = Image.open("test.jpg")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
x = transform(img).unsqueeze(0).numpy()   # [1, 3, 32, 32] numpy 数组
#                          ↑ .numpy() ← 从 tensor 转 numpy

# 3. 推理
outputs = session.run(["output"], {"input": x})
predicted = np.argmax(outputs[0])
print(f"ONNX 预测类别: {predicted}")
```

**关键：onnxruntime 不需要 torch。** 部署环境只需要 `pip install onnxruntime`。

---

## 第 5 步：TorchScript（中间方案）

TorchScript 介于 PyTorch 和 ONNX 之间：比 PyTorch 快，比 ONNX 多依赖 torch。

```python
# 方式 1：Trace（跑一遍记录操作）
scripted = torch.jit.trace(model, dummy_input)
scripted.save("model_traced.pt")

# 方式 2：Script（解析代码本身）
scripted = torch.jit.script(model)        # 有些 Python 写法不支持
scripted.save("model_scripted.pt")

# 加载
model = torch.jit.load("model_traced.pt")
output = model(dummy_input)
```

---

## 第 6 步：YOLO 这边是怎么做的

YOLO 把上面所有步骤全封装了：

```python
from ultralytics import YOLO

# 训练 → best.pt 自动生成（整个模型对象序列化，和 torch.save(model) 一样）
model = YOLO("yolo11n.pt")
model.train(data="data.yaml", epochs=50)
# 结果：runs/.../weights/best.pt

# 加载（一行，和 torch.load 一样但多了自动识别架构）
model = YOLO("runs/.../weights/best.pt")

# 预测（内部就是 model.eval() + no_grad() + 后处理）
results = model.predict(source="test.jpg")

# 导出（内部调 torch.onnx.export + 自动预处理后处理适配）
model.export(format="onnx")           # 导出 ONNX
model.export(format="engine")         # 导出 TensorRT
model.export(format="openvino")       # 导出 OpenVINO
```

**YOLO 做了 PyTorch 没做的：后处理和预处理也写进 ONNX 图里。** 所以 YOLO 的 ONNX 模型输入是原始图像，输出直接是检测框——不需要你手动做 NMS。

---

## 对比总结

| | PyTorch 原生 | YOLO 框架 |
|---|---|---|
| 保存 | `torch.save(state_dict)` | `best.pt` 自动生成 |
| 加载 | 建模型 + `load_state_dict` | `YOLO("best.pt")` 一行 |
| 推理预处理 | 自己写（transform） | 框架自动 |
| 推理后处理 | 自己写（argmax/NMS） | 框架自动 |
| ONNX 导出 | `torch.onnx.export()` 手动配 | `model.export("onnx")` |
| ONNX 推理 | 自己写预处理 + NMS | 预处理/NMS 已内置在 ONNX 里 |
| 适合阶段 | 学习底层、小模型部署 | 快速产出、生产部署 |

---

## 你现在该会的

1. 训练完你的 SimpleCNN → `torch.save(model.state_dict(), "my_cnn.pth")`
2. 推理时加载 → `model.load_state_dict(torch.load("my_cnn.pth"))`
3. YOLO 这边 → `YOLO("best.pt")` 一行搞定，但你知道它底层干了啥

---

## 参考

- PyTorch 官方 Save/Load 教程：https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch ONNX 导出文档：https://pytorch.org/docs/stable/onnx.html
- Ultralytics 导出文档：https://docs.ultralytics.com/modes/export/
- 你已有的实战脚本：`02_export_deploy/01_export.py`, `02_onnx_inference.py`

---

## 复习速答

- `state_dict` 保存什么：只保存参数。
- `torch.save(model.state_dict())`：最推荐的保存方式。
- `load_state_dict()`：把参数灌进同结构模型。
- `torch.save(model)`：直接保存整个模型对象，不太推荐。
- `ONNX`：更适合部署推理的通用格式。

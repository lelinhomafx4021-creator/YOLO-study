# 22_onnx 和 pt 测试在运行时的内存区别

在写代码做批量测试时，我们可能会写出这种代码：
```python
from ultralytics import YOLO
# 导入了 YOLO，相当于在后台导入了 PyTorch
model = YOLO("best.onnx") 
results = model(img)
```
**这时候有人会问**：既然都导入了 PyTorch，占用了好几个G的导入库内存，那加载 `.onnx` 运行还有什么省内存的意义呢？

**答案是：意义依然巨大！** 导入库占用的“静态基础内存”只是起步开销；在模型**真正跑前向推理（Inference Runtime）的阶段**，加载 `.onnx` 运行和加载 `.pt` 运行在动态内存和显存分配上，底层有本质的差别。

---

## ⚡ 一、 核心机制对比：为什么 ONNX 推理能省内存？

ONNX Runtime (简称 ORT) 之所以比原生 PyTorch 省动态内存和显存，主要是因为 ORT 是为**推理而生**的静态图引擎，而 PyTorch 默认是为**训练而生**的动态图引擎。

### 1. 算子融合（Node Fusion）—— 中间变量消失术
*   **PyTorch 推理 (`.pt`)**：
    PyTorch 采用**动态图**计算。在推理前向传播时（哪怕加了 `with torch.no_grad()`），它也是一层一层挨个算。
    例如计算：`卷积 ➡️ 批归一化(BN) ➡️ 激活函数(ReLU)`
    PyTorch 会在内存/显存中依次创建：`Tensor A (卷积输出)` ➡️ `Tensor B (BN输出)` ➡️ `Tensor C (ReLU输出)`。这些中间特征图同时驻留在内存中，直到该步骤执行完才被回收。
*   **ONNX Runtime 推理 (`.onnx`)**：
    ORT 在加载模型时，会把计算图编译成**静态图**，并进行**算子融合**。
    它会把 `Conv + BN + ReLU` 这一连串计算直接融合成一个大算子，在底层直接输出最终的 `Tensor C`。
    **中间的 Tensor A 和 Tensor B 根本不会在内存中创建**，这直接省去了大量的中间特征图内存开销！

### 2. 内存重用（Memory Reuse）与静态预分配
*   **PyTorch 推理 (`.pt`)**：
    由于每一层计算是动态分配的，PyTorch 会不断地申请新的 Tensor 空间。
    为了防频繁向系统申请内存变慢，PyTorch 内部有一个 **Caching Allocator (缓存分配器)**，占了内存就不愿意还给操作系统。即使推理完成了，内存里依然有一大坨被预留（Reserved）的缓存，这会导致运行期内存曲线波动大且基数高。
*   **ONNX Runtime 推理 (`.onnx`)**：
    静态图模型在加载 `Session` 时，各层的尺寸和结构都已经定死了。
    ORT 能够**提前精准算出一张图在前向推理时，最多需要多大的缓冲区**。
    它只开辟一块足够大的内存池，在推理时，**后面的层会直接覆盖和复用前面已经计算完的内存块（即地址复用）**。这种极致的内存重用机制，使得 ORT 在运行时的动态内存开销基本是一条极其稳定的水平线。

### 3. 显存分配器（VRAM）的饥饿感
如果在 GPU 显卡（CUDA）环境下跑测试：
*   **PyTorch**：不管三七二十一，启动 CUDA 运行时和内部缓存池，先占大约 **800MB ~ 1GB 的显存（VRAM）**，极度贪婪。
*   **ONNX Runtime**：对显存非常克制，有多少数据就申请多少显存，几乎没有额外的显存缓冲池开销。

---

## 📊 二、 推理运行期内存状态对比

| 内存特性 | PyTorch 原生推理 (`.pt`) | ONNX Runtime 推理 (`.onnx`) |
| :--- | :--- | :--- |
| **内存/显存分配机制** | 动态分配 + 贪婪的 Caching 缓冲池（占着不还） | 静态预分配，仅保留必要的前向通道空间 |
| **中间临时变量** | 必须按层逐步生成中间 Tensor | 依靠**算子融合**合并计算，干掉中间特征图 |
| **空间重用** | 靠自动垃圾回收，容易产生内存碎片 | 引擎级**内存重用**，后面层直接覆盖复用前面地址 |
| **运行时显存 (VRAM)** | 极高（CUDA Context 基础开销大） | 极低（按需精准申请，无多余缓存） |

---

## 🔧 三、亲手验证：写代码测一下

```python
import time, os, psutil, onnxruntime as ort
import torch, numpy as np
from ultralytics import YOLO

process = psutil.Process(os.getpid())

def mem_used():
    """返回当前进程的内存占用（MB）"""
    return process.memory_info().rss / 1024**2

# ── 1. 测 PyTorch .pt 推理内存 ──
torch.cuda.empty_cache() if torch.cuda.is_available() else None
mem_before = mem_used()
model_pt = YOLO("best.pt").to("cuda" if torch.cuda.is_available() else "cpu")

# 跑 10 次推理，取平均时间
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
t0 = time.time()
for _ in range(10):
    results = model_pt.predict(source=dummy, verbose=False, conf=0.25)
t_pt = (time.time() - t0) / 10
mem_pt = mem_used() - mem_before
print(f".pt 推理: {t_pt*1000:.1f}ms/张, 内存增量: {mem_pt:.0f}MB")

# ── 2. 测 ONNX .onnx 推理内存 ──
del model_pt; torch.cuda.empty_cache() if torch.cuda.is_available() else None

mem_before = mem_used()
session = ort.InferenceSession("best.onnx",
    providers=['CUDAExecutionProvider','CPUExecutionProvider'])

# ONNX 需要手动预处理（和训练时一致）
img = cv2.resize(dummy, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

t0 = time.time()
for _ in range(10):
    session.run(None, {"images": img})
t_onnx = (time.time() - t0) / 10
mem_onnx = mem_used() - mem_before
print(f".onnx 推理: {t_onnx*1000:.1f}ms/张, 内存增量: {mem_onnx:.0f}MB")
```

你会在终端看到 `.pt` 的内存增量明显大于 `.onnx` — 这就是算子融合和静态内存预分配的效果。

---

## 🎯 四、 面试通关密语（如何向面试官装杯？）

**面试官**：你之前把 YOLO 导出成了 ONNX，既然你的代码顶层还是导了 `ultralytics` 和 `torch`，那你用 ONNX 推理和用 PT 推理还有区别吗？

**你应该这样回答**：
> “依然有很大区别。
> 
> 第一，虽然 Python 脚本导入 PyTorch 会带来较大的基础库载入内存，但是在**模型运行推理（Forward）的执行期**，两者的内存和显存管理机制截然不同。
> 
> 第二，`.pt` 采用 PyTorch 原生的动态图计算，每一层计算都会动态申请内存，且因为 PyTorch 内部 `Caching Allocator` 的缓存机制，会占着内存/显存不放。
> 
> 第三，`.onnx` 经由 **ONNX Runtime** 静态图引擎执行，在初始化时就完成了**算子融合（Node Fusion）**，融合并消除了卷积、归一化和激活层之间的中间特征图变量。同时，它支持**内存地址重用（Memory Reuse）**，后面层可以直接复用前面层已经结束计算的缓冲区。
> 
> 因此，在实际批量测试和运行过程中，ONNX 推理的动态内存峰值和 GPU 显存占用依然显著低于 PyTorch 原生推理。”

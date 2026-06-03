# PyTorch 是什么

## 一句话

PyTorch 是一个 Python 库，让你能用显卡（GPU）做数学计算，专门用来训练神经网络。

## 你已经知道的东西

你之前用 OpenCV 读图，图片在 Python 里长这样：

```python
img = cv2.imread("test.jpg")
print(img.shape)   # (1000, 1000, 3)
```

这是一个 numpy 数组，1000 行 × 1000 列 × 3 个颜色通道。

**PyTorch 里的 Tensor 和这个几乎一样，只是它能在显卡上跑。**

## Tensor 是什么

Tensor 就是"多维数组"，和 numpy array 长得一模一样：

```python
import numpy as np
import torch

# numpy 的写法
a = np.array([1, 2, 3])

# pytorch 的写法
b = torch.tensor([1, 2, 3])
```

唯一的区别：Tensor 能放到 GPU 上，numpy 不行。

```python
# numpy 只能在 CPU 上算
a = np.array([1, 2, 3])          # 只能在 CPU

# tensor 可以放到 GPU 上算
b = torch.tensor([1, 2, 3])      # 默认在 CPU
b = b.to('cuda')                 # 移到 GPU
```

**GPU 算矩阵比 CPU 快几十倍，所以深度学习都用 Tensor。**

## 为什么需要 GPU

想象你要算 100 万次乘法：

- CPU：只有几个核心，一个一个算，要算很久
- GPU：有几千个小核心，同时算，快几十倍

训练神经网络就是在不断算矩阵乘法，所以需要 GPU。

---

## 🚀 深度补充一：动态计算图（Dynamic Computational Graph）
这是 PyTorch 最引以为傲的“大杀器”，也是面试时常问的底层机制。

### 什么是计算图？
当你写下以下代码时：
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3
```
PyTorch 在后台悄悄画了一张有向图（就像一张关系网）：
```
[ x ] ──(平方运算)──> [ x² ] ──(+3运算)──> [ y ]
```
当你调用 `y.backward()` 时，PyTorch 就会顺着这张图**反着走**，自动利用导数公式算出 `y` 对 `x` 的梯度。

### 动态图 vs 静态图
* **静态图（例如旧版 TensorFlow 1.x）**：你需要先用代码“声明”好一整张计算图，不能有任何 Python 的 `if` 分支或 `for` 循环，声明好之后再喂数据运行。一旦图定死了，训练过程中不能改，调试极其痛苦，报错信息根本看不懂。
* **动态图（PyTorch）**：**定义即运行（Define-by-Run）**。你每用 Tensor 做一次加减乘除，计算图就在内存中**实时构建**出来。这允许我们在模型里自由地写 `if-else` 分支（比如根据概率决定走哪个卷积层），还能像调试普通 Python 程序一样加断点调试，体验极佳。

---

## 🚀 深度补充二：GPU 显存的管理与释放（OOM 终结者）

在跑 YOLO 训练或者推理时，你一定会遇到这个噩梦：
`RuntimeError: CUDA out of memory. Tried to allocate...` (俗称 OOM，显存溢出)

### 显存都去哪了？
1. **模型权重**：网络层数越多，参数量越大，占用的基础显存就越多。
2. **中间特征图（Feature Maps）**：前向传播时，每一层卷积算出来的中间矩阵都必须保存在显存里，用来在反向传播时算梯度。图片越大（`imgsz` 大）或者 Batch Size 越大，这些中间矩阵就呈指数级膨胀！
3. **PyTorch 缓存**：PyTorch 底层有一个**显存垃圾回收缓存区**（Memory Caching Allocator）。当你把一个 Tensor 删掉后，PyTorch 并不会立刻把显存还给 Windows 系统，而是缓存着，方便下次快速申请。

### 实战清理显存的代码：
当你手动运行了推理后，想清空没用的显存，可以使用这行代码：
```python
import torch
import gc

# 1. 删掉没用的 Tensor 变量
del my_large_tensor

# 2. 强制 Python 进行垃圾回收
gc.collect()

# 3. 释放 PyTorch 占用的未释放缓存显存
torch.cuda.empty_cache()
```
**注意**：`torch.cuda.empty_cache()` **不能**让你在训练时免于 OOM（因为训练时的特征图是必须占着显存的），但它能在你完成训练或推理后，把被占着的显存归还给显卡，防止别的软件打不开。

---

## PyTorch 和 YOLO 的关系

你已经知道的：

```
YOLO = 目标检测模型（找图片里的东西）
Ultralytics = 跑 YOLO 的工具包
```

它们的关系：

```
Ultralytics 工具包
    │
    ├── YOLO 的模型结构（用 PyTorch 写的）
    ├── 数据处理代码（用 PyTorch 写的）
    ├── 训练循环代码（用 PyTorch 写的）
    └── 导出功能（用 PyTorch 写的）

所有代码都是用 PyTorch 写的
```

**YOLO 就是用 PyTorch 写的一个程序。** Ultralytics 帮你把几百行 PyTorch 代码封装成了 3 行调用。

## PyTorch 和 CUDA 的关系

```
你的 Python 代码
      ↓
   PyTorch（做数学计算）
      ↓
   CUDA（让 PyTorch 能用显卡）
      ↓
   GPU（显卡硬件）
```

- PyTorch：做计算的库
- CUDA：让 PyTorch 能指挥显卡的接口
- GPU：显卡硬件

**你装 PyTorch 时选了 cu118，就是告诉 PyTorch："用 CUDA 11.8 版本的接口来指挥显卡。"**

## 你现在需要记住的

| 概念 | 一句话解释 |
|------|-----------|
| Tensor | 能在 GPU 上跑的数组，和 numpy array 长得一样 |
| PyTorch | 做深度学习计算的 Python 库，采用**动态计算图**机制 |
| CUDA | 让 PyTorch 能用显卡的接口 |
| GPU / 显存 | 显卡硬件，显存大小限制了你的 Batch Size 和 imgsz |

**不用记住更多，先记住这几个。** 后面遇到再补充。

# Tensor 入门

## Tensor 是什么

Tensor 就是"多维数组"。你用 OpenCV 读的图片，本质就是一个数组：

```python
import cv2
img = cv2.imread("test.jpg")
print(img.shape)   # (1000, 1000, 3)  → 1000行 × 1000列 × 3个颜色
```

Tensor 和这个一模一样，只是换了个名字：

```python
import torch
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t.shape)    # torch.Size([2, 3])  → 2行 × 3列
```

## 不同维度的 Tensor

| 维度 | 长什么样 | 例子 | 实际含义 |
|------|---------|------|---------|
| 0 维 | 一个数 | `tensor(3.14)` | 损失值（loss）就是一个数 |
| 1 维 | 一行数 | `tensor([1, 2, 3])` | 一个样本的特征 |
| 2 维 | 一个表格 | `tensor([[1,2], [3,4]])` | 一批样本 |
| 3 维 | 一个立方体 | `(3, 28, 28)` | 一张彩色图片（通道×高×宽） |
| 4 维 | 一批立方体 | `(32, 3, 28, 28)` | 一批 32 张彩色图片 |

## 创建 Tensor

```python
import torch

# 从列表创建
a = torch.tensor([1, 2, 3])           # [1, 2, 3]

# 创建全 0
b = torch.zeros(3, 4)                  # 3行4列全是0

# 创建全 1
c = torch.ones(3, 4)                   # 3行4列全是1

# 创建随机数（0~1 之间）
d = torch.rand(3, 4)                   # 3行4列随机数

# 创建随机数（正态分布）
e = torch.randn(3, 4)                  # 均值0，标准差1

# 指定范围
f = torch.arange(0, 10, 2)             # [0, 2, 4, 6, 8]
```

## 查看 Tensor 信息

```python
x = torch.rand(3, 4, 5)

x.shape          # torch.Size([3, 4, 5])  → 形状
x.dtype          # torch.float32           → 数据类型
x.device         # cpu                     → 在哪个设备上
x.ndim           # 3                       → 几维
x.numel()        # 60                      → 总共多少个数（3×4×5）
```

## 索引和切片（和 numpy 一样）

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

x[0]             # [1, 2, 3]        第 0 行
x[1]             # [4, 5, 6]        第 1 行
x[:, 0]          # [1, 4, 7]        第 0 列
x[0, 2]          # 3                第 0 行第 2 列
x[0:2]           # [[1,2,3], [4,5,6]]  前 2 行
```

## 形状变换与面试终极考点

```python
x = torch.rand(3, 4)           # (3, 4)

# reshape：改变形状
x.reshape(2, 6)                # (2, 6)   重新排列
x.reshape(12)                  # (12,)    展平成一维
x.reshape(-1, 6)               # (2, 6)   -1 表示自动算

# squeeze：去掉大小为 1 的维度
y = torch.rand(1, 3, 1, 4)
y.squeeze()                    # (3, 4)   去掉所有 1

# unsqueeze：增加一个维度
x.unsqueeze(0)                 # (1, 3, 4)  在第 0 维加 1
x.unsqueeze(1)                 # (3, 1, 4)  在第 1 维加 1
```

---

### 🚀 形状变换面试必问一：`.view()` 和 `.reshape()` 到底有什么区别？
这是 PyTorch 极其经典的面试底层原理考点。

* **底层概念：内存连续性（Contiguous）**
  Tensor 在物理内存中其实是**一维扁平化**存储的一连串数据。当我们在 PyTorch 里做了一些旋转、转置操作（比如 `transpose`）后，Tensor 的形状变了，但底层的物理内存数据其实**没有跟着变**，这就导致了“逻辑上的维度”和“物理上的内存排列”不连续了，这就是**非连续 Tensor (Non-contiguous)**。
* **`.view()` 的物理限制**：
  `.view()` **不复制内存**，它只是创建了一个指向原内存的新“视图”。因为不复制内存，所以它要求 Tensor 在内存中**必须是连续的**。如果原 Tensor 是转置过的非连续状态，调用 `.view()` 会直接报错：
  `RuntimeError: input is not contiguous`。
  * **解决办法**：在调用 `.view()` 前先调用 `.contiguous()`（即 `x.contiguous().view(...)`），这会强制 PyTorch 复制一份内存，让它在物理上变连续。
* **`.reshape()` 的防脑残设计**：
  `.reshape()` 是 PyTorch 后期加入的函数，它相当于自动帮你在底层做了判断：
  * 如果内存是连续的，它直接调用 `.view()`，不复制内存，速度极快。
  * 如果内存不连续，它自动在后台调用 `.contiguous()` 帮你复制一份连续的内存再做 view，绝不报错。
* **面试官追问：既然 `.reshape()` 这么智能，那我们写代码全用 `.reshape()` 岂不是爽歪歪？**
  * **答**：**并不推荐全用 `reshape`**。因为 `reshape` 在悄悄复制内存时是**没有任何提示**的。这可能导致我们在不知不觉中写了许多复制内存的操作，白白消耗大量的 CPU/GPU 内存与时间。全用 `.view()` 可以帮助我们在写出低效代码时，通过抛错“立刻发现并重构它”。

---

### 🚀 形状变换面试必问二：`.transpose()` 和 `.permute()` 的区别？
我们在 YOLO 源码中经常看到这俩兄弟。

* **核心区别：通道数不同**
  * **`.transpose()`**：只能交换 **2 个维度**。比如把图片的宽高转置：`x.transpose(1, 2)`。
  * **`.permute()`**：可以同时重新排列 **多个（任意多）维度**。比如把一堆图片从 YOLO 默认的 `(Batch, 通道, 高, 宽)` 重排成 Matplotlib 绘图要求的 `(Batch, 高, 宽, 通道)`：
    ```python
    # YOLO 格式: [16, 3, 640, 640]
    x = torch.rand(16, 3, 640, 640)
    # 重排为: [16, 640, 640, 3]
    y = x.permute(0, 2, 3, 1)  # 填入旧维度的索引顺序
    ```
* **相同点**：这俩兄弟都**只是返回了一个视图，并没有物理移动内存**。所以执行这两个操作后，Tensor 都会变成 **“非连续（Non-contiguous）”** 状态。如果你想对它们再次调用 `.view()`，必须在中间加上 `.contiguous()`：
  `x.permute(0, 2, 3, 1).contiguous().view(...)`。


## 数学运算

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 加减乘除
x + y              # [5, 7, 9]
x - y              # [-3, -3, -3]
x * y              # [4, 10, 18]
x / y              # [0.25, 0.4, 0.5]

# 求和、均值、最大值
x.sum()            # 6
x.mean()           # 2
x.max()            # 3
```

## CPU 和 GPU 互移

```python
x = torch.tensor([1, 2, 3])    # 默认在 CPU

# 移到 GPU
x = x.to('cuda')
x = x.cuda()                   # 另一种写法

# 移回 CPU
x = x.to('cpu')
x = x.cpu()
```

**两个 Tensor 必须在同一个设备上才能运算：**

```python
a = torch.tensor([1, 2, 3]).to('cuda')
b = torch.tensor([4, 5, 6])              # 在 CPU
# a + b  → 报错！不在同一个设备
b = b.to('cuda')
a + b    → 正确
```

## Tensor 和 numpy 互转

```python
# Tensor → numpy
x = torch.tensor([1, 2, 3])
n = x.numpy()                  # [1, 2, 3]

# numpy → Tensor
import numpy as np
n = np.array([1, 2, 3])
x = torch.from_numpy(n)        # tensor([1, 2, 3])
```

## 你只需要记住的

| 操作 | 代码 | 说明 |
|------|------|------|
| 创建 | `torch.tensor([1,2,3])` | 从列表创建 |
| 随机 | `torch.rand(3,4)` | 3行4列随机数 |
| 全0 | `torch.zeros(3,4)` | 3行4列全0 |
| 形状 | `x.shape` | 查看形状 |
| 索引 | `x[0]`, `x[:,0]` | 取行、取列 |
| 移到GPU | `x.to('cuda')` | 移到显卡 |

**先记住这些，够用了。** 后面用到再查。

---

## 复习速答

- `Tensor`：PyTorch 的多维数组。
- `shape`：Tensor 的维度大小。
- `to('cuda')`：把 Tensor 放到显卡上算。
- `view/reshape`：改形状，`reshape` 更智能。
- `transpose/permute`：换维度顺序，`permute` 更自由。

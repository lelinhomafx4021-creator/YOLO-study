# Tensor 基础学习文档

## 这份文档适合谁

这份文档适合刚接触深度学习、刚听过梯度和反向传播、准备开始学 `PyTorch Tensor` 的人。

目标不是一次学全，而是先把最基础、最常用、最不容易绕的部分学明白。

---

## 1. 先说结论：Tensor 是什么

`Tensor` 可以先理解成：

**PyTorch 里的多维数组。**

它和普通 Python 列表不一样，也和一般数字不一样。

它的特点是：

- 能表示标量、向量、矩阵和更高维的数据
- 能参与高效的数学运算
- 能放到 GPU 上计算
- 能记录梯度，用于反向传播

如果只记一句话，就记这句：

**Tensor 是深度学习里装数据、装参数、装梯度的核心容器。**

---

## 2. 为什么学 Tensor

你前面已经接触到这些概念：

- 输入数据
- 模型参数
- 预测值
- loss
- 梯度

在 `PyTorch` 里，这些东西几乎都是 `Tensor`。

比如：

- 图片可以是 `Tensor`
- 一批样本可以是 `Tensor`
- 模型里的权重可以是 `Tensor`
- loss 可以是 `Tensor`
- `backward()` 算出来的梯度也会放在 `Tensor` 里

所以学 Tensor，不是在背 API，而是在学习深度学习代码最基本的语言。

---

## 3. Tensor 和你熟悉的东西怎么对应

先做一个简单对应：

- 一个数字：0 维 Tensor
- 一排数字：1 维 Tensor
- 一个表格：2 维 Tensor
- 一堆表格：3 维及以上 Tensor

例如：

```python
import torch

a = torch.tensor(3.0)
b = torch.tensor([1, 2, 3])
c = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
```

这里：

- `a` 是一个数
- `b` 是一行数
- `c` 是一个 2 行 3 列的表格

---

## 4. 第一个必须会的 API：`torch.tensor`

它的作用是：

**把数据创建成 Tensor**

```python
import torch

x = torch.tensor([1, 2, 3])
print(x)
print(type(x))
```

输出：

```python
tensor([1, 2, 3])
<class 'torch.Tensor'>
```

这说明：

- `x` 已经不是普通列表
- `x` 是 `torch.Tensor`

---

## 5. 看懂 `shape`

`shape` 是最重要的属性之一。

它表示：

**这个 Tensor 的结构长什么样。**

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(a.shape)
print(b.shape)
```

输出：

```python
torch.Size([3])
torch.Size([2, 3])
```

理解：

- `[3]` 表示这一维有 3 个数
- `[2, 3]` 表示 2 行 3 列

你现在先只要稳稳记住两个：

- `torch.Size([3])`：一维，3 个元素
- `torch.Size([2, 3])`：二维，2 行 3 列

一开始不用急着钻更高维。

---

## 6. 常见创建方式

除了 `torch.tensor(...)`，还有一些很常用的创建方法。

### 6.1 `torch.zeros`

创建全 0 Tensor

```python
import torch

x = torch.zeros(2, 3)
print(x)
```

输出：

```python
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

### 6.2 `torch.ones`

创建全 1 Tensor

```python
x = torch.ones(2, 2)
print(x)
```

输出：

```python
tensor([[1., 1.],
        [1., 1.]])
```

### 6.3 `torch.rand`

创建 0 到 1 之间的随机数

```python
x = torch.rand(2, 3)
print(x)
```

输出每次会不一样，例如：

```python
tensor([[0.13, 0.51, 0.84],
        [0.27, 0.92, 0.44]])
```

### 6.4 `torch.arange`

按顺序生成数字

```python
x = torch.arange(0, 6)
print(x)
```

输出：

```python
tensor([0, 1, 2, 3, 4, 5])
```

---

## 7. 常见属性

学习 Tensor 时，先盯住这几个属性就够了。

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.shape)
print(x.ndim)
print(x.dtype)
```

输出：

```python
torch.Size([2, 3])
2
torch.int64
```

解释：

- `shape`：形状
- `ndim`：几维
- `dtype`：数据类型

---

## 8. 索引和切片

这一步非常重要，因为后面训练时你经常会取一行、取一列、取一个元素。

```python
import torch

x = torch.tensor([[10, 20, 30],
                  [40, 50, 60]])

print(x[0])
print(x[0, 1])
print(x[:, 1])
print(x[1, :])
```

输出：

```python
tensor([10, 20, 30])
tensor(20)
tensor([20, 50])
tensor([40, 50, 60])
```

理解：

- `x[0]`：第 0 行
- `x[0, 1]`：第 0 行第 1 列
- `x[:, 1]`：所有行的第 1 列
- `x[1, :]`：第 1 行全部元素

---

## 9. 基本运算

Tensor 最常见的使用方式就是做运算。

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
```

输出：

```python
tensor([5., 7., 9.])
tensor([-3., -3., -3.])
tensor([ 4., 10., 18.])
tensor([0.2500, 0.4000, 0.5000])
```

这些都是：

**逐元素运算**

也就是对应位置分别计算。

---

## 10. `reshape`：改形状

有时数据内容不变，但我们想改成另一种排列方式，这时就用 `reshape`。

```python
import torch

x = torch.arange(6)
print(x)

y = x.reshape(2, 3)
print(y)
```

输出：

```python
tensor([0, 1, 2, 3, 4, 5])
tensor([[0, 1, 2],
        [3, 4, 5]])
```

理解：

- 原来是一维
- 现在变成 2 行 3 列
- 数据没变，只是组织方式变了

---

## 11. 求和和平均

这类操作在训练和统计里经常会用到。

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(x.sum())
print(x.mean())
```

输出：

```python
tensor(21.)
tensor(3.5000)
```

还可以指定维度：

```python
print(x.sum(dim=0))
print(x.sum(dim=1))
```

输出：

```python
tensor([5., 7., 9.])
tensor([ 6., 15.])
```

理解：

- `dim=0`：按列聚合
- `dim=1`：按行聚合

一开始先记结论，不急着深挖。

---

## 12. Tensor 和 GPU

这是 Tensor 很重要的一个优势。

普通列表不能直接放到 GPU 上算，但 Tensor 可以。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])

if torch.cuda.is_available():
    x = x.to("cuda")
    print(x.device)
```

如果你的电脑支持 CUDA，输出可能是：

```python
cuda:0
```

这说明 Tensor 已经被放到 GPU 上了。

现在你先知道：

- `cpu`：在 CPU 上算
- `cuda`：在 GPU 上算

---

## 13. 为什么 Tensor 和梯度有关系

你前面已经接触到：

- 正向传播
- loss
- 反向传播
- 梯度

那 Tensor 在这里扮演什么角色？

答案是：

**PyTorch 用 Tensor 记录计算，并把梯度存到 Tensor 里。**

如果一个 Tensor 需要参与求导，就要打开：

```python
requires_grad=True
```

---

## 14. 第一个梯度例子

这是最值得你亲手敲一遍的例子。

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
target = torch.tensor(10.0)

y = w * x
loss = (y - target) ** 2

print(y)
print(loss)

loss.backward()

print(w.grad)
```

输出：

```python
tensor(6., grad_fn=<MulBackward0>)
tensor(16., grad_fn=<PowBackward0>)
tensor(-24.)
```

一步一步理解：

1. `w = 2`
2. `x = 3`
3. 所以预测值 `y = 6`
4. 真实值是 `10`
5. 所以误差是 `(6 - 10)^2 = 16`
6. `loss.backward()` 会自动算 `w` 的梯度
7. `w.grad = -24`

这个梯度的意义可以先这样理解：

**当前 `w` 太小了，往更大的方向改，会让 loss 下降。**

这里你不用先去抠公式，先把流程看懂最重要。

---

## 15. 这几个词你一定要分清

### 15.1 `requires_grad=True`

意思是：

**这个 Tensor 需要被跟踪，用来求梯度。**

一般模型参数会开这个。

### 15.2 `loss.backward()`

意思是：

**从 loss 开始做反向传播，自动计算梯度。**

### 15.3 `.grad`

意思是：

**梯度会存到这个位置。**

例如：

```python
print(w.grad)
```

---

## 16. 正向传播和反向传播在代码里怎么对应

看这一段：

```python
y = w * x
loss = (y - target) ** 2
loss.backward()
```

这里：

- `y = w * x` 是正向传播的一部分
- `loss = ...` 是计算误差
- `loss.backward()` 是反向传播

也就是说：

- 前面负责“算出答案”
- 后面负责“知道怎么改”

---

## 17. 一个最小训练流程长什么样

你现在先不用写完整网络，但要先看懂最小训练流程长什么样。

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
target = torch.tensor(10.0)
lr = 0.1

for step in range(3):
    y = w * x
    loss = (y - target) ** 2

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad

    w.grad.zero_()

    print(f"step={step}, w={w.item():.4f}, loss={loss.item():.4f}")
```

这个例子展示了训练的核心逻辑：

1. 正向传播算预测
2. 算 loss
3. 反向传播算梯度
4. 用梯度更新参数
5. 清空旧梯度

你现在只要先知道这就是训练的骨架。

---

## 18. 初学 Tensor 时最该会的东西

如果你能熟练掌握下面这些，就已经打下很好的基础了：

1. `torch.tensor`
2. `torch.zeros / ones / rand / arange`
3. `shape / ndim / dtype`
4. 索引切片
5. 基本运算
6. `reshape`
7. `sum / mean`
8. `requires_grad`
9. `backward`
10. `.grad`

---

## 19. 不要一开始就纠结的东西

初学时，先别把精力放在这些地方：

- 高维张量特别复杂的形状
- 太多零散 API
- 多卡训练
- 特别复杂的数学推导
- Transformer 结构

现在更重要的是：

**把 Tensor 当成训练流程里的基础数据结构看懂。**

---

## 20. 建议的学习顺序

你可以按这个顺序学：

1. 创建 Tensor
2. 看 shape
3. 做索引和切片
4. 做加减乘除
5. 做 reshape
6. 做 sum 和 mean
7. 理解 `requires_grad=True`
8. 跑通一个 `loss.backward()` 小例子

这个顺序是为了先有直觉，再接训练。

---

## 21. 一组最小练习

建议你自己敲一遍，不要只看。

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

print(a)
print(a.shape)

print(b)
print(b.shape)

print(b[0])
print(b[:, 1])

print(a + 10)
print(a * 2)

print(torch.arange(6).reshape(2, 3))

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
target = torch.tensor(10.0)

y = w * x
loss = (y - target) ** 2
loss.backward()

print(loss)
print(w.grad)
```

---

## 22. 学完这份文档后，你应该能回答的问题

如果下面这些问题你都能大致回答，说明你已经入门了：

1. Tensor 是什么
2. `shape` 是什么
3. `torch.tensor` 是干什么的
4. 为什么深度学习里几乎到处都是 Tensor
5. `requires_grad=True` 是干什么的
6. `loss.backward()` 在做什么
7. 梯度最后存在哪里

---

## 23. 一句话收尾

不要把 Tensor 当成一堆零散 API。

更好的理解方式是：

**Tensor 是 PyTorch 中承载数据、参数、计算结果和梯度的统一对象。**

只要这句话你越来越有感觉，后面学模型、训练、YOLO、部署都会顺很多。

---

## 24. 当前阶段必须记住的训练总结

下面这几句话，是你现在这个阶段最应该记住的。

### 24.1 `loss` 是什么

`loss` 是一个数，用来表示：

**模型这次错了多少。**

`loss` 越大，说明错得越多。  
`loss` 越小，说明预测越接近目标。

---

### 24.2 `gradient` 是什么

梯度就是：

**参数应该往哪个方向改，才能让 loss 变小。**

例如：

```python
w.grad = d(loss) / d(w)
```

它表示：

**如果 `w` 变化一点点，`loss` 会怎么变化。**

你现在先记这条最重要：

- `w.grad > 0`：说明 `w` 应该变小
- `w.grad < 0`：说明 `w` 应该变大

---

### 24.3 `loss.backward()` 在做什么

这一步是在做：

**从 loss 开始反向传播，自动计算所有需要求导参数的梯度。**

如果某个 Tensor 设置了：

```python
requires_grad=True
```

那么它在 `backward()` 之后就会得到自己的梯度。

---

### 24.4 梯度最后存在哪里

梯度默认存在对应参数的：

```python
.grad
```

比如：

```python
print(w.grad)
```

这就是查看参数 `w` 的梯度。

---

### 24.5 学习率 `lr` 是什么

学习率控制的是：

**参数每次改多大。**

它不负责决定方向，方向由梯度决定。

你可以记成：

- 梯度：指方向
- 学习率：控步长
- 参数：真正被更新

---

### 24.6 参数更新公式

训练时最经典的参数更新公式是：

```python
w = w - lr * w.grad
```

含义是：

1. 先通过 `loss.backward()` 算出 `w.grad`
2. 再用学习率 `lr` 控制更新幅度
3. 最后真正更新参数 `w`

---

### 24.7 当前阶段的正确理解

你现在最应该记住的不是复杂公式，而是这条训练主线：

```python
参数 w
-> 正向传播得到预测值
-> 计算 loss
-> backward() 算梯度
-> 用 lr 和 grad 更新参数
-> loss 逐步下降
```

再压缩成一句：

**loss 负责告诉你错多少，grad 负责告诉你怎么改，lr 负责控制改多猛。**

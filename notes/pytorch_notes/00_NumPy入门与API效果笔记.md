# NumPy 入门与 API 效果笔记

## 这份笔记是干什么的

这份笔记不是为了背语法，而是为了先把后面学 Tensor 必须会的底层感觉练出来。

你学习每个 NumPy API 时，都重点看 4 件事：

1. 这个数组的 `shape` 是什么
2. 这个操作是逐元素，还是矩阵级别
3. 这个函数沿着哪个 `axis` 计算
4. 这里有没有发生 `broadcasting`

---

## 1. NumPy 最核心的对象

NumPy 最重要的对象是 `ndarray`，也就是多维数组。

```python
import numpy as np

a = np.array([1, 2, 3])
print(a)
print(type(a))
print(a.shape)
print(a.dtype)
```

输出：

```python
[1 2 3]
<class 'numpy.ndarray'>
(3,)
int64
```

你现在先记住：

- `np.array(...)`：把列表变成 NumPy 数组
- `shape`：数组形状
- `dtype`：数据类型

---

## 2. 一维、二维、三维分别长什么样

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3],
              [4, 5, 6]])
c = np.array([[[1], [2]],
              [[3], [4]]])

print(a.shape)
print(b.shape)
print(c.shape)
```

输出：

```python
(3,)
(2, 3)
(2, 2, 1)
```

理解：

- `(3,)`：1 维，3 个元素
- `(2, 3)`：2 行 3 列
- `(2, 2, 1)`：3 维数组

这一步就是张量思维的起点。

---

## 3. 常用创建 API

### 3.1 `np.array`

作用：把 Python 列表转成数组

```python
import numpy as np

x = np.array([1, 2, 3, 4])
print(x)
print(x.shape)
```

输出：

```python
[1 2 3 4]
(4,)
```

### 3.2 `np.zeros`

作用：创建全 0 数组

```python
x = np.zeros((2, 3))
print(x)
```

输出：

```python
[[0. 0. 0.]
 [0. 0. 0.]]
```

### 3.3 `np.ones`

作用：创建全 1 数组

```python
x = np.ones((2, 2))
print(x)
```

输出：

```python
[[1. 1.]
 [1. 1.]]
```

### 3.4 `np.full`

作用：创建指定值填充的数组

```python
x = np.full((2, 3), 7)
print(x)
```

输出：

```python
[[7 7 7]
 [7 7 7]]
```

### 3.5 `np.arange`

作用：按步长生成数字

```python
x = np.arange(0, 10, 2)
print(x)
```

输出：

```python
[0 2 4 6 8]
```

### 3.6 `np.linspace`

作用：在区间里平均切出若干个点

```python
x = np.linspace(0, 1, 5)
print(x)
```

输出：

```python
[0.   0.25 0.5  0.75 1.  ]
```

### 3.7 `np.random.rand`

作用：生成 0 到 1 之间的随机数

```python
x = np.random.rand(2, 3)
print(x.shape)
print(x)
```

可能输出：

```python
(2, 3)
[[0.21 0.87 0.43]
 [0.65 0.12 0.91]]
```

注意：随机数每次运行结果都不同，但 `shape` 一定是 `(2, 3)`。

---

## 4. 查看数组信息的 API

```python
import numpy as np

x = np.array([[1, 2, 3],
              [4, 5, 6]])

print(x.shape)
print(x.ndim)
print(x.size)
print(x.dtype)
```

输出：

```python
(2, 3)
2
6
int64
```

解释：

- `shape`：形状
- `ndim`：几维
- `size`：总元素个数
- `dtype`：元素类型

---

## 5. 索引和切片 API

```python
import numpy as np

x = np.array([[10, 20, 30],
              [40, 50, 60]])

print(x[0, 1])
print(x[1, 2])
print(x[:, 1])
print(x[0, :])
print(x[:, :2])
```

输出：

```python
20
60
[20 50]
[10 20 30]
[[10 20]
 [40 50]]
```

理解：

- `x[0, 1]`：第 0 行第 1 列
- `x[:, 1]`：所有行的第 1 列
- `x[:, :2]`：所有行的前 2 列

后面学 Tensor，你会天天用这个。

---

## 6. 逐元素运算 API

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** 2)
```

输出：

```python
[5 7 9]
[-3 -3 -3]
[ 4 10 18]
[0.25 0.4  0.5 ]
[1 4 9]
```

这类都是逐元素运算。

也就是：

- 第一个位置和第一个位置算
- 第二个位置和第二个位置算
- 第三个位置和第三个位置算

---

## 7. 广播机制 `broadcasting`

这是必须搞懂的重点。

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

print(a + b)
```

输出：

```python
[[11 22 33]
 [14 25 36]]
```

为什么能加？

因为：

- `a.shape == (2, 3)`
- `b.shape == (3,)`

NumPy 会把 `b` 自动“拉伸理解”为：

```python
[[10 20 30]
 [10 20 30]]
```

然后再做逐元素相加。

这就叫广播。

---

## 8. 形状变换 API

### 8.1 `reshape`

作用：改变数组形状，不改数据内容

```python
import numpy as np

x = np.arange(6)
print(x)
print(x.reshape(2, 3))
print(x.reshape(3, 2))
```

输出：

```python
[0 1 2 3 4 5]
[[0 1 2]
 [3 4 5]]
[[0 1]
 [2 3]
 [4 5]]
```

### 8.2 `flatten`

作用：拉平成一维

```python
x = np.array([[1, 2],
              [3, 4]])
print(x.flatten())
```

输出：

```python
[1 2 3 4]
```

### 8.3 `transpose` 或 `.T`

作用：转置，行列交换

```python
x = np.array([[1, 2, 3],
              [4, 5, 6]])

print(x.T)
```

输出：

```python
[[1 4]
 [2 5]
 [3 6]]
```

---

## 9. 聚合计算 API 和 `axis`

`axis` 是 NumPy 和 Tensor 都非常重要的概念。

```python
import numpy as np

x = np.array([[1, 2, 3],
              [4, 5, 6]])

print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))
```

输出：

```python
21
[5 7 9]
[ 6 15]
```

解释：

- `np.sum(x)`：全部加起来
- `axis=0`：每一列往下加
- `axis=1`：每一行往右加

再看平均值：

```python
print(np.mean(x))
print(np.mean(x, axis=0))
print(np.mean(x, axis=1))
```

输出：

```python
3.5
[2.5 3.5 4.5]
[2. 5.]
```

你要养成习惯：

每次看到 `axis`，都先问自己一句：

“它是沿着哪一维算的？”

---

## 10. 常用统计 API

```python
import numpy as np

x = np.array([1, 2, 3, 4])

print(np.sum(x))
print(np.mean(x))
print(np.max(x))
print(np.min(x))
print(np.std(x))
```

输出：

```python
10
2.5
4
1
1.118033988749895
```

解释：

- `sum`：求和
- `mean`：平均值
- `max`：最大值
- `min`：最小值
- `std`：标准差

---

## 11. 常用数学 API

```python
import numpy as np

x = np.array([1, 4, 9])

print(np.sqrt(x))
print(np.exp(np.array([1, 2])))
print(np.log(np.array([1, np.e])))
```

输出：

```python
[1. 2. 3.]
[2.71828183 7.3890561 ]
[0. 1.]
```

这些函数也都是逐元素计算。

---

## 12. 矩阵乘法和逐元素乘法的区别

这是高频易错点。

```python
import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])

print(a * b)
print(a @ b)
```

输出：

```python
[[ 5 12]
 [21 32]]
[[19 22]
 [43 50]]
```

区别：

- `a * b`：逐元素乘法
- `a @ b`：矩阵乘法

后面神经网络里的线性层，本质上就是大量矩阵乘法。

---

## 13. 拼接 API

### 13.1 `np.concatenate`

作用：按某个轴拼接数组

```python
import numpy as np

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6]])

print(np.concatenate([a, b], axis=0))
```

输出：

```python
[[1 2]
 [3 4]
 [5 6]]
```

再看按列拼接：

```python
a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5],
              [6]])

print(np.concatenate([a, b], axis=1))
```

输出：

```python
[[1 2 5]
 [3 4 6]]
```

理解：

- `axis=0`：上下拼
- `axis=1`：左右拼

---

## 14. 条件筛选 API

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6])

print(x > 3)
print(x[x > 3])
```

输出：

```python
[False False False  True  True  True]
[4 5 6]
```

这类写法在后面图像处理、目标筛选、置信度过滤里也很常见。

---

## 15. 一张总表快速记忆

| API | 作用 | 例子 | 结果 |
|---|---|---|---|
| `np.array` | 列表转数组 | `np.array([1,2,3])` | `[1 2 3]` |
| `np.zeros` | 全 0 | `np.zeros((2,2))` | `[[0. 0.],[0. 0.]]` |
| `np.ones` | 全 1 | `np.ones((2,2))` | `[[1. 1.],[1. 1.]]` |
| `np.arange` | 按步长生成 | `np.arange(0,6,2)` | `[0 2 4]` |
| `np.linspace` | 按数量均匀生成 | `np.linspace(0,1,5)` | `[0. 0.25 0.5 0.75 1.]` |
| `shape` | 看形状 | `x.shape` | 如 `(2,3)` |
| `ndim` | 看维度数 | `x.ndim` | 如 `2` |
| `reshape` | 改形状 | `x.reshape(2,3)` | 变成 2 行 3 列 |
| `T` | 转置 | `x.T` | 行列交换 |
| `sum` | 求和 | `np.sum(x, axis=0)` | 每列求和 |
| `mean` | 平均值 | `np.mean(x)` | 所有元素平均 |
| `max` | 最大值 | `np.max(x)` | 最大元素 |
| `*` | 逐元素乘法 | `a * b` | 对位相乘 |
| `@` | 矩阵乘法 | `a @ b` | 矩阵相乘 |
| `concatenate` | 拼接数组 | `np.concatenate([...], axis=0)` | 上下或左右拼 |

---

## 16. 一组完整练习

把下面这段自己手敲一遍，不要只看。

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([[1, 2],
              [3, 4]])
c = np.array([10, 20])

print("a =", a)
print("b =")
print(b)

print("a.shape =", a.shape)
print("b.shape =", b.shape)

print("a + 10 =", a + 10)
print("a * 2 =", a * 2)

print("b[0, 1] =", b[0, 1])
print("b[:, 0] =", b[:, 0])

print("b.reshape(4) =", b.reshape(4))
print("b.T =")
print(b.T)

print("np.sum(b, axis=0) =", np.sum(b, axis=0))
print("np.sum(b, axis=1) =", np.sum(b, axis=1))

print("b + c =")
print(b + c)

print("b @ b =")
print(b @ b)
```

对应输出：

```python
a = [1 2 3 4]
b =
[[1 2]
 [3 4]]
a.shape = (4,)
b.shape = (2, 2)
a + 10 = [11 12 13 14]
a * 2 = [2 4 6 8]
b[0, 1] = 2
b[:, 0] = [1 3]
b.reshape(4) = [1 2 3 4]
b.T =
[[1 3]
 [2 4]]
np.sum(b, axis=0) = [4 6]
np.sum(b, axis=1) = [3 7]
b + c =
[[11 22]
 [13 24]]
b @ b =
[[ 7 10]
 [15 22]]
```

---

## 17. 学完 NumPy 后，你应该会什么

如果下面这些你都能说清楚，说明你已经可以很顺地进入 Tensor：

1. `ndarray` 是什么
2. `shape / ndim / dtype` 分别是什么意思
3. 怎么切片取行、取列、取子块
4. 什么是逐元素运算
5. 什么是广播
6. 什么是 `reshape`、什么是转置
7. `axis=0` 和 `axis=1` 到底在干什么
8. `*` 和 `@` 的区别

---

## 18. 下一步怎么接 Tensor

后面你学 Tensor 时，可以直接对照去理解：

- `np.array` 对应 `torch.tensor`
- `shape / ndim` 基本同逻辑
- 索引切片几乎一样
- `reshape / transpose / sum / mean` 基本一样
- 广播机制也几乎一样

真正多出来的，只是：

- Tensor 可以放到 GPU
- Tensor 可以自动求导

所以你现在学的，不是“额外内容”，而是在给后面的 Tensor 打地基。

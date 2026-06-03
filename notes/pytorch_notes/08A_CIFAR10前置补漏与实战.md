# 08A_CIFAR10前置补漏与实战

## 0. 这节解决什么问题
你现在最容易混的不是“模型结构”，而是这几个前置细节：

1. `transform` 到底做了什么，为什么不写会影响训练。
2. `shuffle=True/False` 到底影响什么。
3. 类别名和标签数字为什么必须一一对应。
4. 分类里 `CrossEntropyLoss`、`logits`、`argmax` 是怎么配合的。
5. `train` / `eval` / `no_grad` 各自干什么。

这节不是纯理论，下面每块都有可运行代码。

---

## 1. 一张图看数据流（分类任务）

```text
原始图片(PIL/ndarray)
-> transform(预处理)
-> Dataset.__getitem__(返回 image_tensor, label_int)
-> DataLoader(组batch, 控制shuffle)
-> model(images) 得到 logits
-> CrossEntropyLoss(logits, labels)
-> backward + optimizer.step
```

关键是：`label` 不是字符串，是整数索引（例如 0~9）。

---

## 2. transform：到底在改什么

### 2.1 `ToTensor()`
作用：

1. 把图片从 `H x W x C` 变为 `C x H x W`。
2. 把像素从 `[0, 255]` 变为浮点 `[0.0, 1.0]`（常见图像输入）。

### 2.2 `Normalize(mean, std)`
作用：按通道做标准化，公式是：

```text
output[channel] = (input[channel] - mean[channel]) / std[channel]
```

如果 `mean=std=0.5`，并且输入在 `[0,1]`，那么输出大致到 `[-1,1]`。

### 2.3 `Compose([...])`
作用：把多个 transform 串起来，按顺序执行。

---

## 3. 最小代码：看 transform 前后数值

```python
import torch
import torchvision
import torchvision.transforms as transforms

plain_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transforms.ToTensor()
)
img, label = plain_set[0]
print("ToTensor后:", img.shape, img.dtype, img.min().item(), img.max().item(), label)

norm_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
)
img2, label2 = norm_set[0]
print("Normalize后:", img2.shape, img2.dtype, img2.min().item(), img2.max().item(), label2)
```

你会看到：

1. `shape` 变成 `(3, 32, 32)`。
2. `ToTensor` 后范围大约在 `[0,1]`。
3. `Normalize` 后范围通常落在接近 `[-1,1]`。

---

## 4. shuffle：到底在打乱什么

`shuffle=True`：每个 epoch 开始时，样本索引顺序会重新洗牌。  
`shuffle=False`：按固定顺序取数据。

实战习惯：

1. 训练集：`shuffle=True`（减少模型“记顺序”）。
2. 验证/测试集：`shuffle=False`（结果稳定、可复现、便于排错）。

---

## 5. 最小代码：看 shuffle 的真实行为

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.arange(10).float().unsqueeze(1)  # [0..9]
y = torch.arange(10)
ds = TensorDataset(x, y)

loader_no = DataLoader(ds, batch_size=5, shuffle=False)
loader_yes = DataLoader(ds, batch_size=5, shuffle=True)

print("shuffle=False 第一轮:")
for _, labels in loader_no:
    print(labels.tolist())

print("shuffle=True 第一轮:")
for _, labels in loader_yes:
    print(labels.tolist())

print("shuffle=True 第二轮:")
for _, labels in loader_yes:
    print(labels.tolist())
```

你会看到 `shuffle=True` 的两轮顺序不一样。

---

## 6. 类别名和标签数字映射：必须一一对应

在 CIFAR10 中，标签是 `0~9`，类别顺序固定。  
你可以直接打印官方映射，别靠猜：

```python
import torchvision

ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=False)
print(ds.classes)
print(ds.class_to_idx)
```

典型输出：

```text
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
```

你自定义中文类别名可以，但顺序必须跟标签一致。例如：

```python
classes_zh = ("飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车")
```

然后显示预测时用：

```python
pred_idx = 3
print(classes_zh[pred_idx])  # 猫
```

---

## 7. CrossEntropyLoss、logits、argmax 的关系

分类训练的核心三句：

```python
outputs = model(images)            # shape: [B, 10]，这是 logits
loss = criterion(outputs, labels)  # criterion = nn.CrossEntropyLoss()
_, pred = torch.max(outputs, dim=1)
```

注意：

1. `outputs` 是 logits（未归一化分数），不是概率。
2. `CrossEntropyLoss` 直接吃 logits，不需要你手动再 `softmax`。
3. `torch.max(outputs, dim=1)` 取每行最大分数的类别索引，就是预测类别。

---

## 8. train / eval / no_grad：三者分工

### 8.1 训练阶段

```python
model.train()
for images, labels in trainloader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### 8.2 验证/测试阶段

```python
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        loss = criterion(outputs, labels)  # 可选：算验证loss
```

记忆：

1. `model.eval()`：切到评估模式（影响 Dropout/BN）。
2. `torch.no_grad()`：不建梯度图，省显存和算力。
3. 验证阶段不做 `backward()`，也不做 `optimizer.step()`。

---

## 9. 你现在最容易踩的 8 个坑

1. 训练和验证都开 `shuffle=True`，导致验证结果每次波动难排查。
2. 训练用了 `Normalize`，预测脚本忘了同样预处理。
3. 把标签当 one-hot 传给 `CrossEntropyLoss`（常见入门误区）。
4. 在 `CrossEntropyLoss` 前手动 `softmax`（通常不需要）。
5. 忘了 `model.eval()`，导致验证阶段 Dropout/BN 行为不稳定。
6. 忘了 `torch.no_grad()`，验证占用显存明显变大。
7. 自定义 `classes` 顺序和真实标签映射不一致。
8. 只看 `train loss`，不看 `val loss/accuracy`。

---

## 10. 下一步执行清单（直接做）

1. 先跑你现有的 `00_basics/08_cifar10_classifier.py`。
2. 确认能输出每轮 `loss` 和最终测试准确率。
3. 增加“每类准确率”打印（对齐官方 CIFAR10 教程）。
4. 再进入“单张图片 predict”流程（用同样的 transform）。

---

## 11. API速查（本节相关）

1. `torchvision.transforms.Compose`
2. `torchvision.transforms.ToTensor`
3. `torchvision.transforms.Normalize`
4. `torchvision.datasets.CIFAR10`
5. `torch.utils.data.DataLoader`
6. `torch.nn.CrossEntropyLoss`
7. `model.train()` / `model.eval()`
8. `torch.no_grad()`
9. `torch.max(outputs, dim=1)`


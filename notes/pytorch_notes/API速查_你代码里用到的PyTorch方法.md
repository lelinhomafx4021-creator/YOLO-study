# PyTorch API 速查 — 你代码里实际用到的

只列你真的写过或见过的，不列没用的。

---

## 1. 张量操作

| API | 干什么 | 你哪里用过 |
|-----|--------|-----------|
| `torch.rand(100,1)` | 生成 (100,1) 的随机数张量 | 07_training_loop.py |
| `tensor.to(device)` | 把张量搬到 GPU/CPU | 所有训练脚本 |
| `tensor.item()` | 把只有一个数的张量转成 Python 数字 | `loss.item()` |
| `tensor.size(0)` | 取第 0 维的长度（batch 大小） | `labels.size(0)` |
| `tensor.shape` | 返回张量的形状 | 调试用 |
| `torch.max(output, 1)` | 沿第 1 维（类别维）取最大值 | 分类预测 |
| `torch.argmax(output, 1)` | 同 max，但只返回位置（索引） | 等价 `torch.max()[1]` |
| `tensor.sum()` | 张量所有元素求和 | `correct += (pred==label).sum()` |
| `tensor.argmax()` | 数组里最大值的索引位置 | `best_epoch = mAP50.argmax()` |
| `torch.randn(...)` | 生成正态分布的随机数 | 初始化噪声 |

---

## 2. 模型层

| API | 干什么 | 你的代码 |
|------|--------|---------|
| `nn.Conv2d(in, out, kernel, padding)` | 2D 卷积层 | 09_truning_test1.py |
| `nn.Linear(in, out)` | 全连接层（矩阵乘） | 所有模型 |
| `nn.ReLU()` | 激活函数：负数变 0 | 所有模型 |
| `nn.MaxPool2d(k)` | k×k 池化：取最大值，缩小尺寸 | 09_truning_test1.py |
| `nn.Flatten()` | 把多维张量拉成一维 | 09_truning_test1.py |
| `nn.Sequential(...)` | 把多层按顺序串起来 | features / classifier |
| `nn.Module` | 所有模型必须继承的父类 | `class SimpleCNN(nn.Module)` |
| `nn.Dropout(p)` | 训练时随机扔 p% 神经元，防过拟合 | 笔记 11 示例 |
| `nn.BatchNorm2d(ch)` | 对每个通道做归一化，加速训练 | 笔记 11 示例 |
| `nn.MSELoss()` | 均方误差（回归任务用） | 07_training_loop.py |
| `nn.CrossEntropyLoss()` | 交叉熵损失（分类任务用） | 09_truning_test1.py |

---

## 3. 优化器

| API | 干什么 |
|------|--------|
| `torch.optim.SGD(params, lr, momentum, weight_decay)` | 带动量的 SGD |
| `torch.optim.Adam(params, lr, betas, eps, weight_decay)` | 自适应优化器 |
| `optimizer.zero_grad()` | 清零所有参数的梯度 |
| `optimizer.step()` | 用梯度更新参数 |
| `model.parameters()` | 返回模型所有可学习参数（喂给优化器） |

---

## 4. 训练/验证

| API | 干什么 |
|------|--------|
| `model.train()` | 切训练模式（Dropout 开启，BN 用 batch 统计） |
| `model.eval()` | 切评估模式（Dropout 关闭，BN 用全局统计） |
| `loss.backward()` | 反向传播：算出所有参数的梯度 |
| `torch.no_grad()` | 上下文管理器：不建计算图，省显存 |
| `model(x)` | 前向传播 = 调用 `model.forward(x)` |
| `model.to(device)` | 把模型所有参数搬到 GPU/CPU |

---

## 5. 模型保存与加载

| API | 干什么 |
|------|--------|
| `torch.save(model.state_dict(), "model.pth")` | 只保存参数（推荐） |
| `model.load_state_dict(torch.load("model.pth"))` | 加载参数 |
| `torch.save(model, "model_full.pth")` | 保存整个模型（含结构，不推荐） |
| `model = torch.load("model_full.pth")` | 加载整个模型 |

```python
# 标准保存
torch.save(model.state_dict(), "my_cnn.pth")

# 标准加载
model = SimpleCNN()
model.load_state_dict(torch.load("my_cnn.pth"))
model.eval()
```

---

## 6. 设备相关

| API | 干什么 |
|------|--------|
| `torch.cuda.is_available()` | 检测有没有 GPU |
| `torch.device("cuda" if ... else "cpu")` | 创建 device 对象 |
| `torch.cuda.get_device_name(0)` | 获取 GPU 型号名 |

---

## 7. 数据处理

| API | 干什么 |
|------|--------|
| `DataLoader(dataset, batch_size, shuffle, num_workers, pin_memory)` | 包装数据集，自动分批 |
| `torchvision.datasets.CIFAR10(root, train, transform, download)` | CIFAR-10 数据集 |
| `transforms.Compose([...])` | 把多个 transform 串起来 |
| `transforms.ToTensor()` | PIL 图像/NumPy → Tensor（像素值 0-1） |
| `transforms.Normalize(mean, std)` | 标准化：(x-mean)/std |

---

## 8. 你 09_truning_test1.py 里用到的完整模式

```python
# 造数据
x = torch.rand(100, 1)

# 建模型
model = SimpleCNN().to(device)

# 数据加载
trainloader = DataLoader(train_set, batch_size=64, shuffle=True)

# 损失函数 + 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
model.train()
for images, labels in trainloader:
    images, labels = images.to(device), labels.to(device)
    output = model(images)
    loss = criterion(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 验证
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

# 保存
torch.save(model.state_dict(), "my_cnn.pth")
```

---

## 还没用到但值得知道的

| API | 干什么 | 什么时候用 |
|------|--------|-----------|
| `torch.optim.lr_scheduler` | 学习率衰减 | 训到后期，lr 自动变小调细 |
| `torch.save({'epoch':e, 'model':m.state_dict(), 'opt':o.state_dict()}, "ckpt.pth")` | 断点续训 | 训一半停了下次接着训 |
| `tensor.view(shape)` | 重新排列张量形状 | 改 batch/通道/空间排列 |
| `nn.AdaptiveAvgPool2d(1)` | 全局平均池化 | 替代 Flatten，YOLO 常用 |
| `torch.stack([a,b])` | 把多个张量沿新维度叠起来 | 拼接 batch |
| `tensor.cpu().numpy()` | Tensor → NumPy 数组 | 画图、传给非 PyTorch 库 |

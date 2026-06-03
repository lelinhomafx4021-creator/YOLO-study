# -*- coding: utf-8 -*-
"""
09_tuning_experiments.py - 调参实验
在同一个 CNN 模型上改不同超参数，对比训练效果
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ======================================================================
# 设备检测
# ======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 注意：print 不放在这里！Windows 上 num_workers>0 时子进程会重新 import，
# 这句 print 会被每个 worker 执行一次，刷屏且拖慢启动。

# ======================================================================
# 公共部分：数据加载 + 模型定义（和 08_mytestforcnn.py 一样）
# ======================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_and_eval(batch_size, lr, epochs, optimizer_name="adam", weight_decay=0.0):
    """训练一个模型并返回测试准确率"""
    # num_workers=0: 主进程自己读数据
    #   CIFAR-10 是 32×32 小图，磁盘读取本来就不是瓶颈
    #   Windows 上 num_workers>0 用 spawn 启动子进程 → import torch 等 2-3 秒开销
    #   对于小数据集，开 worker 的启动开销 > 并行收益，反而更慢
    #   什么时候用 >0: ImageNet 级别的大图（≥224×224）、大量数据增强
    # pin_memory=True: 锁页内存，GPU DMA 直接拷，快 2-3 倍（小图也有收益）
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # ---- 选优化器 ----
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # ---- 训练 ----
    for epoch in range(epochs):
        model.train()
        for images, label in trainloader:
            images, label = images.to(device), label.to(device)
            output = model(images)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ---- 测试 ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, label in testloader:
            images, label = images.to(device), label.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    print(f"使用设备：{device}")
    if device.type == "cuda":
        print(f"GPU 型号：{torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True  # cuDNN 自动找最快的卷积算法

    # ======================================================================
    # 实验 1：不同学习率对比
    # ======================================================================
    print("=" * 55)
    print("实验1：不同学习率对比（batch=64, epochs=5, Adam）")
    print("-" * 55)
    for lr in [0.01, 0.001, 0.0001]:
        acc = train_and_eval(batch_size=64, lr=lr, epochs=5)
        print(f"lr={lr:<8} → 测试准确率 {acc:.1f}%")
    print("结论：lr=0.001 通常是最稳的起点，太大不稳定太小太慢\n")

    # ======================================================================
    # 实验 2：不同 batch_size 对比
    # ======================================================================
    print("=" * 55)
    print("实验2：不同 batch_size 对比（lr=0.001, epochs=5, Adam）")
    print("-" * 55)
    for bs in [32, 64, 128]:
        acc = train_and_eval(batch_size=bs, lr=0.001, epochs=5)
        print(f"batch_size={bs:<5} → 测试准确率 {acc:.1f}%")
    print("结论：batch_size 适中最好，太小噪声大，太大泛化差\n")

    # ======================================================================
    # 实验 3：Adam vs SGD
    # ======================================================================
    print("=" * 55)
    print("实验3：Adam vs SGD（batch=64, lr=0.001, epochs=5）")
    print("-" * 55)
    for opt in ["adam", "sgd"]:
        acc = train_and_eval(batch_size=64, lr=0.001, epochs=5, optimizer_name=opt)
        print(f"optimizer={opt:<5} → 测试准确率 {acc:.1f}%")
    print("结论：Adam 收敛快适合入门，SGD 需要更大 lr（试试0.01）和更多 epochs\n")

    # ======================================================================
    # 实验 4：weight_decay 防过拟合
    # ======================================================================
    print("=" * 55)
    print("实验4：weight_decay 对比（batch=64, lr=0.001, epochs=10, Adam）")
    print("-" * 55)
    for wd in [0.0, 1e-4, 1e-3]:
        acc = train_and_eval(batch_size=64, lr=0.001, epochs=10, weight_decay=wd)
        print(f"weight_decay={wd:<6} → 测试准确率 {acc:.1f}%")
    print("结论：weight_decay 不是越大越好，适度（1e-4）可能提升泛化\n")

    # ======================================================================
    # 实验 5：epochs 的影响
    # ======================================================================
    print("=" * 55)
    print("实验5：不同 epochs 对比（batch=64, lr=0.001, Adam）")
    print("-" * 55)
    for ep in [3, 5, 10]:
        acc = train_and_eval(batch_size=64, lr=0.001, epochs=ep)
        print(f"epochs={ep:<5} → 测试准确率 {acc:.1f}%")
    print("结论：epochs 增加到一定程度后收益递减，找到'够用'的点\n")

    print("=" * 55)
    print("调参核心原则：一次只改一个参数，否则不知道是哪个起的作用")

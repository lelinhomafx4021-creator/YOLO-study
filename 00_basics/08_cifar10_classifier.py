# -*- coding: utf-8 -*-
"""
08_cifar10_classifier.py - 用 PyTorch 从零写一个 CNN 图片分类器
数据集：CIFAR-10（10类：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）
目标：训练一个简单 CNN，准确率达到 60% 以上
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ======================================================================
# 第一步：图片预处理
# 原始图片像素是 0~255 的整数，模型需要 -1~1 的小数
# ======================================================================
transform = transforms.Compose([
    transforms.ToTensor(),                                    # 0~255 → 0~1（除以255）
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 0~1 → -1~1（公式：(值-0.5)/0.5）
])

# ======================================================================
# 第二步：下载数据 + 创建 DataLoader
# CIFAR-10 是 PyTorch 自带的教学数据集，download=True 自动下载
# ======================================================================
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader = 把数据分成一批一批的，每次给模型喂 64 张图
# shuffle=True = 每轮训练打乱顺序，防止模型记住顺序
# shuffle=False = 测试时不打乱，保证结果可复现
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

# 10 个类别名（和标签数字对应：0=plane, 1=car, 2=bird, ...）
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ======================================================================
# 第三步：定义 CNN 模型
# 维度变化：
#   输入 (3, 32, 32) → Conv1 → (16, 32, 32) → Pool → (16, 16, 16)
#                    → Conv2 → (32, 16, 16) → Pool → (32, 8, 8)
#                    → Flatten → (2048,)
#                    → Linear → (128,) → Linear → (10,)
# ======================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 前半段：从图片里提取特征（用卷积核扫描）
        self.features = nn.Sequential(
            # 第一层卷积：
            #   输入 3 个通道（RGB 三色）
            #   用 16 个不同的 3×3 卷积核扫描，提取 16 种特征
            #   padding=1 保证输出图片大小不变（还是 32×32）
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (3,32,32) → (16,32,32)
            nn.ReLU(),                                     # 负数变0，正数不变

            # 池化：每 2×2 区域取最大值，图片缩小一半
            nn.MaxPool2d(2),                               # (16,32,32) → (16,16,16)

            # 第二层卷积：
            #   输入 16 个通道（上一层的输出）
            #   用 32 个不同的 3×3 卷积核扫描，提取 32 种更高级的特征
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (16,16,16) → (32,16,16)
            nn.ReLU(),

            # 再池化一次，图片再缩小一半
            nn.MaxPool2d(2),                               # (32,16,16) → (32,8,8)
        )

        # 后半段：根据特征做分类判断
        self.classifier = nn.Sequential(
            # 展平：把 (32,8,8) 的立体特征图拉成一维
            # 32 × 8 × 8 = 2048 个数字
            nn.Flatten(),                                  # (32,8,8) → (2048,)

            # 全连接层：2048 个输入 → 128 个输出
            # 每个输出都看着全部 2048 个特征做综合判断
            nn.Linear(32 * 8 * 8, 128),                   # (2048,) → (128,)
            nn.ReLU(),

            # 最后一层：128 个特征 → 10 个类别的分数
            # 输出的 10 个数代表"这张图是各类别的可能性"
            nn.Linear(128, 10),                            # (128,) → (10,)
        )

    def forward(self, x):
        x = self.features(x)      # 先用 CNN 提取特征
        x = self.classifier(x)    # 再用 Linear 做分类
        return x


# ======================================================================
# 第四步：训练（和任务 1 完全一样的 5 步循环）
# ======================================================================
model = SimpleCNN()

# CrossEntropyLoss = 多分类专用 loss 函数
# 和 MSELoss 的区别：
#   MSELoss：预测值和真实值的差的平方（适合回归，比如预测房价）
#   CrossEntropyLoss：预测类别和真实类别的差距（适合分类，比如猫/狗/鸟）
criterion = nn.CrossEntropyLoss()

# Adam 优化器（比 SGD 更智能）
# SGD：每次改参数的步长固定
# Adam：根据历史自动调整步长，收敛更快
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("开始训练...")
for epoch in range(10):  # 训练 10 轮
    total_loss = 0
    for images, labels in trainloader:
        # ---- 训练循环 5 步（和任务 1 一模一样）----

        # 第 1 步：前向传播（模型做预测）
        # images 形状：(64, 3, 32, 32) → 64 张 32×32 彩色图
        # outputs 形状：(64, 10) → 每张图输出 10 个类别的分数
        outputs = model(images)

        # 第 2 步：算 loss（预测和真实答案差多少）
        # labels 形状：(64,) → 每张图的真实类别（0~9 的数字）
        loss = criterion(outputs, labels)

        # 第 3 步：清空旧梯度
        optimizer.zero_grad()

        # 第 4 步：反向传播（算出每个参数的梯度）
        loss.backward()

        # 第 5 步：更新参数
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(trainloader)
    print(f"Epoch {epoch+1}/10 | Loss: {avg_loss:.4f}")


# ======================================================================
# 第五步：在测试集上验证准确率
# ======================================================================
correct = 0
total = 0

model.eval()  # 切换到"评估模式"（关闭 Dropout 等训练特有的行为）
with torch.no_grad():  # 推理时不需要算梯度，关掉可以加速
    for images, labels in testloader:
        outputs = model(images)              # (64, 10) → 每张图的 10 个类别分数
        _, predicted = torch.max(outputs, 1) # 取分数最高的那个类别作为预测结果
        # predicted 形状：(64,) → 每张图预测的类别（0~9）
        total += labels.size(0)              # 累计总图片数
        correct += (predicted == labels).sum().item()  # 累计猜对的数量

print(f"\n测试集准确率: {100 * correct / total:.1f}%")
print(f"10000 张测试图里猜对了 {correct} 张")

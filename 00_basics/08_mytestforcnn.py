# -*- coding: utf-8 -*-
"""
08_mytestforcnn.py - 自己写的 CNN 图片分类器（带详细参数说明）
数据集：CIFAR-10（10类：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ======================================================================
# 第一步：图片预处理
# ======================================================================
# transforms.Compose([...]) = 把多个处理步骤串成流水线，按顺序执行
transform = transforms.Compose([
    transforms.ToTensor(),                                    # 把图片从 [0,255] 整数变成 [0,1] 小数
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 把 [0,1] 变成 [-1,1]
    # Normalize 参数：(均值R,均值G,均值B), (标准差R,标准差G,标准差B)
    # 公式：(值 - 均值) / 标准差 → (0.5-0.5)/0.5 = 0, (1.0-0.5)/0.5 = 1.0, (0-0.5)/0.5 = -1.0
])

# ======================================================================
# 第二步：下载数据 + 创建 DataLoader
# ======================================================================
# torchvision.datasets.CIFAR10 = PyTorch 自带的 CIFAR-10 数据集类
#   root：数据下载到哪个目录
#   train=True：加载训练集（50000张），train=False：加载测试集（10000张）
#   download=True：如果本地没有就自动下载
#   transform=transform：每次取数据时自动执行上面的预处理流水线
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# DataLoader = 把数据分成一批一批给模型
#   batch_size=64：每次喂 64 张图
#   shuffle=True：每轮打乱顺序（训练用），shuffle=False：不打乱（测试用）
#   num_workers=0：用几个进程加载数据（0=主进程，Windows 下建议 0）
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

# 10 个类别名
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ======================================================================
# 第三步：定义 CNN 模型
# ======================================================================
# nn.Module = 所有 PyTorch 模型的基类，必须继承
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的初始化，必须写

        # nn.Sequential = 按顺序执行的容器，数据从第一个层流到最后一个层
        self.features = nn.Sequential(
            # nn.Conv2d = 卷积层，用卷积核在图片上滑动提取特征
            #   参数1：in_channels=3，输入通道数（彩色图片=3，灰度=1）
            #   参数2：out_channels=16，输出通道数（用几个卷积核=几种特征）
            #   参数3：kernel_size=3，卷积核大小 3×3
            #   参数4：padding=1，边缘补一圈0，保证输出大小不变
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (3,32,32) → (16,32,32)

            # nn.ReLU = 激活函数，负数变0，正数不变
            #   没有参数，不需要填
            nn.ReLU(),

            # nn.MaxPool2d = 池化层，每 2×2 区域取最大值，图片缩小一半
            #   参数：kernel_size=2，池化窗口大小 2×2
            nn.MaxPool2d(2),                               # (16,32,32) → (16,16,16)

            # 第二层卷积：输入16通道（上一层输出），输出32通道
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (16,16,16) → (32,16,16)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (32,16,16) → (32,8,8)
        )

        self.classifier = nn.Sequential(
            # nn.Flatten = 展平层，把多维张量拉成一维
            #   没有参数
            #   (32,8,8) → (32×8×8,) = (2048,)
            nn.Flatten(),

            # nn.Linear = 全连接层，每个输入和每个输出都连一条线
            #   参数1：in_features=2048，输入个数（通道×高×宽 = 32×8×8）
            #   参数2：out_features=128，输出个数（自己选）
            #   内部参数：weight(128,2048) + bias(128) = 262,272 个可训练参数
            nn.Linear(32 * 8 * 8, 128),                   # (2048,) → (128,)
            nn.ReLU(),

            # 最后一层：128 → 10（10个类别）
            nn.Linear(128, 10),                            # (128,) → (10,)
        )

    def forward(self, x):
        # forward = 定义数据怎么流过模型
        #   x：输入的图片 Tensor，形状 (batch, 3, 32, 32)
        x = self.features(x)      # 先用 CNN 提特征
        x = self.classifier(x)    # 再用 Linear 做分类
        return x                  # 返回 10 个类别的分数


# ======================================================================
# 第四步：训练
# ======================================================================
model = SimpleCNN()

# nn.CrossEntropyLoss = 多分类专用 loss 函数
#   内部自动做了 SoftMax（把分数转成概率） + NLLLoss（算交叉熵）
#   输入：模型输出的原始分数 (batch, 10)
#   输入：真实标签 (batch,)，每个是 0~9 的整数
#   输出：一个标量 loss 值
criterion = nn.CrossEntropyLoss()

# torch.optim.Adam = Adam 优化器（比 SGD 更智能）
#   参数1：model.parameters()，告诉它去优化模型里的哪些参数
#   参数2：lr=0.001，学习率，每次更新参数的步长
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch_num = 10  # 训练几轮
print("开始训练...")

for epoch in range(epoch_num):
    loss_total = 0.0  # 每轮重新累加

    for images, label in trainloader:
        # images 形状：(64, 3, 32, 32) 一批64张图
        # label 形状：(64,) 每张图的真实类别（0~9）

        output = model(images)              # 前向传播，output 形状：(64, 10)
        loss = criterion(output, label)     # 算 loss

        optimizer.zero_grad()               # 清空旧梯度
        loss.backward()                     # 反向传播，算梯度
        optimizer.step()                    # 更新参数

        loss_total += loss.item()           # .item() 把 Tensor 变成普通数字

    print(f"Epoch {epoch+1}/{epoch_num} | Loss: {loss_total / len(trainloader):.4f}")

# ======================================================================
# 第五步：测试
# ======================================================================
model.eval()  # 切换到评估模式
correct = 0
total = 0

# 测试只跑一轮，不需要循环 epoch
for images, label in testloader:
    outputs = model(images)                # (64, 10) 每张图的 10 个类别分数
    _, predicted = torch.max(outputs, 1)   # 取分数最高的位置作为预测类别
    # _ = 最大值（丢掉），predicted = 最大值的位置（0~9）
    total += label.size(0)                 # label.size(0) = 这一批有多少张图 = 64
    correct += (predicted == label).sum().item()  # 预测和真实答案一致的数量

print(f"\n测试集准确率: {100 * correct / total:.1f}%")
print(f"10000 张图里猜对了 {correct} 张")

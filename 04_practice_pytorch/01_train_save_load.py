# -*- coding: utf-8 -*-
"""
关卡 1 PyTorch版: CNN 训练 → 保存 → 加载 → 推理

═══════════════════════════════════════════════════════════════════════════════
这个脚本做的事情（一句话）
═══════════════════════════════════════════════════════════════════════════════

  下载 CIFAR10 图片数据集 → 搭一个 2 层 CNN → 训练 3 轮 → 保存权重 → 加载 → 推理

  对比 YOLO: model.train() 一行的事，这里要亲手写每条语句。
  目的: 知道 YOLO 封装的 model.train() 内部到底在干嘛。

═══════════════════════════════════════════════════════════════════════════════
文件里每条语句的参数说明
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn                          # nn: 神经网络模块 (Conv2d, Linear, ReLU...)
import torchvision                              # 视觉数据集 (CIFAR10, MNIST...) + 预训练模型
import torchvision.transforms as transforms     # 图像预处理 (转tensor, 归一化, 裁剪, 翻转...)

# ═════════════════════════════════════════════════════════════════════════════
# 0. 设备选择
# ═════════════════════════════════════════════════════════════════════════════
# torch.cuda.is_available() → True=有NVIDIA显卡  False=没有
# 有显卡就用 "cuda" (RTX 3050)，没有就用 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. 数据准备
# ═════════════════════════════════════════════════════════════════════════════

# ── transforms.Compose: 把多个预处理步骤串起来 ──
#   参数: 一个列表，里面的 transform 按顺序执行
#   例: [ToTensor, Normalize] → 先转tensor → 再归一化
transform = transforms.Compose([
    # ── ToTensor(): 把 PIL Image 或 numpy array 转成 torch tensor ──
    #   转换前: 图片是 PIL Image，像素值 0~255 (整数)，形状 H×W×C
    #   转换后: torch.Tensor，像素值 0.0~1.0 (浮点数)，形状 C×H×W
    #   注意: 通道顺序变了! HWC → CHW。PIL 是 H×W×C，PyTorch 要 C×H×W
    transforms.ToTensor(),

    # ── Normalize(mean, std): 把像素值标准化到 [-1, 1] ──
    #   参数:
    #     mean = (0.5,)*3 = (0.5, 0.5, 0.5)   ← R, G, B 三通道各自的均值
    #     std  = (0.5,)*3 = (0.5, 0.5, 0.5)   ← R, G, B 三通道各自的标准差
    #   公式: output = (input - mean) / std
    #   效果: input 范围 [0, 1] → output 范围 [-1, 1]
    #   为什么做归一化? 输入值范围统一 → 梯度更稳定 → 训练更快收敛
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
#   (0.5,)*3 是 Python 语法: (0.5,) * 3 = (0.5, 0.5, 0.5)
#   对 RGB 三个通道分别用同样的均值和标准差

# ── CIFAR10: 10 类彩色小图数据集 ──
#   参数:
#     root = "./data"    → 下载/读取数据存放的文件夹
#     train = True/False → True=训练集(50000张) False=测试集(10000张)
#     transform = ...    → 每次取图片时自动做的预处理 (上面定义的 Compose)
#     download = True    → 文件夹里没有就自动下载 (~170MB)
#   数据集内容: 32×32 的彩色图片，共 10 类:
#     ['飞机','汽车','鸟','猫','鹿','狗','青蛙','马','船','卡车']
train_set = torchvision.datasets.CIFAR10("./data", train=True,
                                          transform=transform, download=True)
test_set  = torchvision.datasets.CIFAR10("./data", train=False,
                                          transform=transform, download=True)

# ── DataLoader: 把数据集包装成"每次取一批"的迭代器 ──
#   参数:
#     dataset     → 数据集对象 (上面创建的 train_set / test_set)
#     batch_size  → 每批多少张图 (一次 forward 处理 batch_size 张)
#                   为什么用 batch? 1张1张训练太慢; 全量一起训显存放不下
#                   64 是小 batch，显存够用、梯度方向也比较稳定
#     shuffle     → 训练集: True  = 每个 epoch 随机打乱顺序
#                          为什么打乱? 防止模型记住"先看到鸟、后看到猫"的顺序
#                          不打乱可能导致模型按顺序作弊
#                   测试集: False = 不打乱，评估不需要
#   DataLoader 用法:
#     for imgs, labels in trainloader:
#         imgs:   torch.Tensor 形状 [batch_size, 3, 32, 32]  ← 一批图
#         labels: torch.Tensor 形状 [batch_size]             ← 一批标签(0~9)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_set,  batch_size=64, shuffle=False)
# 50000 张训练图 / 64 = 782 个 batch (最后一个 batch 只有 32 张)
# 10000 张测试图 / 64 = 157 个 batch


# ═════════════════════════════════════════════════════════════════════════════
# 2. 模型定义
# ═════════════════════════════════════════════════════════════════════════════

class TinyCNN(nn.Module):
    """
    一个极简 CNN，只有 2 层卷积 + 1 层全连接。

    数据流:
      输入 [64, 3, 32, 32]    ← 一批 32×32 彩色图
        ↓ Conv2d(3→8, 3×3)    → [64, 8, 32, 32]   (padding=1 保持尺寸不变)
        ↓ ReLU                 → 激活: 负数变0
        ↓ MaxPool2d(2)         → [64, 8, 16, 16]   (尺寸减半)
        ↓ Conv2d(8→16, 3×3)   → [64, 16, 16, 16]
        ↓ ReLU
        ↓ MaxPool2d(2)         → [64, 16, 8, 8]    (尺寸再减半)
        ↓ Flatten              → [64, 1024]         (16×8×8 拉成一维)
        ↓ Linear(1024→10)      → [64, 10]           (10 类每类的得分)
    """
    def __init__(self):
        super().__init__()  # 必须调用父类 nn.Module 的 __init__

        # ── nn.Sequential: 把多个层串成一条流水线 ──
        #   数据依次流过每一层，不用手动写中间变量
        self.net = nn.Sequential(
            # ── Conv2d: 2D 卷积层 (CNN 的核心) ──
            #   参数:
            #     in_channels  = 3    → 输入通道数 (RGB 彩色图 = 3 通道)
            #     out_channels = 8    → 输出通道数 (= 8 个卷积核, 每个学一种特征)
            #     kernel_size  = 3    → 卷积核大小 3×3 (在图上用 3×3 窗口滑动扫描)
            #     padding      = 1    → 在图片周围补一圈 0, 让输出尺寸和输入一样大
            #                           不加 padding: 32×32 → 30×30 (每次缩小 2 像素)
            #                           加 padding=1: 32×32 → 32×32 (尺寸不变)
            #   参数量: (3×3×3 + 1) × 8 = 224 个参数 (3×3窗口×3通道 + 1偏置 × 8个核)
            nn.Conv2d(3, 8, 3, padding=1),
            #   Conv2d 做的事: 一个小窗口在图上滑动, 每个位置做"像素×权重求和"
            #   第1层学习: 边缘、颜色、简单纹理

            # ── ReLU: 激活函数 ──
            #   公式: f(x) = max(0, x)  → 负数变 0, 正数不变
            #   为什么需要? 没有 ReLU, 多层 Conv 叠在一起等价于一层 (全是线性变换)
            #               加了 ReLU → 引入非线性 → 能学到更复杂的特征
            #   没有参数, 不改变形状
            nn.ReLU(),

            # ── MaxPool2d: 最大池化层 (下采样, 缩小尺寸) ──
            #   参数: kernel_size=2 → 用 2×2 窗口, 取窗口里最大的值
            #   效果: 长宽各减半  → 32×32 → 16×16
            #   为什么用池化?
            #     1. 减少计算量 (尺寸越小, 后面层计算越快)
            #     2. 防止过拟合 (压缩信息, 强制模型学最重要的特征)
            #     3. 提供平移不变性 (特征稍微挪一点, 最大值还在窗口里)
            nn.MaxPool2d(2),           # 32×32 → 16×16

            # 第2层卷积: 输出通道翻倍 (8→16), 学更抽象的特征
            nn.Conv2d(8, 16, 3, padding=1),
            #   第2层学到: 纹理组合、局部形状 (眼睛、轮子等)
            nn.ReLU(),
            nn.MaxPool2d(2),           # 16×16 → 8×8

            # ── Flatten: 把多维 tensor 拉成一维向量 ──
            #   输入: [batch, 16, 8, 8]   ← 4 维
            #   输出: [batch, 16×8×8] = [batch, 1024]  ← 2 维
            #   为什么? 全连接层只能接收一维向量, 不能接收多维特征图
            nn.Flatten(),

            # ── Linear: 全连接层 (分类头) ──
            #   公式: y = W·x + b   (矩阵乘法 + 偏置)
            #   参数:
            #     in_features  = 16*8*8 = 1024  → 输入: 卷积提取的 1024 个特征
            #     out_features = 10             → 输出: CIFAR10 的 10 个类别
            #   参数量: 1024×10 + 10 = 10250 个参数 (占了整个模型 95% 的参数)
            #   输出 [batch, 10] 是 10 个原始分数 (logits), 不是概率
            #   后面接 CrossEntropyLoss 会自动做 softmax 转概率
            nn.Linear(16*8*8, 10),
        )

    def forward(self, x):
        """前向传播: 输入 x → 经过 net → 输出 10 类得分"""
        return self.net(x)


model = TinyCNN().to(device)  # .to(device) → 把模型参数搬到 GPU (如果有的话)


# ═════════════════════════════════════════════════════════════════════════════
# 3. 训练前: 记录参数初始值
# ═════════════════════════════════════════════════════════════════════════════
# model.net[0] = Sequential 的第 0 层 = 第1个 Conv2d
# .weight.data  = 卷积核的权重 (不包括偏置)
# .clone()      = 复制一份, 否则后面训练会跟着一起变 (tensor 是引用)
before_weight = model.net[0].weight.data.clone()
# before_weight 形状: [8, 3, 3, 3]  ← [输出通道8, 输入通道3, 高3, 宽3]
# before_weight[0,0,0,0]: 第0个卷积核 第0个输入通道 第0行第0列 的权重值
print(f"训练前 Conv1 weight[0,0,0,0] = {before_weight[0,0,0,0]:.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. 训练 3 轮
# ═════════════════════════════════════════════════════════════════════════════

# ── CrossEntropyLoss: 交叉熵损失 ──
#   做的事: 比较"模型预测的 10 类得分"和"真实标签"，输出一个"错得多离谱"的数字
#   内部自动包含 softmax (把得分转成概率) + 负对数似然
#   所以模型最后一层不需要 softmax，直接出原始分数 (logits) 就行
#   例: 真实标签=3(猫), 模型对"猫"的得分很高 → loss 小
#       真实标签=3(猫), 模型对"狗"的得分很高 → loss 大 → 梯度推动权重修正
criterion = nn.CrossEntropyLoss()

# ── Adam 优化器: 用 loss 的梯度来更新模型参数 ──
#   参数:
#     model.parameters() → 模型中所有可训练的权重和偏置 (Conv 的核 + Linear 的 W,b)
#     lr = 0.001         → 学习率 (learning rate): 每次更新参数的步长
#                          太大(0.1): 训练不稳定, loss 来回跳, 可能永远不收敛
#                          太小(1e-5): 收敛太慢, 3 个 epoch 几乎没变化
#                          0.001 是 Adam 的常用默认值, 对大多数任务合适
#   Adam 比 SGD 好在哪? 自适应调整每个参数的学习率 → 不用手动调, 收敛快
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):                    # 训练 3 轮 (epoch = 完整遍历一次训练集)
    model.train()                         # 切换到训练模式
    #                                       影响: Dropout 打开, BatchNorm 用当前 batch 的统计量
    #                                       这个模型没有 Dropout 和 BN, 但养成习惯先写

    for imgs, labels in trainloader:      # 每批取 batch_size=64 张图
        imgs, labels = imgs.to(device), labels.to(device)  # 搬到 GPU

        # ── 一轮训练 = 5 步 ──
        # 第1步: 清空梯度
        #        PyTorch 默认梯度会累加 (`+=`), 不清零就变成"当前梯度 + 之前梯度"
        #        每轮开始前必须归零
        optimizer.zero_grad()

        # 第2步: 前向传播
        #        imgs [64,3,32,32] → Conv → ReLU → Pool → Conv → ReLU → Pool → Flat → Linear
        #        → outputs [64, 10]  (64张图每张10个类别的得分)
        loss = criterion(model(imgs), labels)

        # 第3步: 反向传播 → 计算梯度
        #        链式法则: 从 loss 往回算, 算出每个参数对 loss 的偏导数
        #        Conv1 的某个权重 w: d(loss)/dw = ?  → 算完存在 w.grad 里
        loss.backward()

        # 第4步: 更新参数
        #        用 Adam 公式, 根据梯度更新所有权重
        #        w_new = w_old - lr × 修正后的梯度
        optimizer.step()

    # ── 验证 (每个 epoch 结束后跑一次) ──
    model.eval()                          # 切换到评估模式
    #                                       影响: Dropout 关闭, BatchNorm 用全局统计量
    correct = 0                           # 累计预测正确的图片数

    with torch.no_grad():                 # 关闭梯度计算 (不训练, 不需要梯度)
        #                                      好处: 省显存 + 加速
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)

            # model(imgs) → [64, 10] 每类的得分
            # .argmax(1)  → [64] 得分最高的那个类的索引 (0~9)
            #   argmax(dim=1) = 沿着第 1 维 (类别维) 找最大值
            #   例: 输出 [0.1, 0.2, 8.3, ...] → argmax 返回 2
            # == labels    → 逐元素比较, 返回 bool tensor [True, False, True, ...]
            # .sum()       → 统计 True 的数量 → 这一批预测对了几张
            # .item()      → 把单元素 tensor 转成 Python 数字
            correct += (model(imgs).argmax(1) == labels).sum().item()

    # 10000 是测试集总图片数, correct/10000 = 准确率
    print(f"Epoch {epoch+1}: acc={100*correct/10000:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# 5. 训练后: 参数值变了!
# ═════════════════════════════════════════════════════════════════════════════
after_weight = model.net[0].weight.data  # 不需要 clone, 只是读一下
print(f"\n训练后 Conv1 weight[0,0,0,0] = {after_weight[0,0,0,0]:.6f}")

# torch.equal: 两个 tensor 每个元素都完全相等 → True; 否则 False
print(f"参数变了: {not torch.equal(before_weight, after_weight)}")  # True → 训练有效


# ═════════════════════════════════════════════════════════════════════════════
# 6. 保存 + 加载
# ═════════════════════════════════════════════════════════════════════════════

# ── state_dict: 模型的"参数清单" ──
#   是一个 Python dict: {"net.0.weight": tensor(...), "net.0.bias": tensor(...), ...}
#   包含所有可学习参数的当前值，不包含模型结构
#   state_dict 就是模型最核心的东西 —— 拿到了参数，就能复现模型
torch.save(model.state_dict(), "04_practice_pytorch/cnn_test.pth")
print("\n已保存: 04_practice_pytorch/cnn_test.pth")
# .pth 或 .pt 都是 PyTorch 约定的后缀，本质就是 pickle 序列化文件

# ── 加载: 创建一个"空壳"模型，把保存的参数填进去 ──
new_model = TinyCNN().to(device)          # 新建一个模型 (参数是随机初始化的)

# torch.load: 从 .pth 文件读取 dict
#   map_location=device → 如果保存时在 GPU, 加载时想放 CPU, 这个参数负责搬迁
#                         这里 device 可能是 cuda, 保持一致
# load_state_dict: 把 dict 里的参数值覆盖到模型里
new_model.load_state_dict(torch.load("04_practice_pytorch/cnn_test.pth",
                                     map_location=device))
new_model.eval()  # 推理模式


# ═════════════════════════════════════════════════════════════════════════════
# 7. 验证: 加载后的参数 == 训练后的参数
# ═════════════════════════════════════════════════════════════════════════════
loaded_weight = new_model.net[0].weight.data
print(f"加载后 == 训练后: {torch.equal(after_weight, loaded_weight)}")  # True → 保存成功


# ═════════════════════════════════════════════════════════════════════════════
# 8. 推理: 拿测试集前 5 张图预测
# ═════════════════════════════════════════════════════════════════════════════

# next(iter(testloader)): 从测试 DataLoader 取第一个 batch
#   iter(testloader)  → 创建迭代器
#   next(...)         → 取下一个 batch → (imgs [64,3,32,32], labels [64])
imgs, labels = next(iter(testloader))
imgs = imgs.to(device)

with torch.no_grad():                          # 推理不需要梯度
    preds = new_model(imgs).argmax(1)          # [64,10] → [64] 每张图的预测类别

classes = ["飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"]
for i in range(5):
    print(f"图{i+1}: 预测={classes[preds[i]]}  真值={classes[labels[i]]}  "
          f"{'✅' if preds[i]==labels[i] else '❌'}")

print("\n✅ 关卡 1 PyTorch版 完成")

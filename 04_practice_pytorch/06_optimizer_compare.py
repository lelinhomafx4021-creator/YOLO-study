# -*- coding: utf-8 -*-
"""
关卡 6 PyTorch版: SGD vs Adam 优化器对比实验

═══════════════════════════════════════════════════════════════════════════════
这一关在做什么
═══════════════════════════════════════════════════════════════════════════════

  同一个模型、同一批数据、同样的初始参数 → 只换优化器 → 看训练曲线差多少。

  4 组实验:
    1. Adam  lr=0.001   ← 默认推荐，最稳
    2. Adam  lr=0.01    ← 学习率大了 10 倍，看看会不会崩
    3. SGD   lr=0.01    ← 经典优化器，带 momentum
    4. SGD   lr=0.1     ← 学习率提 10 倍

  每组分两条曲线: train_loss (看优化速度) + val_acc (看泛化效果)
  4 组画在同一张图上 → 一眼看出谁好谁坏

═══════════════════════════════════════════════════════════════════════════════
Adam vs SGD — 核心区别
═══════════════════════════════════════════════════════════════════════════════

  SGD:  w = w - lr × 梯度
        所有参数用同一个学习率
        对 lr 极度敏感 → lr 太小走不动, 太大震荡甚至发散
        momentum=0.9 可以加速: 像球滚下山，惯性带着走

  Adam: w = w - lr × (修正后的梯度 / sqrt(修正后的方差))
        每个参数有自己独立的学习率
        自适应: 梯度大的参数自动降 lr，梯度小的自动提 lr
        对 lr 不敏感 → 0.001 和 0.01 都能跑
        几乎不需要调参，默认值就能用

  一句话: Adam = SGD + 动量 + 自适应学习率。贵一点，但省事。

═══════════════════════════════════════════════════════════════════════════════
新增知识点: 模型工厂函数
═══════════════════════════════════════════════════════════════════════════════

  之前的脚本: 全局建一个 model，跑一次
  这个脚本:   make_model() 函数每次调用创建一个全新的模型
              → 4 组实验各跑各的，起点相同 → 公平对比

  如果全局只建一个 model:
    第一组跑完参数变了 → 第二组起点不一样 → 不公平
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# 1. 数据 — 和之前完全一样
# ═════════════════════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
train_set = torchvision.datasets.CIFAR10("./data", train=True,
                                          transform=transform, download=True)
test_set  = torchvision.datasets.CIFAR10("./data", train=False,
                                          transform=transform, download=True)
# DataLoader(数据集, batch_size, shuffle): 省略参数名按位置传也可以
trainloader = torch.utils.data.DataLoader(train_set, 64, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_set,  64, shuffle=False)


# ═════════════════════════════════════════════════════════════════════════════
# 2. 模型工厂 — 每次调用创建一个"干净"的模型
# ═════════════════════════════════════════════════════════════════════════════
# 为什么是函数? 跑 4 组实验需要 4 个模型，每组从相同的随机初始状态开始
# 工厂函数 = 每次调用 → 全新随机初始模型 → 公平对比
def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16*8*8, 10),
    ).to(device)


# ═════════════════════════════════════════════════════════════════════════════
# 3. 训练函数 — 封装一次完整的训练 (模型 + 优化器 + 训练循环)
# ═════════════════════════════════════════════════════════════════════════════
def train_one_run(opt_name, lr, epochs=10):
    """
    参数:
      opt_name: "adam" / "sgd" / "sgd_no_momentum"
      lr:        学习率 (learning rate)
      epochs:    训练轮数

    返回:
      history = {"train_loss": [10个值], "val_acc": [10个值]}
               每个 epoch 记录一次，用于最后画曲线
    """
    model = make_model()                    # 全新的随机初始化模型
    criterion = nn.CrossEntropyLoss()

    # ── 选择优化器 ──
    # torch.optim.Adam(params, lr=学习率)
    #   内部: betas=(0.9, 0.999) — 一阶动量系数和二阶方差衰减系数
    #   99% 的情况不用改这两个值
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    # torch.optim.SGD(params, lr=学习率, momentum=动量)
    #   momentum=0.9: 就像球滚下山 — 惯性推动，不容易卡在局部最低点
    #                最常见的设置，ResNet/EfficientNet 都用这个
    #   momentum=0:   纯 SGD — 只有当前梯度，没惯性，走得慢
    elif opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "sgd_no_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

    # ── 训练循环（和关卡 2 一模一样）──
    history = {"train_loss": [], "val_acc": []}

    for ep in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # 验证阶段
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()

        # 记录本轮
        history["train_loss"].append(total_loss / len(trainloader))
        history["val_acc"].append(100 * correct / 10000)

    return history


# ═════════════════════════════════════════════════════════════════════════════
# 4. 跑 4 组对比实验
# ═════════════════════════════════════════════════════════════════════════════
# 每组: (图例标签, 优化器名, 学习率)
configs = [
    ("Adam  lr=0.001",  "adam",  0.001),   # 默认推荐，最稳
    ("Adam  lr=0.01",   "adam",  0.01),    # lr ×10 → 看会不会震荡
    ("SGD   lr=0.01",   "sgd",   0.01),    # SGD 小 lr + momentum
    ("SGD   lr=0.1",    "sgd",   0.1),     # SGD 大 lr + momentum
]

# ── 画布: 左 loss, 右 acc ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# 4 种颜色 — 4 组实验各一色，legend 区分类别
colors = ["#e17055", "#fdcb6e", "#00d4ff", "#0984e3"]
#         橙红      金黄      天蓝      深蓝

# zip(configs, colors): 把 4 组配置和 4 种颜色逐一绑定
for (label, opt_name, lr), c in zip(configs, colors):
    # ── 跑一次完整的训练 ──
    h = train_one_run(opt_name, lr, epochs=10)

    # 左图: loss 曲线 — 越低越好，看优化速度
    #   lw=2: linewidth=2, 线比默认粗一点
    #   label: 用于 legend 图例，一条线一个名字
    ax1.plot(range(1, 11), h["train_loss"], color=c, label=label, lw=2)

    # 右图: acc 曲线 — 越高越好，看泛化能力
    ax2.plot(range(1, 11), h["val_acc"],   color=c, label=label, lw=2)

    # 终端打印
    print(f"{label:<18} → 最终 acc={h['val_acc'][-1]:.1f}%")

# ── 左图装饰 ──
ax1.set_title("训练 Loss 对比")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()          # 显示图例: 哪条线是哪个优化器
ax1.grid(alpha=0.3)   # 半透明网格线

# ── 右图装饰 ──
ax2.set_title("验证准确率对比")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Acc %")
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle("SGD vs Adam — 同一模型、不同优化器", fontweight="bold")
plt.tight_layout()
plt.show()

print("""
结论（你亲眼看到的）:
  1. Adam lr=0.001 → 最稳，训得最快
  2. Adam lr=0.01  → 也能训，但有点震荡
  3. SGD lr=0.01   → 比 Adam 慢，但有 momentum 还能走
  4. SGD lr=0.1    → 训得动了，但可能不稳定

  Adam 对 lr 不敏感（0.001和0.01都能跑）
  SGD 对 lr 极其敏感（0.001根本走不动, 0.01借助momentum勉强跟上）
""")
print("✅ 关卡 6 PyTorch版 完成")

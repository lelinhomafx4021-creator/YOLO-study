# -*- coding: utf-8 -*-
"""
关卡 5 PyTorch版: CNN 逐层特征图可视化

═══════════════════════════════════════════════════════════════════════════════
这一关在做什么
═══════════════════════════════════════════════════════════════════════════════

  之前所有脚本: 喂图 → 拿分类结果 → 中间是黑盒
  这个脚本:     喂图 → 把 Conv1 / Conv2 的输出画出来 → 亲眼看到"CNN 学到了什么"

  核心发现:
    Conv1 (浅层): 特征图还能看出原图轮廓 → 学的是边缘、颜色、纹理
    Conv2 (深层): 特征图变抽象，不再像原图 → 学的是"眼睛形状"、"轮子形状"

  面试高频题: "CNN 浅层和深层分别学到什么特征？"
  跑一遍这个脚本，眼睛看到证据，比背十遍概念都有用。

═══════════════════════════════════════════════════════════════════════════════
为什么不用 Sequential
═══════════════════════════════════════════════════════════════════════════════

  之前: self.net = nn.Sequential(Conv1, ReLU, Pool, Conv2, ...)
        → 简洁，但中间层的输出拿不到，全封死在流水线里

  现在: 每一层单独命名 → forward 里手动走每步 → 中间输出可以存下来
        self.conv1 = nn.Conv2d(...)   ← 单独命名
        self.feat1 = self.conv1(x)    ← 存中间结果
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# 1. 定义模型 — 每层单独命名，方便取出中间输出
# ═════════════════════════════════════════════════════════════════════════════
class VisualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # ── 每一层都明确命名，不是塞进 Sequential ──
        # Conv2d: in→out 通道翻倍（3→16→32），让特征从简单到复杂
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)    # 3 通道→16 通道
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)                    # 32×32 → 16×16

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # 16 通道→32 通道
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)                    # 16×16 → 8×8

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        """手动走每层，关键是把 conv1 和 conv2 的输出存到 self.feat1 / self.feat2"""
        # ── Conv1 + ReLU → 存下中间结果 ──
        # self.feat1: [B, 16, 32, 32]
        #   16 张 32×32 的特征图，每张检测一种低级特征（边缘、颜色、纹理）
        #   存到 self 上 → 推理完可以从 model.feat1 取出来看
        self.feat1 = self.conv1(x)
        x = self.relu1(self.feat1)
        x = self.pool1(x)                # [B, 16, 16, 16] — 尺寸减半

        # ── Conv2 + ReLU → 也存下来 ──
        # self.feat2: [B, 32, 16, 16]
        #   32 张 16×16 的特征图，每张检测一种高级特征（形状、纹理组合）
        self.feat2 = self.conv2(x)
        x = self.relu2(self.feat2)
        x = self.pool2(x)                # [B, 32, 8, 8] — 再减半

        # ── 分类头（不存中间结果，没必要）──
        x = self.flatten(x)              # [B, 32×8×8] = [B, 2048]
        return self.fc(x)                # [B, 10]


# ═════════════════════════════════════════════════════════════════════════════
# 2. 随便找一张 CIFAR-10 图
# ═════════════════════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
test_set = torchvision.datasets.CIFAR10("./data", train=False,
                                         transform=transform, download=True)
img, label = test_set[0]    # 第一张测试图 — 随机，随便哪张都行
# img 形状: [3, 32, 32]    ← 单张图，没有 batch 维
# label: 0~9 的整数

classes = ["飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"]
print(f"真值: {classes[label]}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. 前向传播 + 提取特征
# ═════════════════════════════════════════════════════════════════════════════
model = VisualCNN().to(device)
model.eval()                         # 推理模式（不训练，只是看）

with torch.no_grad():                # 不计算梯度
    # img.unsqueeze(0): [3,32,32] → [1, 3, 32, 32]
    #   在第 0 维（最前面）加一个 batch 维
    #   .to(device): 搬到 GPU
    out = model(img.unsqueeze(0).to(device))
    # out: [1, 10] — 10 类得分，但这次我们不关心它
    # 我们关心的是 model.feat1 和 model.feat2（forward 里存的中间结果）


# ═════════════════════════════════════════════════════════════════════════════
# 4. 可视化特征图
# ═════════════════════════════════════════════════════════════════════════════

# ── .cpu().numpy(): GPU → CPU → numpy 数组 ──
# model.feat1: [1, 16, 32, 32] — batch=1, 16 张 32×32 特征图
# feat1[0]:    [16, 32, 32]     — 去掉 batch 维（只喂了一张图所以取第 0 位）
feat1 = model.feat1[0].cpu().numpy()   # [16, 32, 32] — 16 张浅层特征图
feat2 = model.feat2[0].cpu().numpy()   # [32, 16, 16] — 32 张深层特征图

# ── 创建画布: 2 行 1 列 ──
#   figsize=(宽, 高) 单位英寸
#   上排: Conv1 的 16 张特征图
#   下排: Conv2 的 32 张特征图
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f"真值: {classes[label]} — 浅层 vs 深层特征图",
             fontweight="bold", fontsize=14)

# ── 上排: 浅层特征图 (Conv1) ────────────────────────────────────────────
ax = axes[0]

# np.concatenate([16 张图], axis=1): 把 16 张 32×32 的图横着拼成一行
#   feat1[0]: [32, 32]     第 0 张特征图
#   feat1[1]: [32, 32]     第 1 张特征图
#   ...                    全部横着拼
#   feat1[15]: [32, 32]
#   结果: [32, 32×16] = [32, 512]  ← 一长条，每段 32 像素是一张特征图
all1 = np.concatenate([feat1[i] for i in range(16)], axis=1)

# imshow: 把二维数值矩阵当图片渲染
#   cmap="viridis": 颜色映射 — 数值高=亮黄, 数值低=深紫
#   效果: 亮的区域 = Conv1 这个通道对这个输入"反应强"的位置
#         暗的区域 = 没什么反应
ax.imshow(all1, cmap="viridis")
ax.set_title("Conv1 输出 — 16 张特征图，每张 32×32（浅层：边缘、颜色）", fontsize=11)
ax.axis("off")   # 不显示坐标轴刻度（看了没用，干扰视觉）

# ── 下排: 深层特征图 (Conv2) ────────────────────────────────────────────
ax = axes[1]
# 同样操作: 32 张 16×16 横着拼成 [16, 16×32] = [16, 512]
all2 = np.concatenate([feat2[i] for i in range(32)], axis=1)
ax.imshow(all2, cmap="viridis")
ax.set_title("Conv2 输出 — 32 张特征图，每张 16×16（深层：纹理、形状）", fontsize=11)
ax.axis("off")

plt.tight_layout()
plt.show()

print("""
观察要点:
  1. 浅层特征图还能看出原图的轮廓（亮度对应激活强度）
  2. 深层特征图变抽象了 — 不再像原图，更像"概念热力图"
  3. 有些特征图全黑 → 这个通道对这个输入没有反应（正常）

这就是 CNN 的"逐层抽象" — 卷积→ReLU→Pool 一层层把像素变成"概念"
""")
print("✅ 关卡 5 PyTorch版 完成")

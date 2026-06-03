# -*- coding: utf-8 -*-
"""
关卡 5 PyTorch版: CNN 逐层特征图可视化
把每一层卷积输出画出来，亲眼看到"浅层=边缘，深层=形状"
"""
import torch, torch.nn as nn, torchvision.transforms as transforms, torchvision
import numpy as np
import matplotlib
matplotlib.use("TkAgg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══ 1. 定义模型（把每一层单独命名，方便取出中间输出）═══
class VisualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)    # 第一层卷积
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # 第二层卷积
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*8*8, 10)

    def forward(self, x):
        # 手动走每层，存下中间结果
        self.feat1 = self.conv1(x)      # [B, 16, 32, 32]  ← 低级特征
        x = self.relu1(self.feat1)
        x = self.pool1(x)               # [B, 16, 16, 16]
        self.feat2 = self.conv2(x)      # [B, 32, 16, 16]  ← 高级特征
        x = self.relu2(self.feat2)
        x = self.pool2(x)               # [B, 32, 8, 8]
        x = self.flatten(x)
        return self.fc(x)

# ═══ 2. 随便找一张 CIFAR-10 图 ═══
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
test_set = torchvision.datasets.CIFAR10("./data", train=False, transform=transform, download=True)
img, label = test_set[0]   # 第一张测试图
classes = ["飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"]
print(f"真值: {classes[label]}")

# ═══ 3. 前向传播 + 提取特征 ═══
model = VisualCNN().to(device)
model.eval()
with torch.no_grad():
    out = model(img.unsqueeze(0).to(device))

# ═══ 4. 可视化 ═══
feat1 = model.feat1[0].cpu().numpy()   # [16, 32, 32]
feat2 = model.feat2[0].cpu().numpy()   # [32, 16, 16]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f"真值: {classes[label]} — 浅层 vs 深层特征图", fontweight="bold", fontsize=14)

# 上层：第一层 16 张特征图
ax = axes[0]
all1 = np.concatenate([feat1[i] for i in range(16)], axis=1)
ax.imshow(all1, cmap="viridis")
ax.set_title(f"Conv1 输出 — 16 张特征图，每张 32×32（浅层：边缘、颜色）", fontsize=11)
ax.axis("off")

# 下层：第二层 32 张特征图
ax = axes[1]
all2 = np.concatenate([feat2[i] for i in range(32)], axis=1)
ax.imshow(all2, cmap="viridis")
ax.set_title(f"Conv2 输出 — 32 张特征图，每张 16×16（深层：纹理、形状）", fontsize=11)
ax.axis("off")

plt.tight_layout(); plt.show()

print("""
观察要点:
  1. 浅层特征图还能看出原图的轮廓（亮度对应激活强度）
  2. 深层特征图变抽象了 — 不再像原图，更像"概念热力图"
  3. 有些特征图全黑 → 那层对这个输入没有反应（正常）

这就是 CNN 的"逐层抽象" — 卷积→ReLU→Pool 一层层把像素变成"概念"
""")
print("✅ 关卡 5 PyTorch版 完成")

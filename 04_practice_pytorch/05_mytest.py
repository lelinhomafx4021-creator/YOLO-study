import torch, torch.nn as nn, torchvision.transforms as transforms, torchvision
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*8*8, 10)
    def forward(self, x):
        self.feat1 = self.conv1(x)
        x = self.relu(self.feat1)
        x = self.maxpool1(x)
        self.feat2 = self.conv2(x)
        x = self.relu2(self.feat2)
        x = self.maxpool2(x)
        x = self.flatten(x)
        return self.fc(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
img, label = test_set[0]
classes = ["飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"]

model = TinyModel().to(device)
model.eval()
with torch.no_grad():
    out = model(img.unsqueeze(0).to(device))

feat1 = model.feat1[0].cpu().numpy()
feat2 = model.feat2[0].cpu().numpy()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f"浅层和深层的对照 — 真值: {classes[label]}", fontweight="bold", fontsize=14)

ax = axes[0]
all1 = np.concatenate([feat1[i] for i in range(16)], axis=1)
ax.imshow(all1, cmap="viridis")
ax.set_title("Conv1 输出 — 16 张特征图（浅层：边缘、颜色）")
ax.axis("off")

ax = axes[1]
all2 = np.concatenate([feat2[i] for i in range(32)], axis=1)
ax.imshow(all2, cmap="viridis")
ax.set_title("Conv2 输出 — 32 张特征图（深层：纹理、形状）")
ax.axis("off")

plt.tight_layout()
plt.show()

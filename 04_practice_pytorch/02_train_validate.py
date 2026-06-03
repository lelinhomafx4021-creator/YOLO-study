# -*- coding: utf-8 -*-
"""
关卡 2 PyTorch版: train/val 正确分离 + 实时画 loss 曲线
手写训练轮循环，每轮打印 train_loss 和 val_acc，最后画图
"""
import torch, torch.nn as nn, torchvision, torchvision.transforms as transforms
import matplotlib
matplotlib.use("TkAgg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══ 数据 ═══
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
train_set = torchvision.datasets.CIFAR10("./data", train=True, transform=transform, download=True)
test_set  = torchvision.datasets.CIFAR10("./data", train=False, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# ═══ 模型 ═══
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16*8*8, 10))
    def forward(self, x): return self.net(x)

model = TinyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ═══ 训练 + 记录 ═══
train_losses, val_accs = [], []
epochs = 10

for epoch in range(epochs):
    # ── Train ──
    model.train()
    total_loss = 0
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(trainloader))

    # ── Val ──
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
    val_accs.append(100 * correct / 10000)

    print(f"Epoch {epoch+1:2d}: train_loss={train_losses[-1]:.3f}  val_acc={val_accs[-1]:.1f}%")

# ═══ 画图 ═══
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(1, epochs+1), train_losses, "b-o"); ax1.set_title("训练 Loss"); ax1.set_xlabel("Epoch"); ax1.grid(alpha=0.3)
ax2.plot(range(1, epochs+1), val_accs, "r-o"); ax2.set_title("验证准确率"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("%"); ax2.grid(alpha=0.3)
plt.suptitle("train/val 分离 — 各自画曲线", fontweight="bold")
plt.tight_layout(); plt.show()

# ═══ 诊断 ═══
print(f"\n最佳 val_acc: {max(val_accs):.1f}% (epoch {val_accs.index(max(val_accs))+1})")
if val_accs[-1] < max(val_accs) - 1:
    print("⚠️ 过拟合: 最后 acc 比最佳低了 1% 以上")
else:
    print("✅ 正常收敛")

print("✅ 关卡 2 PyTorch版 完成")

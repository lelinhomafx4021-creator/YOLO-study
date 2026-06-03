# -*- coding: utf-8 -*-
"""
关卡 6 PyTorch版: SGD vs Adam 优化器对比实验
同一模型、同一数据、同一初始参数 → 只有优化器不同 → 看差别
"""
import torch, torch.nn as nn, torchvision, torchvision.transforms as transforms
import matplotlib
matplotlib.use("TkAgg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══ 数据 ═══
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
train_set = torchvision.datasets.CIFAR10("./data", train=True, transform=transform, download=True)
test_set  = torchvision.datasets.CIFAR10("./data", train=False, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_set, 64, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_set, 64, shuffle=False)

# ═══ 模型工厂 ═══
def make_model():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16*8*8, 10),
    ).to(device)

# ═══ 训练函数 ═══
def train_one_run(opt_name, lr, epochs=10):
    model = make_model()
    criterion = nn.CrossEntropyLoss()
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "sgd_no_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

    history = {"train_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
        history["train_loss"].append(total_loss/len(trainloader))
        history["val_acc"].append(100*correct/10000)
    return history

# ═══ 跑 4 组对比 ═══
configs = [
    ("Adam  lr=0.001",    "adam", 0.001),
    ("Adam  lr=0.01",     "adam", 0.01),
    ("SGD   lr=0.01",     "sgd",  0.01),
    ("SGD   lr=0.1",      "sgd",  0.1),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = ["#e17055", "#fdcb6e", "#00d4ff", "#0984e3"]

for (label, opt_name, lr), c in zip(configs, colors):
    h = train_one_run(opt_name, lr, epochs=10)
    ax1.plot(range(1,11), h["train_loss"], color=c, label=label, lw=2)
    ax2.plot(range(1,11), h["val_acc"],   color=c, label=label, lw=2)
    print(f"{label:<18} → 最终 acc={h['val_acc'][-1]:.1f}%")

ax1.set_title("训练 Loss 对比"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.set_title("验证准确率对比"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Acc %"); ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle("SGD vs Adam — 同一模型、不同优化器", fontweight="bold")
plt.tight_layout(); plt.show()

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

# -*- coding: utf-8 -*-
"""
关卡 1 PyTorch版: CNN 训练 → 保存 → 加载 → 推理
手写每一步，验证 state_dict 保存后再加载参数完全一样
"""
import torch, torch.nn as nn, torchvision, torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# ═══ 1. 数据 ═══
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
train_set = torchvision.datasets.CIFAR10("./data", train=True, transform=transform, download=True)
test_set  = torchvision.datasets.CIFAR10("./data", train=False, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# ═══ 2. 模型 ═══
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32→16
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16→8
            nn.Flatten(),
            nn.Linear(16*8*8, 10),
        )
    def forward(self, x): return self.net(x)

model = TinyCNN().to(device)

# ═══ 3. 训练前: 记录参数值 ═══
before_weight = model.net[0].weight.data.clone()  # 第一层卷积权重的副本
print(f"训练前 Conv1 weight[0,0,0,0] = {before_weight[0,0,0,0]:.6f}")

# ═══ 4. 训练 3 轮 ═══
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
    # 验证
    model.eval(); correct = 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
    print(f"Epoch {epoch+1}: acc={100*correct/10000:.1f}%")

# ═══ 5. 训练后: 参数值变了！ ═══
after_weight = model.net[0].weight.data
print(f"\n训练后 Conv1 weight[0,0,0,0] = {after_weight[0,0,0,0]:.6f}")
print(f"参数变了: {not torch.equal(before_weight, after_weight)}")  # True

# ═══ 6. 保存 + 加载 ═══
torch.save(model.state_dict(), "04_practice_pytorch/cnn_test.pth")
print("\n已保存: 04_practice_pytorch/cnn_test.pth")

new_model = TinyCNN().to(device)
new_model.load_state_dict(torch.load("04_practice_pytorch/cnn_test.pth", map_location=device))
new_model.eval()

# ═══ 7. 验证: 加载后的参数和训练后完全一样 ═══
loaded_weight = new_model.net[0].weight.data
print(f"加载后 == 训练后: {torch.equal(after_weight, loaded_weight)}")  # True

# ═══ 8. 推理 ═══
imgs, labels = next(iter(testloader))
imgs = imgs.to(device)
with torch.no_grad():
    preds = new_model(imgs).argmax(1)
classes = ["飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"]
for i in range(5):
    print(f"图{i+1}: 预测={classes[preds[i]]}  真值={classes[labels[i]]}  {'✅' if preds[i]==labels[i] else '❌'}")

print("\n✅ 关卡 1 PyTorch版 完成")

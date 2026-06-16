import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
train_set = torchvision.datasets.CIFAR10("./data", train=True, transform=transform, download=True)
test_set  = torchvision.datasets.CIFAR10("./data", train=False, transform=transform, download=True)
trainsdataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
testdataloder    = torch.utils.data.DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=0)

class Tinymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*8*8, 10)
        )
    def forward(self, x):
        return self.net(x)

model = Tinymodel().to(device)
creterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_losses, test_losses = [], []
epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    model.train()
    for images, labels in trainsdataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = creterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(trainsdataloader))

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in testdataloder:
            images, labels = images.to(device), labels.to(device)
            results = model(images)
            correct += (results.argmax(1) == labels).sum().item()
    test_losses.append(100 * correct / 10000)
    print(f"Epoch {epoch+1:2d}: train_loss={train_losses[-1]:.3f}, 测试的正确率为{test_losses[-1]:.1f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(1, epochs+1), train_losses, "b-o")
ax1.set_title("训练Loss")
ax1.set_xlabel("Epoch")
ax1.grid(alpha=0.3)
ax2.plot(range(1, epochs+1), test_losses, "r-o")
ax2.set_title("准确率验证")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("%")
ax2.grid(alpha=0.3)
plt.suptitle("train/val分离-各自画曲线", fontweight="bold")
plt.tight_layout()
plt.show()
print(f"最佳的正确率为{max(test_losses):.1f}%, 出现在epoch{test_losses.index(max(test_losses))+1}")
if test_losses[-1] < max(test_losses) - 1:
    print("已经过拟合了")
else:
    print("正常收敛")

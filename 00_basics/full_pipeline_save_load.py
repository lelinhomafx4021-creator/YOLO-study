# -*- coding: utf-8 -*-
"""
full_pipeline_save_load.py — 完整流程：训练 → 保存 → 加载 → 推理
每一步都标注了"为什么这样做"，跑一遍就全通了。
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ================================================================
# 第 0 步：设备 + 数据准备（和其他脚本一样）
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_set = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
testloader  = torch.utils.data.DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# ================================================================
# 第 1 步：定义模型 — 和你的 09_truning_test1.py 一样
# ================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 448 个参数
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 4,640 个参数
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8, 128),                      # 262,272 个参数
            nn.ReLU(),
            nn.Linear(128, 10),                           # 1,290 个参数
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ================================================================
# 第 2 步：查看训练前的参数（随机初始化的）
# ================================================================
print("\n" + "=" * 55)
print("训练前 — 查看模型参数")
print("=" * 55)

model = SimpleCNN().to(device)

# state_dict() 返回当前所有参数的快照 — 现在是随机的
before_state = model.state_dict()

# 只看第一层卷积的前 3 个值
conv1_weight_before = before_state["features.0.weight"]  # shape: [16, 3, 3, 3]
print(f"Conv1 weight 形状: {conv1_weight_before.shape}")
print(f"Conv1 weight 前 5 个值(随机): {conv1_weight_before.flatten()[:5].tolist()}")
print(f"Conv1 bias 前 3 个值(随机):   {before_state['features.0.bias'][:3].tolist()}")

# 遍历所有参数层，打印形状和参数量
total_params = 0
print(f"\n{'层名':<30} {'形状':<25} {'参数量':>10}")
print("-" * 70)
for name, param in before_state.items():
    num = param.numel()  # numel() = number of elements = 参数个数
    total_params += num
    print(f"{name:<30} {str(list(param.shape)):<25} {num:>10,}")
print("-" * 70)
print(f"总参数量: {total_params:,}")


# ================================================================
# 第 3 步：训练（参数在每一步都会被修改）
# ================================================================
print("\n" + "=" * 55)
print("开始训练 — 参数每批数据都在变")
print("=" * 55)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5  # 训 5 轮看个效果

for epoch in range(epochs):
    # ── 训练 ──
    model.train()
    train_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ── 验证 ──
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}  "
          f"train loss: {train_loss/len(trainloader):.4f}  "
          f"test acc: {acc:.1f}%")


# ================================================================
# 第 4 步：训练后 — 再看参数（已经变成了"会看图"的值）
# ================================================================
print("\n" + "=" * 55)
print("训练后 — 参数已被梯度下降修改了")
print("=" * 55)

after_state = model.state_dict()

# 同一层，对比训练前后
conv1_weight_after = after_state["features.0.weight"]
print(f"Conv1 weight 前 5 个值(训练后): {conv1_weight_after.flatten()[:5].tolist()}")
print(f"Conv1 bias 前 3 个值(训练后):   {after_state['features.0.bias'][:3].tolist()}")

# 验证：训练前后的参数值确实变了
print(f"\nConv1 weight 是否变了: {not torch.equal(conv1_weight_before, conv1_weight_after)}")
print(f"Conv1 bias 是否变了:   {not torch.equal(before_state['features.0.bias'], after_state['features.0.bias'])}")


# ================================================================
# 第 5 步：保存训练好的参数（推荐方式：只存 state_dict）
# ================================================================
print("\n" + "=" * 55)
print("保存模型 — torch.save(model.state_dict(), ...)")
print("=" * 55)

save_path = "00_basics/my_trained_cnn.pth"

# 只存参数值，不存模型结构 → 文件小、跨环境安全
torch.save(model.state_dict(), save_path)
print(f"模型参数已保存到: {save_path}")

# 检查文件大小
import os
size_kb = os.path.getsize(save_path) / 1024
print(f"文件大小: {size_kb:.0f} KB  (约 {total_params:,} 个参数 × 4 字节/float32)")


# ================================================================
# 第 6 步：模拟"换了一台电脑"——重新加载模型做推理
# ================================================================
print("\n" + "=" * 55)
print("加载模型 — 模拟部署环境")
print("=" * 55)

# 6.1 建一个全新的空模型（参数是随机初始化的）
new_model = SimpleCNN().to(device)

# 6.2 把保存的参数灌进去
new_model.load_state_dict(torch.load(save_path, map_location=device))

# 6.3 切到推理模式
new_model.eval()

print("模型加载完成！现在的参数和训练后完全一样。")

# 验证：加载后的参数和保存前的参数一模一样
loaded_state = new_model.state_dict()
print(f"加载后 Conv1 weight == 训练后 Conv1 weight: "
      f"{torch.equal(loaded_state['features.0.weight'], conv1_weight_after)}")
# 输出应该是 True


# ================================================================
# 第 7 步：用加载的模型做推理
# ================================================================
print("\n" + "=" * 55)
print("推理 — 用加载的模型预测几张图")
print("=" * 55)

# 取验证集的一批图做预测
data_iter = iter(testloader)
images, labels = next(data_iter)
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    output = new_model(images)
    _, predicted = torch.max(output, 1)

# 打印前 8 张图的预测结果
class_names = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
print(f"\n{'图片':<6} {'预测':<6} {'真值':<6} {'正确?':<8}")
print("-" * 35)
for i in range(8):
    pred_cls = predicted[i].item()
    true_cls = labels[i].item()
    ok = "✅" if pred_cls == true_cls else "❌"
    print(f"  {i+1:<4} {class_names[pred_cls]:<6} {class_names[true_cls]:<6} {ok}")


# ================================================================
# 第 8 步（额外）：保存完整 checkpoint（可以断点续训）
# ================================================================
print("\n" + "=" * 55)
print("保存完整 checkpoint（断点续训用）")
print("=" * 55)

checkpoint = {
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'accuracy': acc,
}
torch.save(checkpoint, "00_basics/my_checkpoint.pth")
print("Checkpoint 已保存: 00_basics/my_checkpoint.pth")
print(f"  → 包含: epoch={checkpoint['epoch']}, "
      f"accuracy={checkpoint['accuracy']:.1f}%")
print(f"  → 下次训练时 torch.load 这个文件，可以接着训")


# ================================================================
# 总结
# ================================================================
print("\n" + "=" * 55)
print("完整流程总结")
print("=" * 55)
print("""
  1. model = SimpleCNN()           → 参数随机初始化
  2. 训练 N 轮                       → 参数被梯度下降一步步修改
  3. torch.save(state_dict, ...)   → 保存训练好的参数值
  4. new_model = SimpleCNN()       → 部署时新建空模型
  5. load_state_dict(torch.load()) → 把保存的参数灌进新模型
  6. model.eval() + no_grad()      → 推理预测

  关键理解：
  - state_dict() 是实时快照，任何时候调用都返回当前参数值
  - torch.save 只存参数值，不存模型结构
  - 加载时必须先建一个结构一样的空模型
  - 训练前后的参数完全是两套不同的数值
""")

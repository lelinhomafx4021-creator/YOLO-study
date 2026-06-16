# -*- coding: utf-8 -*-
"""
关卡 2 PyTorch版: train/val 正确分离 + 实时画 loss 曲线

═══════════════════════════════════════════════════════════════════════════════
关卡 1 vs 关卡 2 的区别
═══════════════════════════════════════════════════════════════════════════════

  关卡 1: 训练 → 保存 → 加载 → 推理（一条流水线走到底）
  关卡 2: 训练时每个 epoch 记录 train_loss + val_acc → 画曲线 → 诊断问题

  新增的东西:
    1. train/val 分离: 每轮训练完立刻在测试集上验证
    2. 记录历史: 用列表存每个 epoch 的 loss 和 acc
    3. 画图: matplotlib 画双图（左 loss 右 acc）
    4. 诊断: 自动判断是否过拟合

═══════════════════════════════════════════════════════════════════════════════
为什么 train/val 要分离
═══════════════════════════════════════════════════════════════════════════════

  只用训练集看 loss 是不够的——loss 一直在降不代表模型在"学"，可能只是在"背"。

  train_loss: 模型在训练集上的误差 → 只能看"有没有在学"
  val_acc:    模型在测试集上的准确率 → 能看"学会了没有"

  两个指标一起看:
    train_loss ↓  + val_acc ↑  → 正常，在进步
    train_loss ↓  + val_acc ↓  → 过拟合，在背答案
    train_loss →  + val_acc →  → 欠拟合，没学到东西
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ── matplotlib 导入 + 配置 ───────────────────────────────────────────────
import matplotlib
# matplotlib.use("TkAgg"): 指定后端 = 用什么方式渲染图表窗口
#   TkAgg = Tkinter 窗口（Windows/Mac/Linux 都能用，最通用）
#   Agg   = 只生成图片文件，不弹窗（服务器上跑用这个）
#   Qt5Agg = Qt5 窗口（比 Tkinter 好看一点，但要额外装 PyQt5）
#   必须在 import matplotlib.pyplot 之前设置！否则后端已锁定
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── 让 matplotlib 支持中文 ──
# font.sans-serif = 无衬线字体列表 → 优先用微软雅黑
#   不设置的话中文会显示为 ▯ 方块
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
# axes.unicode_minus = False → 负号正常显示
#   不设置的话负号 `-` 也会变方块
matplotlib.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# 1. 数据
# ═════════════════════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
train_set = torchvision.datasets.CIFAR10("./data", train=True,
                                          transform=transform, download=True)
test_set  = torchvision.datasets.CIFAR10("./data", train=False,
                                          transform=transform, download=True)

# 注意: 这里 testloader 其实是验证集 (val)，不是真正"你没见过的测试集"
# CIFAR10 只给了 train 和 test 两份数据，这里把 test 当 val 用
# 真正做项目时应该是 train / val / test 三份
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_set,  batch_size=64, shuffle=False)
#                                          ↑ 名字叫 test，实际当 val 用


# ═════════════════════════════════════════════════════════════════════════════
# 2. 模型 (和关卡 1 完全一样)
# ═════════════════════════════════════════════════════════════════════════════
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16*8*8, 10)
        )
    def forward(self, x):
        return self.net(x)

model = TinyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ═════════════════════════════════════════════════════════════════════════════
# 3. 训练 + 记录（核心新增: 每个 epoch 存 loss 和 acc）
# ═════════════════════════════════════════════════════════════════════════════

# ── 两个列表，记录每个 epoch 的数据 ──
# train_losses[i] = 第 i 轮的平均训练 loss
# val_accs[i]     = 第 i 轮的验证准确率
# 最后用这两个列表画折线图
train_losses, val_accs = [], []

epochs = 10  # 训练 10 轮（关卡 1 只跑了 3 轮，太少了看不出趋势）

for epoch in range(epochs):

    # ── 3a. 训练阶段 ──────────────────────────────────────────────────
    model.train()               # 切换到训练模式
    total_loss = 0              # 这一轮所有 batch 的 loss 累加

    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()                    # 清梯度
        loss = criterion(model(imgs), labels)    # 前向 + 算 loss
        loss.backward()                          # 反向传播
        optimizer.step()                         # 更新参数

        # loss.item(): 把单元素 tensor 转成 Python float
        #   例: tensor(2.345) → 2.345
        #   为什么要转? tensor 保存着计算图，会占额外显存
        #   .item() 取出来 → 断开计算图 → 省显存
        total_loss += loss.item()

    # 平均 loss = 总 loss / batch 数
    #   len(trainloader) = 782 个 batch (50000/64，最后一个略少)
    #   例: total_loss=1800 → avg=2.30  (每个 batch 平均错多少)
    train_losses.append(total_loss / len(trainloader))

    # ── 3b. 验证阶段 ──────────────────────────────────────────────────
    model.eval()                # 切换到评估模式
    correct = 0

    with torch.no_grad():       # 不计算梯度（验证不需要反向传播）
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()

    # 准确率 = 预测对的 / 总数 × 100 → 百分比
    #   10000 = 测试集总共 10000 张
    val_accs.append(100 * correct / 10000)

    # 每轮打印一行: 第几轮 | loss 多少 | acc 多少
    print(f"Epoch {epoch+1:2d}: train_loss={train_losses[-1]:.3f}  val_acc={val_accs[-1]:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# 4. 画图: 左图 loss 曲线 + 右图 acc 曲线
# ═════════════════════════════════════════════════════════════════════════════

# ── plt.subplots(行数, 列数, figsize=(宽, 高)) ──
#   创建 1 行 2 列的子图布局
#   fig  = 整张画布
#   ax1  = 左边子图 (loss)
#   ax2  = 右边子图 (acc)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ── 左图: 训练 Loss ──
# plot(x数据, y数据, "b-o"):
#   b = blue 蓝色
#   o = 每个数据点画圆点标记
#   - = 点之间用线连起来
ax1.plot(range(1, epochs+1), train_losses, "b-o")  # x=1~10, y=loss值
ax1.set_title("训练 Loss")
ax1.set_xlabel("Epoch")
ax1.grid(alpha=0.3)           # 加网格线，alpha=0.3=半透明，不抢眼

# ── 右图: 验证准确率 ──
# plot(..., "r-o"):
#   r = red 红色
ax2.plot(range(1, epochs+1), val_accs, "r-o")      # x=1~10, y=acc%
ax2.set_title("验证准确率")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("%")           # 跟左图不同: y轴是百分比，所以要标单位
ax2.grid(alpha=0.3)

# ── 总标题 + 显示 ──
plt.suptitle("train/val 分离 — 各自画曲线", fontweight="bold")
plt.tight_layout()            # 自动调整子图间距，避免重叠
plt.show()                    # 弹出窗口显示图表（阻塞，关了窗口才继续）


# ═════════════════════════════════════════════════════════════════════════════
# 5. 诊断: 自动判断有没有问题
# ═════════════════════════════════════════════════════════════════════════════

# max(val_accs)             → 所有轮里最高的准确率，比如 58.2
# val_accs.index(max(...))  → 最高值出现在第几轮 (0-based)
print(f"\n最佳 val_acc: {max(val_accs):.1f}% (epoch {val_accs.index(max(val_accs))+1})")

# ── 过拟合检测 ──
# 如果最后一轮的 acc 比历史最高值低了超过 1%
#   → 说明模型在训练集上继续"死记"，但泛化能力在下降
#   → 典型的过拟合信号
# 例: epoch 5 最高 58%, epoch 10 只有 56% → 过拟合
if val_accs[-1] < max(val_accs) - 1:
    print("⚠️ 过拟合: 最后 acc 比最佳低了 1% 以上")
else:
    print("✅ 正常收敛")

print("✅ 关卡 2 PyTorch版 完成")

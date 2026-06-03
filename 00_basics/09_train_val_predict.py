import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# 第一步：造一份最小数据
# 目标规律：y = 2x + 1 + 一点点噪声
# ============================================================
torch.manual_seed(42)

x = torch.rand(200, 1)
y = 2 * x + 1 + 0.1 * torch.randn(200, 1)


# ============================================================
# 第二步：定义 Dataset
# Dataset 只负责回答两个问题：
# 1. 一共有多少条数据
# 2. 第 idx 条数据是什么
# ============================================================
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ============================================================
# 第三步：定义模型
# 一个最小线性回归模型：输入 1 个数，输出 1 个数
# ============================================================
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# ============================================================
# 第四步：训练一个 epoch
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()  # 切到训练模式
    total_loss = 0.0

    for x_batch, y_batch in dataloader:
        pred = model(x_batch)                 # 前向传播
        loss = criterion(pred, y_batch)       # 计算 loss

        optimizer.zero_grad()                 # 清旧梯度
        loss.backward()                       # 反向传播
        optimizer.step()                      # 更新参数

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# 第五步：验证
# 注意这里不更新参数，只看模型效果
# ============================================================
def validate(model, dataloader, criterion):
    model.eval()  # 切到评估模式
    total_loss = 0.0

    with torch.no_grad():  # 验证时不需要梯度
        for x_batch, y_batch in dataloader:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# 第六步：预测
# 这里只演示“输入几个新样本，看看模型输出”
# ============================================================
def predict_samples(model):
    model.eval()

    test_x = torch.tensor([[0.1], [0.5], [0.9]])

    with torch.no_grad():
        pred_y = model(test_x)

    print("\n预测阶段：")
    for i in range(len(test_x)):
        x_value = test_x[i].item()
        pred_value = pred_y[i].item()
        true_value = 2 * x_value + 1
        print(
            f"x={x_value:.2f} | 预测 y={pred_value:.3f} | 理论 y={true_value:.3f}"
        )


# ============================================================
# 第七步：主流程
# ============================================================
def main():
    dataset = MyDataset(x, y)

    # 划分训练集 / 验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = MyModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 30

    print("开始训练...\n")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

    print("\n训练完成。")
    print(f"学到的 weight: {model.linear.weight.item():.4f}")
    print(f"学到的 bias:   {model.linear.bias.item():.4f}")

    predict_samples(model)


if __name__ == "__main__":
    main()

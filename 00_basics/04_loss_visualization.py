# -*- coding: utf-8 -*-
"""
04_loss_visualization.py - 训练中 Loss 变化过程与可视化
此脚本整合了原先的 03_看loss变化.py。
通过在终端输出“由字符块拼出的简易直方图”，生动直观地演示模型在训练过程中如何一步步逼近最优解，Loss 值如何一路下降。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. 简易模拟二分类数据集
class MyDataset(Dataset):
    def __init__(self, n_samples=100):
        self.x = torch.randn(n_samples, 4)
        self.y = torch.randint(0, 2, (n_samples,)).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 2. 简易单层线性模型
class SingleLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x)).squeeze()

def main():
    dataset = MyDataset(n_samples=120)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SingleLayerModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print("=" * 65)
    print("🚀 开始训练，观察训练轮数（Epoch）增加时，Loss 值的神奇演变：")
    print("=" * 65)

    loss_history = []

    # 跑 20 轮 Epoch
    for epoch in range(20):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        # 🌟 炫酷终端进度条：把当前的 Loss 大小映射为相应数量的方块 “█”
        # Loss 越大，方块越多；Loss 越小，方块越少，形成直观的下降阶梯
        bar_length = int(avg_loss * 60)
        bar = "█" * bar_length
        print(f"Epoch {epoch+1:2d}/20 | Loss: {avg_loss:.4f} | {bar}")

    print("=" * 65)
    print("🎉 训练完成！")
    print(f"  - 初始 Loss: {loss_history[0]:.4f}")
    print(f"  - 最终 Loss: {loss_history[-1]:.4f}")
    print(f"  - 总共下降了: {loss_history[0] - loss_history[-1]:.4f} (代表模型成功在学到规律)")
    print("=" * 65)

if __name__ == "__main__":
    main()

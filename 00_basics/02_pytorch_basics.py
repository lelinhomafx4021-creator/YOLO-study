# -*- coding: utf-8 -*-
"""
02_pytorch_basics.py - 极简 PyTorch 神经网络训练
此脚本整合了原先的 01_minimal_train.py。
展示了一个网络模型是如何从零初始化，并通过 Dataset/DataLoader 输入数据、计算 Loss、反向传播更新参数的完整闭环。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 第一步：定义数据集（数据怎么从硬盘或内存中取出来）
# ============================================================
class MyDataset(Dataset):
    """
    自定义数据集，必须实现两个魔术方法：
    - __len__：告诉 PyTorch 数据集里一共有多少条数据。
    - __getitem__：告诉 PyTorch 怎么根据索引 idx 拿到第 idx 条数据。
    """
    def __init__(self, n_samples=100):
        # 模拟生成 100 条随机数据，每个样本有 4 个特征
        self.x = torch.randn(n_samples, 4)
        # 模拟生成对应的二分类标签（0 或 1），并转为浮点型以配合 BCE 损失函数
        self.y = torch.randint(0, 2, (n_samples,)).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 返回一个包含 (特征, 标签) 的元组
        return self.x[idx], self.y[idx]


# ============================================================
# 第二步：使用 DataLoader 对数据进行分批（Batch）和打乱（Shuffle）
# ============================================================
def test_dataloader():
    dataset = MyDataset(n_samples=100)
    # batch_size=16 代表每次给模型喂 16 张图/数据进行训练
    # shuffle=True 代表每个 Epoch（跑完所有图片）之后，把图片顺序全部打乱防作弊
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 观察一个批次的数据结构
    x_batch, y_batch = next(iter(dataloader))
    print("📋 DataLoader 测试:")
    print(f"  - 一批数据的特征形状: {x_batch.shape} (16个样本，每个4个特征)")
    print(f"  - 一批数据的标签形状: {y_batch.shape} (对应这16个样本的答案)")
    return dataloader


# ============================================================
# 第三步：搭建神经网络模型结构（Model）
# ============================================================
class SimpleClassifier(nn.Module):
    """
    最基础的二分类神经网络。
    包含一个全连接层（nn.Linear）和一个 Sigmoid 激活函数（将预测值压缩在 0~1 的概率区间）。
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)   # 输入维度为 4，输出维度为 1
        self.sigmoid = nn.Sigmoid()     # 激活函数，输出概率

    def forward(self, x):
        # 定义数据的“前向盲猜”流动过程
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.squeeze()    # 压缩多余的一维维度，例如从 [16, 1] 变成 [16]


# ============================================================
# 第四步：开始跑正式训练循环（用 Loss 挨打改错，Optimizer 更新脑细胞）
# ============================================================
def main():
    # 实例化 DataLoader 与模型
    dataloader = test_dataloader()
    model = SimpleClassifier()
    print("\n🤖 我们的模型结构:\n", model)

    # 1. 定义损失函数：二分类任务专用 Binary Cross Entropy Loss
    criterion = nn.BCELoss()

    # 2. 定义优化器：使用 SGD 梯度下降，lr=0.1 是学习率（每一步走多大）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print("\n🔥 开始迭代训练...")
    # 训练 10 轮 (Epochs)
    for epoch in range(10):
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            # ① 前向传播：模型根据当前参数“盲做题目”得到预测值
            y_pred = model(x_batch)

            # ② 对答案算 Loss：对比预测值与真实答案
            loss = criterion(y_pred, y_batch)

            # ③ 清空梯度 + 反向传播：算出每一层参数的梯度（方向）
            optimizer.zero_grad()   # 必须清空梯度，否则会被累加到下一批
            loss.backward()         # 自动算偏导数与梯度

            # ④ 优化器更新参数：利用梯度更新模型权重
            optimizer.step()

            # 累计当前 batch 的 loss 值
            total_loss += loss.item()

        # 计算并打印这一轮的平均 Loss
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1:2d}/10 | 平均 Loss: {avg_loss:.4f}")

    # ============================================================
    # 第五步：验证推理（用学好的模型预测未知数据）
    # ============================================================
    print("\n🎉 训练完毕！测试模型对全新未知数据的预测能力:")
    unknown_data = torch.randn(5, 4)  # 随机生成 5 个新样本
    
    with torch.no_grad():          # 推理测试阶段不需要计算梯度，节省内存和速度
        predictions = model(unknown_data)
        prob_arr = predictions.numpy().round(3)
        class_arr = (predictions > 0.5).int().numpy()
        print(f"  - 预测为1的概率: {prob_arr}")
        print(f"  - 判定最终类别  : {class_arr}")

if __name__ == "__main__":
    main()

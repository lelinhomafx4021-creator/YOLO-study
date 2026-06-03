# -*- coding: utf-8 -*-
"""
03_pytorch_vs_yolo.py - 底层 PyTorch 写法与 YOLO 框架的对比
此脚本整合了原先的 02_对比_底层vs框架.py 与 07_yolotest3.py 的核心讲解。
让初学者明白：
- 底层 PyTorch 写目标检测需要定义 Backbone、Head、数据增强、复杂的后处理、以及庞大的 Epoch 循环。
- 而 YOLO 框架将这些高度封装为 `YOLO("model.pt")` -> `model.train()`，极大地提高了开发效率。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 方式一：底层的 PyTorch 目标检测写法 (以简化版网络结构展示)
# ============================================================

# 1. 定义简化版的 YOLO 检测器
class MiniYOLONet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Backbone：特征提取网络，提取图片中的高维卷积特征
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 再下采样
        )
        # Head：检测头，用于预测物体的候选框坐标 (x,y,w,h)、置信度、以及类别概率
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 160 * 160, 128),
            nn.ReLU(),
            # 模拟输出：10个候选框 * (4个坐标 + 1个置信度 + 3个分类概率)
            nn.Linear(128, 10 * (5 + num_classes))
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions


# 2. 模拟数据集定义
class DummyDetectionDataset(Dataset):
    def __init__(self, n_samples=20):
        # 模拟 20 张三通道、640x640 的图片
        self.images = torch.randn(n_samples, 3, 640, 640)
        # 模拟对应的坐标与类别标注
        self.labels = torch.randn(n_samples, 10 * 8) # 10个框 * 8个预测量

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# 3. 运行底层的训练流程
def run_low_level_pytorch():
    print("=" * 60)
    print("📋 方式一：底层的 PyTorch 原生目标检测训练循环")
    print("=" * 60)

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - 自动检测设备: {device}")

    # 数据集准备
    dataset = DummyDetectionDataset(n_samples=16)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 实例化模型并移至对应设备
    model = MiniYOLONet(num_classes=3).to(device)
    
    # 定义 Loss 与 优化器
    criterion = nn.MSELoss()  # 简化演示用均方根损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("  - 开始迭代...")
    for epoch in range(3):  # 快速迭代 3 轮演示
        epoch_loss = 0.0
        for images, labels in dataloader:
            # 物理迁移到 GPU 或者是 CPU
            images, labels = images.to(device), labels.to(device)

            # 前向计算 ➡️ 算 Loss ➡️ 梯度归零 ➡️ 反向传播 ➡️ 优化更新
            preds = model(images)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"    Epoch {epoch+1}/3 | 均方误差 Loss: {epoch_loss/len(dataloader):.4f}")
    
    # 模拟推理
    model.eval()
    with torch.no_grad():
        test_img = torch.randn(1, 3, 640, 640).to(device)
        output = model(test_img)
        print(f"  - 推理完成！预测候选框张量形状: {output.shape}")
    print("=" * 60 + "\n")


# ============================================================
# 方式二：极简的 Ultralytics YOLO 框架写法
# ============================================================
def run_yolo_framework():
    print("=" * 60)
    print("📋 方式二：使用 Ultralytics YOLO 框架的高速一键式封装")
    print("=" * 60)
    print("在实际项目中，我们只需要引入 YOLO 并传入配置文件路径即可。")
    print("YOLO 框架在底层为你全自动实现了：")
    print("  1. 极其复杂的 Backbone（如 DarkNet/C3/C2f/Elan）")
    print("  2. 分类与定位的多头损失计算（CIoU Loss + DFL Loss 等）")
    print("  3. 输入图片的批量 Resize、归一化、在线数据增强（Mosaic、Mixup等）")
    print("  4. 复杂的推理后处理（非极大值抑制 NMS 算法，过滤重叠框）")
    print("  5. GPU 多卡训练与半精度（FP16）自动混合精度训练控制")
    
    print("\n  核心伪代码仅需 3 行：")
    print("  ------------------------------------------------")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolo11n.pt')                # 1. 极速加载底座模型")
    print("  model.train(data='custom_data.yaml')      # 2. 一键式微调训练")
    print("  results = model('test.jpg')               # 3. 极速执行推理预测")
    print("  ------------------------------------------------")
    print("=" * 60)


if __name__ == "__main__":
    run_low_level_pytorch()
    run_yolo_framework()

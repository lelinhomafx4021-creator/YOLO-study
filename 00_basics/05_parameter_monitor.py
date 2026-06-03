# -*- coding: utf-8 -*-
"""
05_parameter_monitor.py - 参数参数（权重）更新监控与学习率验证
此脚本整合了原先的 04_看参数怎么变.py。
建立最极简的单参数回归：拟合直线 y = 2 * x (输入x, 模型 y = w * x, 目标是学到w=2)。
让初学者肉眼看清：在前向盲猜、计算Loss、反向求出梯度后，优化器如何根据“学习率”一步一步把 w 纠正为 2.0。
"""

import torch
import torch.optim as optim

def main():
    # 1. 初始化我们要学习的参数 w
    # 我们故意猜一个离 2.0 很远的值，比如 w = 0.5。并声明需要求梯度（requires_grad=True）
    w = torch.tensor(0.5, requires_grad=True)
    print("==============================================================")
    print(f"🎯 训练目标：让模型通过数据，自动学到公式 [y = 2 * x] 的斜率。")
    print(f"🚀 初始化参数：w = {w.item():.4f} (我们期待它通过训练逼近 2.0)")
    print("==============================================================")

    # 2. 准备简易物理数据集 (输入 x 与真实结果 y)
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_data = torch.tensor([2.0, 4.0, 6.0, 8.0])  # y = 2 * x

    # 3. 使用 SGD 优化器，学习率设为 0.01
    optimizer = optim.SGD([w], lr=0.01)

    print(f"{'Epoch':^8} | {'当前参数 w':^12} | {'模型预测结果':^20} | {'当前 Loss':^10} | {'计算出的梯度':^10}")
    print("-" * 75)

    # 训练 20 轮，观察参数的跃迁
    for epoch in range(20):
        # ① 前向传播：用当前 w 计算预测值
        y_pred = w * x_data

        # ② 计算损失：使用 Mean Squared Error (均方误差)
        loss = ((y_pred - y_data) ** 2).mean()

        # ③ 清空上一轮的梯度
        optimizer.zero_grad()
        
        # ④ 反向传播计算当前的梯度
        loss.backward()

        # 记录本轮打印所需的数据
        curr_w = w.item()
        curr_loss = loss.item()
        curr_grad = w.grad.item()

        # ⑤ 优化器沿着梯度负方向更新参数：w = w - lr * grad
        optimizer.step()

        # 每隔两轮打印一次数据，看 w 的蜕变过程
        if epoch % 2 == 0:
            pred_str = ", ".join([f"{v:.1f}" for v in y_pred.detach().numpy()])
            print(f"{epoch+1:^8d} | {curr_w:^12.4f} | [{pred_str:^18s}] | {curr_loss:^10.4f} | {curr_grad:^10.4f}")

    print("-" * 75)
    print(f"🎉 训练完毕！")
    print(f"  - 最终学到的参数: w = {w.item():.4f} (非常接近 2.0000！)")
    print("\n🧠 概念总结：")
    print("  1. 梯度 (Gradient) 是指南针：告之参数要想降低 Loss，应该往哪个方向移动。")
    print("  2. 学习率 (Learning Rate) 是步伐：决定每次沿此方向迈多大的步子。步伐太大会跨过最优解，步伐太小会走得很慢。")
    print("==============================================================")

if __name__ == "__main__":
    main()

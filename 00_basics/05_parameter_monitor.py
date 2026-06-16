# -*- coding: utf-8 -*-
"""
05_parameter_monitor.py - 参数更新监控与学习率验证

目标: 用最简例子看清梯度下降的每一步

  任务: 拟合 y = 2x （已知一个公式，让模型自己学到斜率 w=2）

  只用一个参数 w:
    模型:  y = w * x
    真实:  y = 2 * x
    目标:  让 w 在训练过程中自己逼近 2.0

  训练四步走 (每轮):
    ① 前向传播:  用当前 w 算预测值 y_pred = w * x
    ② 算 Loss:   比较 y_pred 和真实值 y_data 的差距
    ③ 反向传播:  算梯度 (Loss 对 w 的偏导数)，知道该往哪改
    ④ 更新参数:  w = w - lr × grad  (往梯度反方向迈一小步)

  跑完 20 轮就能看到 w 从 0.5 一点点逼近 2.0
"""

import torch
# torch: PyTorch 核心库
#  - torch.tensor(): 创建张量 (就是多维数组，类似 NumPy 但多了 GPU + 自动求导)
import torch.optim as optim
# torch.optim: 优化器模块
#  - 包含 SGD, Adam, AdamW 等
#  - 统一接口: optim.SGD(参数列表, lr=学习率)

def main():
    # ═══════════════════════════════════════════════════════════
    # 1. 初始化参数 w
    # ═══════════════════════════════════════════════════════════

    # torch.tensor(值, requires_grad=True)
    #   值=0.5: 随便猜的初始值（真实值是 2.0，我们故意猜错）
    #   requires_grad=True: 告诉 PyTorch"这个变量要被训练，帮我追踪它的梯度"
    #     - 不设的话，backward() 跳过它，optimizer 改不了它
    #     - 设了之后，所有用到 w 的计算都会被记录下来（计算图）
    w = torch.tensor(0.5, requires_grad=True)

    print("=" * 70)
    print(f"目标: 让模型学到 y = 2x 的斜率")
    print(f"初始化 w = {w.item():.4f}   ← 随便猜的，期待训练后逼近 2.0")
    # .item(): 把标量 Tensor (shape=[]) 转成 Python float
    #   - 只对只含一个数的 Tensor 有效
    #   - .item() 后就不参与自动求导了，纯粹打印用
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # 2. 准备训练数据（输入 x + 正确答案 y）
    # ═══════════════════════════════════════════════════════════

    # torch.tensor([...]): 从 Python 列表创建一维张量
    #   dtype 默认 float32 (4 字节浮点)
    #   这些是"训练数据"——假设我们从某个物理实验中采集到的

    # x_data: 输入（4 个样本，每个是 1 个数）
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # y_data: 真实标签（正确答案，我们希望 model(x) 能输出这个）
    y_data = torch.tensor([2.0, 4.0, 6.0, 8.0])   # 每对符合 y = 2x

    # 这 4 对数据:  (1→2) (2→4) (3→6) (4→8)
    # 模型看到 x=1 应该输出接近 2，看到 x=3 应该输出接近 6

    pass

    # ═══════════════════════════════════════════════════════════
    # 3. 创建优化器
    # ═══════════════════════════════════════════════════════════

    # optim.SGD(参数列表, lr=学习率)
    #   SGD = Stochastic Gradient Descent (随机梯度下降)
    #         这里只有 4 个数据点，用的不是"随机"而是全部数据，所以是纯 GD

    #   参数列表 [w]: 告诉优化器"你要负责更新哪些变量"
    #                必须是把 requires_grad=True 的张量放进去
    #                这里只更新 w 一个

    #   lr=0.01: Learning Rate (学习率)
    #            = 每步迈多大
    #            0.01 是比较保守的学习率
    #            太小 (如 0.0001): 训练太慢，20 轮走不了多远
    #            太大 (如 0.5):   直接跨过最优解，甚至越训越远
    #            SGD 对 lr 特别敏感，所以选了个稳的 0.01
    optimizer = optim.SGD([w], lr=0.01)

    # 打印表头
    print(f"{'Epoch':^8} | {'当前 w':^12} | {'模型预测':^24} | {'Loss':^10} | {'梯度 grad':^10}")
    # ^ 居中对齐，数字是列宽
    print("-" * 80)

    # ═══════════════════════════════════════════════════════════
    # 4. 训练循环: 20 轮
    # ═══════════════════════════════════════════════════════════

    for epoch in range(20):

        # ─── ① 前向传播 (Forward) ───
        # 用当前的 w 算出预测值
        # w * x_data: 张量乘法 (标量 × 向量 = 向量)
        #   w=0.5, x_data=[1,2,3,4] → y_pred=[0.5, 1.0, 1.5, 2.0]
        #   但正确答案是 [2,4,6,8]，差了 4 倍
        y_pred = w * x_data

        # ─── ② 计算损失 (Loss) ───
        # MSE (Mean Squared Error): 预测值和真实值的平均平方差
        #
        #   (y_pred - y_data)²: 算每个样本的误差的平方
        #     - 平方的作用: 把正负误差都变成正数（不然正负抵消了）
        #     - 误差大的惩罚更重（因为平方放大了大误差）
        #
        #   .mean(): 对 4 个样本取平均
        #     如果不除，Loss 会随样本数变化，不方便比较
        #
        #   第 1 轮: y_pred=[0.5,1,1.5,2], 真=[2,4,6,8]
        #           差=[-1.5,-3,-4.5,-6], 平方=[2.25,9,20.25,36], mean=16.875
        loss = ((y_pred - y_data) ** 2).mean()

        # ─── ③ 清空梯度 ───
        # optimizer.zero_grad(): 把上一轮累积的梯度清零
        #
        #   为什么要清零？
        #     PyTorch 的梯度默认是**累加**的（不是替换）
        #     如果不调 zero_grad()，每次 backward() 的梯度会堆在上次的上面
        #     → 第一轮 grad=某值，第二轮 grad=两轮的和，越来越乱
        #
        #   类比: 每次跑新一圈之前，按一下秒表归零
        optimizer.zero_grad()

        # ─── ④ 反向传播 (Backward) ───
        # loss.backward(): 从 loss 出发，沿着计算图往回走
        #                       算出 loss 对每个 requires_grad=True 的变量的偏导数
        #
        #   计算图: x_data → (*w) → y_pred → (-y_data) → (²) → (mean) → loss
        #                                        ↑                         ↓
        #                                 w 是怎么影响 loss 的？← backward() 算出来
        #
        #   y = w*x, loss = ((y - y_true)²).mean()
        #   d(loss)/dw = 2 * mean((y - y_true) * x)  ← 链式法则
        #
        #   结果存在 w.grad 里（一个数值，表示 loss 随 w 增大的变化率）
        loss.backward()

        # 在 step() 之前先把本轮数据记下来（用于打印）
        # step() 之后 w 就变了，grad 也被清零了
        curr_w = w.item()              # 当前的 w 值（float）
        curr_loss = loss.item()        # 当前的 loss（float）
        curr_grad = w.grad.item()      # 当前的梯度（float）
        # w.grad: 是 Tensor，存的是 backward() 算出来的梯度
        #   第 1 轮大概是 -8.25 → Loss 随 w 增大而减小（w 太小），所以梯度是负的
        #   optimizer.step() 会减掉 lr*负值 = 加上一个正数 → w 增大 ✅

        # ─── ⑤ 更新参数 (Step) ───
        # optimizer.step(): 根据梯度更新参数
        #
        #   SGD 的更新公式:  w_new = w_old - lr × grad
        #   第 1 轮: w_new = 0.5 - 0.01 × (-8.25) = 0.5 + 0.0825 = 0.5825
        #
        #   梯度是负的 → 说明 w 太小了 → w 增大 ✓
        #   梯度是正的 → 说明 w 太大了 → w 减小 ✓
        #
        #   这就是"梯度下降": 沿着 Loss 减小的方向走
        optimizer.step()

        # 每 2 轮打印一行
        if epoch % 2 == 0:
            # .detach(): 从计算图中分离（不再追踪梯度），只拿数值
            # .numpy():  转成 NumPy 数组（方便格式化打印）
            pred_str = ", ".join([f"{v:.1f}" for v in y_pred.detach().numpy()])
            #         ↑ 列表推导式: 把 y_pred 每个元素格式化成 1 位小数
            #           y_pred=[0.5, 1.0, 1.5, 2.0] → "0.5, 1.0, 1.5, 2.0"
            print(f"{epoch+1:^8d} | {curr_w:^12.4f} | [{pred_str:^22s}] | {curr_loss:^10.4f} | {curr_grad:^10.4f}")
            #   ^8d:  8 列宽居中整数
            #   ^12.4f: 12 列宽居中，4 位小数浮点
            #   ^22s: 22 列宽居中字符串

    print("-" * 80)

    # ═══════════════════════════════════════════════════════════
    # 5. 训练结果
    # ═══════════════════════════════════════════════════════════

    print(f"训练完毕！")
    print(f"  初始 w: 0.5000")
    print(f"  最终 w: {w.item():.4f}  -> 接近 2.0000 说明学会了 [OK]")

    print(f"""
概念总结:
  1. 梯度 (Gradient)     → 指南针: 告诉 w 往哪个方向改 Loss 降最快
  2. 学习率 (Learning Rate) → 步长:  决定每步迈多大
       lr 太小 → 走得慢
       lr 太大 → 可能错过最优值
  3. 前向→Loss→反向→更新 → 这一轮就是"训练的一步"
  4. requires_grad=True   → 只有声明了这个，PyTorch 才追踪梯度
  5. zero_grad()          → 每次 backward() 之前要清空，否则梯度会累积
{'=' * 70}
""")


if __name__ == "__main__":
    # __name__: Python 内置变量
    #   直接跑 python 05_parameter_monitor.py → __name__ = "__main__" → 执行 main()
    #   被 import 05_parameter_monitor           → __name__ = "05_parameter_monitor" → 不执行
    # 好处: 子进程 (Windows spawn) 重新 import 时不会递归执行
    main()

# 2026-05-24 Day 2 学习计划

## 今日定位

笔记 01-09 理论已到位，今天的核心是：**把理论变成自己的代码能力**。

搜索结论（联网验证）：
- PyTorch 官方教程推荐的入门路径：Tensor → Dataset/DataLoader → 定义模型 → 训练循环 → 验证
- CIFAR-10 图片分类是 PyTorch 官方推荐的第一个实战项目（官方 tutorial: Training a Classifier）
- 最佳实践：先跑通简单线性回归（理解训练循环），再做图片分类（理解 CNN + DataLoader + 数据增强）
- 关键资源：PyTorch 官方中文教程 https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

---

## 任务清单

### 任务 1：自己从零写完整训练循环（不看参考）✅ 已完成
- [x] 新建 `00_basics/07_training_loop.py`
- [x] 定义一个 `nn.Module` 模型（线性回归：y = w*x + b）
- [x] 构造假数据（`torch.randn` 生成 x，用 `2*x + 1 + noise` 生成 y）
- [x] 定义 `Dataset` 和 `DataLoader`
- [x] 写完整训练循环：`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`
- [x] 打印每个 epoch 的 loss，验证 loss 稳定下降
- [x] 推理测试：用训练好的模型预测新数据
- **为什么做这个**：笔记 03（什么是训练）和 05（完整训练流程）的代码落地。不看任何参考自己写出来，才算真正掌握。
- **预计时间**：30 分钟
- **完成后打勾**：[ ]

### 任务 2：用 PyTorch 做 CIFAR-10 图片分类 ✅ 已完成
- [x] 新建 `00_basics/08_cifar10_classifier.py`
- [x] 用 `torchvision.datasets.CIFAR10` 加载数据（自动下载）
- [x] 定义 `transforms`：`ToTensor()` + `Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))`
- [x] 定义 `DataLoader`（batch_size=64, shuffle=True, num_workers=0）
- [x] 设计一个简单 CNN 模型：`Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → Flatten → Linear → Linear`
- [x] 训练 10 个 epoch，Loss 从 1.44 降到 0.47
- [x] 测试集准确率 68.7%（目标 60%）
- **为什么做这个**：笔记 05（完整训练流程）+ 06（底层vs框架）的实战落地。CNN 是 YOLO backbone 的基础组件。
- **预计时间**：1-1.5 小时
- **完成后打勾**：[ ]

### 任务 3：分析 Baseline 模型指标
- [ ] 运行 `python 01_helmet_detect/03_val.py`
- [ ] 仔细看终端输出的每个类别的 Precision / Recall
- [ ] 打开 `runs/safety_helmet/yolo11n_baseline_v1/confusion_matrix_normalized.png`
- [ ] 记录：哪个类别表现最差？误检多还是漏检多？
- **为什么做这个**：笔记 07（评估指标详解）的实际应用。看懂数字才能做调优。
- **预计时间**：30 分钟
- **完成后打勾**：[ ]

### 任务 4：Bad Case 可视化分析
- [ ] 挑 3-5 张复杂背景的测试图
- [ ] 用 `01_helmet_detect/05_pro_render.py` 做推理渲染
- [ ] 亲眼观察模型在哪些图上翻车
- [ ] 总结 Baseline 模型的 2-3 个硬伤（如：小目标漏检、红衣服误检等）
- **为什么做这个**：笔记 08（数据增强）和 09（调参优化）的前提——必须先知道问题在哪才能针对性优化。
- **预计时间**：30 分钟
- **完成后打勾**：[ ]

---

## 优先级

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | 任务 1：自己写训练循环 | ✅ 完成 |
| P0 | 任务 2：CIFAR-10 图片分类 | ✅ 完成 |
| P1 | 任务 3：Baseline 指标分析 | 待做 |
| P1 | 任务 4：Bad Case 可视化 | 待做 |

**规则**：先完成 P0，再做 P1。如果时间不够，保 P0。

---

## 学习资源（已联网验证）

- PyTorch 官方 CIFAR-10 教程：https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- PyTorch 中文教程（李沐）：https://zh.d2l.ai/
- GitHub CIFAR-10 分类合集：https://github.com/Kedreamix/Pytorch-Image-Classification
- Ultralytics 中文文档：https://docs.ultralytics.com/zh

---

## 参考：笔记 01-09 知识地图

| 笔记 | 核心概念 | 今天哪个任务落地 |
|------|---------|----------------|
| 01 PyTorch 是什么 | Tensor、GPU、动态计算图 | 任务 1 |
| 02 Tensor 入门 | 创建、索引、形状变换、设备迁移 | 任务 1、2 |
| 03 什么是训练 | loss、梯度、学习率、optimizer | 任务 1 |
| 04 什么是模型 | nn.Module、__init__、forward、层 | 任务 1、2 |
| 05 完整训练流程 | Dataset → DataLoader → 训练循环 | 任务 1、2 |
| 06 底层vs框架 | PyTorch 原生 vs Ultralytics 封装 | 任务 2 |
| 07 评估指标详解 | IoU、Precision、Recall、mAP | 任务 3 |
| 08 数据增强 | 翻转、旋转、Mosaic、Mixup | 任务 4 |
| 09 调参优化 | 超参数、lr、batch_size、imgsz | 任务 4 |

# Day 1 知识目录：环境链路与安装日

## 今天的主题

今天只做第 1 周里最关键的启动部分：

`搞懂 CUDA / PyTorch / YOLO 的关系 + 配最小可用环境 + 做基础验证`

## 今天为什么先学这个

因为后面所有训练、推理、微调，都会依赖这条链路：

`你的代码 -> PyTorch -> CUDA -> GPU`

如果今天这条链路不通，后面：

- YOLO 跑不起来
- 训练会变慢
- 显卡用不上
- 你会在环境问题上反复卡住

所以今天的目标不是学一堆新理论，而是把后面 6 周的地基搭好。

## 今天要搞懂的 4 个概念

1. `GPU`
显卡，负责大规模并行计算。

2. `CUDA`
NVIDIA 提供的 GPU 计算平台，让程序能调用显卡算力。

3. `PyTorch`
深度学习框架。你主要写给它看，它再把计算交给 CUDA 和 GPU。

4. `Ultralytics`
YOLO 的现成工具包，让你可以直接训练、验证、推理、导出模型。

## 一句话关系图

`你的 Python 代码 -> PyTorch -> CUDA -> GPU`

如果后面进入 YOLO 实战，就是：

`你的 Python 代码 -> Ultralytics(YOLO) -> PyTorch -> CUDA -> GPU`

## 今天目录里有什么

- [today_plan.md](D:/vision_algo_workspace/vision-bootcamp/notes/2026-05-18_day1/today_plan.md)
- [execution_log_template.md](D:/vision_algo_workspace/vision-bootcamp/notes/2026-05-18_day1/execution_log_template.md)

## 今天结束时至少要达到

- 知道 `CUDA` 是干嘛的
- 知道 `PyTorch` 是干嘛的
- 知道 `Ultralytics` 是干嘛的
- 能清楚说出：
  - `代码不是直接调显卡`
  - `而是 PyTorch 通过 CUDA 调 GPU`
- 有一份安装和验证记录

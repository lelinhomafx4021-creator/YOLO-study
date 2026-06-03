# Day 1 执行记录模板

> 用法：今天边做边填，不要等全部做完再回忆。

## 基本信息

- 日期：2026-05-18
- 开始时间：约 19:50
- 结束时间：约 21:00
- 总耗时：约 1 小时

## 今天目标

- [x] 搞懂 CUDA / PyTorch / Ultralytics 的关系
- [x] 配置最小环境链路
- [x] 完成基础验证

## 我今天理解到的概念

### CUDA

- 我的理解：CUDA 是 NVIDIA 显卡的"方言"。你的 RTX 3050 显卡有几千个小核心，但普通 Python 代码只能用 CPU 的几个核心。CUDA 让 Python 代码能指挥显卡干活——矩阵乘法、卷积这些操作在 GPU 上比 CPU 快几十倍。没有 CUDA，深度学习训练就是蜗牛速度。
- 一句话：**CUDA = 让 Python 能用显卡加速计算的驱动层。**

### PyTorch

- 我的理解：PyTorch 是深度学习的"计算框架"。它做了两件核心的事：(1) 把数据封装成 Tensor（张量，就是多维数组），自动追踪计算过程；(2) 自动求导——你只需要定义前向计算，反向传播的梯度它帮你算。你写的训练循环 `loss.backward()` 就是触发自动求导。
- 一句话：**PyTorch = 帮你做张量计算 + 自动求导的框架，是所有深度学习代码的地基。**

### Ultralytics

- 我的理解：Ultralytics 是 YOLO 系列模型的"一站式工具包"。它在 PyTorch 之上又封装了一层，让你不用自己写网络结构、数据加载、训练循环。你只需要准备好数据集（图片 + 标注文件），然后一行命令 `yolo train` 就能开始训练。它同时支持检测、分割、姿态估计、分类、导出 ONNX 等所有任务。
- 一句话：**Ultralytics = YOLO 的高级封装，让你不用手写深度学习代码就能训练模型。**

### 它们的关系

- 我的理解：**你的 Python 代码 → 调用 Ultralytics → Ultralytics 内部用 PyTorch → PyTorch 调用 CUDA → CUDA 指挥 GPU 计算。** 这是一条从高到低的调用链。你日常操作的是最顶层（Ultralytics），但出问题时需要往下层排查：装错 PyTorch 版本会导致 CUDA 不可用，CUDA 驱动版本不对会导致 GPU 识别不到。
- 一句话：**Ultralytics 是方向盘，PyTorch 是发动机，CUDA 是燃油，GPU 是轮子。**

## 今日实际操作记录

### 1. Python

- 是否已安装：是
- 版本：3.11.9（通过 uv 管理，pin 到 3.11）
- 是否可运行：是
- 备注：用 uv 管理项目环境，`pyproject.toml` 中 `requires-python = ">=3.11"`

### 2. PyTorch

- 是否已安装：是
- 版本：2.7.1+cu118
- 是否成功导入：是
- 备注：首次安装时用了 `--index-url`（一次性参数），后来被 ultralytics 覆盖成 CPU 版。最终通过在 `pyproject.toml` 中配置 `[tool.uv.sources]` 解决，永久绑定 cu118 源。

### 3. CUDA / GPU 可用性

- 是否识别到 GPU：是
- `torch.cuda.is_available()` 结果：True
- 备注：NVIDIA GeForce RTX 3050 Laptop GPU，驱动支持 CUDA 11.8

### 4. Ultralytics

- 是否已安装：是
- 是否成功导入：是
- 备注：ultralytics >= 8.4.51，YOLO11n 模型可正常加载

## 报错记录

### 报错 1：PyTorch 被覆盖成 CPU 版

- 出现在哪一步：安装 ultralytics 之后
- 大概报错内容：`torch.cuda.is_available()` 返回 False，版本变成 `2.12.0+cpu`
- 是否解决：是
- 怎么解决：在 `pyproject.toml` 中配置 `[[tool.uv.index]]` 和 `[tool.uv.sources]`，把 torch/torchvision 永久绑定到 cu118 源。用 `--default-index` 会导致所有包都从 cu118 源下载（opencv 找不到），所以必须用 `explicit = true` + `[tool.uv.sources]` 精确指定。

### 报错 2：uv 环境警告

- 出现在哪一步：每次 uv run 时
- 大概报错内容：`VIRTUAL_ENV=...Python311 does not match the project environment path .venv`
- 是否解决：不影响运行，忽略即可
- 怎么解决：uv 会优先使用项目目录下的 `.venv`，警告是因为系统 PATH 里有另一个 Python

## 今日结果

- [x] Python 正常（3.11.9）
- [x] PyTorch 正常（2.7.1+cu118）
- [x] CUDA 可用（RTX 3050 Laptop GPU）
- [x] Ultralytics 可导入（YOLO11n 可加载）

## 今日总结

### 今天最卡的点

- PyTorch 被 ultralytics 覆盖成 CPU 版。原因是一开始用 `--index-url`（一次性参数），没写入配置。后来又试了 `--default-index`，导致 opencv 从 cu118 源下载失败。最终通过 `pyproject.toml` 里配置 `[tool.uv.sources]` + `explicit = true` 解决。

### 今天学到的最关键一句话

- uv 的 `--index-url` 是一次性的，不会写入配置。需要永久绑定源，必须在 `pyproject.toml` 里配置 `[tool.uv.sources]`。

### 明天准备做什么

- OpenCV 最小脚本：读图、显示、缩放、画框、保存

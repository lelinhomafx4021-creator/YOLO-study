# 💾 解密 YOLO 权重文件：best.pt 的自包含与蜕变

在 YOLO 目标检测的工程应用与算法面试中，**模型权重文件（.pt）**的内部机制和加载原理是一个非常高频的考点。

本笔记系统总结了 `best.pt` 的自包含机制、训练前后的参数与类别字典变化，并提供了一个可以直接查看 `.pt` 内部信息的 Python 小彩蛋。

---

## 🔍 第一部分：`best.pt` 的自包含机制（为什么不需要指定版本）

当我们写下 `model = YOLO("best.pt")` 时，我们并没有告诉 YOLO 任何关于版本、网络结构的信息，但它却能完美加载。

### 1. 什么是“自包含（Self-contained）”？
`.pt` 文件（使用 PyTorch 的 `torch.save` 保存）本质上是一个**打包字典**，而不仅仅是几百万个浮点数。它在内部封装了模型复活所需的一切：

| 内部组件 | 英文术语 | 物理作用 |
| :--- | :--- | :--- |
| **1. 身份元数据** | Metadata | 记录模型诞生版本、属于 YOLO11 还是 YOLOv8、输入尺寸等。 |
| **2. 骨架说明书** | Model Arch | 记录了每一层卷积是如何连接和堆叠的。 |
| **3. 脑细胞数值** | State Dict | 存储着模型中几百万个神经元参数的真实浮点数值。 |
| **4. 类别通讯录** | Names | 存储着类别 ID 与类别名称的字典，例如 `{0: 'helmet', 1: 'vest'}`。 |

**物理过程**：YOLO 库打开 `best.pt` ➡️ 读元数据 ➡️ 按网络骨架在内存中搭好架子 ➡️ 把权重数值填入卷积层 ➡️ 挂载类别字典 ➡️ 模型加载完成。

---

## 🔄 第二部分：训练前（yolo11n.pt）与 训练后（best.pt）的蜕变对比

在微调过程中，权重文件内部的数据发生了**颠覆性的变化**：

### 1. 类别标签（Names）的“格式化与重写”
*   **yolo11n.pt (原版)**：拥有一张包含 80 个类别的通讯录：
    `{0: 'person', 1: 'bicycle', ..., 16: 'dog', ..., 79: 'toothbrush'}`。
*   **best.pt (微调后)**：由于我们在 `custom_data.yaml` 里只规定了 2 个类别，YOLO 在保存时，**直接把那 80 个类别格式化并彻底抹去**，重写为：
    `{0: 'helmet', 1: 'vest'}`。
*   **结果**：微调后的模型再也不认识猫狗，它只认安全帽和反光衣。

### 2. 权重参数（State Dict）的“特征重塑”
*   **yolo11n.pt (原版)**：参数是官方在 COCO 数据集上训练好的通用特征提取器，能够识别猫眼、狗耳朵、汽车轮子等。
*   **best.pt (微调后)**：在 30 轮微调的“反向传播挨揍”中，这几百万个权重数值被强行改写：
    *   原先用来提炼“猫耳朵”的卷积通道，被改塑为提炼“安全帽的半圆弧边缘”。
    *   原先用来提炼“汽车轮胎”的卷积通道，被改塑为提炼“反光衣上的灰色反光条”。

---

## 🧪 第三部分：Python 实证实战（用代码扒开 .pt 文件的衣服）

### 3.1 基础：查看类别映射

你可以不用盲信理论，在你的 VS Code 里直接新建一个 `check_pt.py`（或在终端交互式 Python 里运行），手打这 4 行代码，就能直接打印出我们训练好的权重文件内部的**类别字典**：

```python
import torch

# 1. 直接用 PyTorch 加载你训练好的权重文件
# map_location="cpu" = 强制加载到 CPU（即使你有 GPU），防止 GPU 不够
ckpt = torch.load("runs/safety_helmet/yolo11n_baseline_v1/weights/best.pt",
                   map_location="cpu")

# 2. 打印出里面的标签映射表（在 model 键下）
print("📊 权重里保存的类别映射为:", ckpt['model'].names)
```

**运行输出预期**：
> `📊 权重里保存的类别映射为: {0: 'helmet', 1: 'vest'}`
> *（原版的 person, dog, car 已经全部被自动删除并重写！）*

### 3.2 进阶：对比训练前后的权重

```python
import torch

# ⚠️ 预训练和微调后的模型必须是同一个架构！
# 你的 best.pt 是从 yolo11n.pt 开始训的 → 两个文件内部结构完全对应

# 加载原始预训练权重
pretrain = torch.load("yolo11n.pt", map_location="cpu")

# 加载你微调后的权重
finetuned = torch.load("runs/safety_helmet/yolo11n_baseline_v1/weights/best.pt",
                       map_location="cpu")
# 不能拿 yolo8n.pt 和 yolo11n 微调的 best.pt 对比——结构根本不一样

# 对比类别
print("预训练模型类别:", pretrain['model'].names)  # {0: 'person', ..., 79: 'toothbrush'}
print("微调后模型类别:", finetuned['model'].names)  # {0: 'helmet', 1: 'no-helmet', 2: 'safety-vest'}

# 对比参数量（应该一样，因为结构没变）
pretrain_params = sum(p.numel() for p in pretrain['model'].parameters())
finetune_params = sum(p.numel() for p in finetuned['model'].parameters())
print(f"预训练参数量: {pretrain_params:,}")
print(f"微调后参数量: {finetune_params:,}")
print(f"参数量相同: {pretrain_params == finetune_params}")  # True
```

### 3.3 深入：看看 .pt 文件里到底有什么

```python
import torch

ckpt = torch.load("runs/safety_helmet/yolo11n_baseline_v1/weights/best.pt",
                   map_location="cpu")

# ckpt 是一个字典（dict），看看有哪些钥匙
print("顶层键:", list(ckpt.keys()))
# 输出: ['model', 'epoch', 'optimizer', 'train_args', ...]

# 看看 epoch
if 'epoch' in ckpt:
    print(f"这个权重是训练了 {ckpt['epoch']} 轮后保存的")

# 看看训练参数（训练时传了什么 config）
if 'train_args' in ckpt:
    args = ckpt['train_args']
    print(f"训练时的 imgsz: {args.get('imgsz', '未知')}")
    print(f"训练时的 batch: {args.get('batch', '未知')}")

# state_dict：所有参数的真实数值（几百万个浮点数）
state = ckpt['model'].state_dict()
print(f"\n共 {len(state)} 个参数层/键")
print(f"前 5 个层的名字:")
for i, key in enumerate(list(state.keys())[:5]):
    print(f"  {key} — 形状 {state[key].shape}")

# 看某一个参数的实际数值（前几个）
first_weights = list(state.values())[0]
print(f"\n第一层权重的前 5 个值: {first_weights.flatten()[:5].tolist()}")
```

### 3.4 best.pt vs last.pt 的区别

```python
import torch

best = torch.load("runs/safety_helmet/yolo11n_baseline_v1/weights/best.pt",
                   map_location="cpu")
last = torch.load("runs/safety_helmet/yolo11n_baseline_v1/weights/last.pt",
                   map_location="cpu")

# epoch 不同
print(f"best.pt 保存于 epoch: {best.get('epoch', '未知')}")
print(f"last.pt 保存于 epoch: {last.get('epoch', '未知')}")

# 参数数值不同（best 是 mAP 最高那轮，last 是最后那轮）
best_w = best['model'].state_dict()['model.0.conv.weight']
last_w = last['model'].state_dict()['model.0.conv.weight']
print(f"best.pt 和 last.pt 第一层是否完全一样: {torch.equal(best_w, last_w)}")
# 输出通常是 False，因为两个 epoch 的参数值不同
```

### 3.5 完整的权重检查脚本

把上面整合成一个脚本，保存为 `00_basics/check_pt.py`：

```python
# -*- coding: utf-8 -*-
"""check_pt.py — 查看 YOLO 权重文件内部"""
import torch
from pathlib import Path

pt_path = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")

ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

print("=" * 50)
print(f"文件: {pt_path.name}")
print("=" * 50)

# 类别
names = ckpt['model'].names
print(f"类别数: {len(names)}")
print(f"类别映射: {names}")

# 基本信息
if 'epoch' in ckpt:
    print(f"保存于 epoch: {ckpt['epoch']}")

# 参数统计
total = sum(p.numel() for p in ckpt['model'].parameters())
print(f"总参数量: {total:,}")

# 训练配置
if 'train_args' in ckpt:
    print(f"训练参数: imgsz={ckpt['train_args'].get('imgsz')}, "
          f"batch={ckpt['train_args'].get('batch')}, "
          f"epochs={ckpt['train_args'].get('epochs')}")
```

---

## 复习速答

- `best.pt`：训练好的权重文件，通常包含模型和相关信息。
- `state_dict`：模型参数字典。
- `torch.save`：把参数或 checkpoint 存成文件。
- `torch.load`：把保存的权重读回来。
- `YOLO("best.pt")`：直接加载整套权重和结构信息。

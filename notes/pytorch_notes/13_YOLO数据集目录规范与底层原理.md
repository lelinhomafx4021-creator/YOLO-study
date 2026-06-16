# 📁 YOLO 数据集目录规范与底层匹配原理（必学规范）

目标检测框架（如 YOLOv8 / YOLO11）对数据集的存放目录有着**极其严格的专门规定**。一旦放错一个字母或者目录层级不对，就会直接触发 `RuntimeError: No labels found` 报错。

本讲义为你详解这个目录规范的**底层匹配逻辑**与**科学设计原因**。

---

## 🔍 第一部分：标准的 YOLO 数据集目录树

不管是你自己的项目，还是工业界、学术界的开源项目，大家都严格遵守以下目录树：

```text
datasets/                      <-- 统一的数据集根目录
  └── safety_helmet/           <-- 你的数据集文件夹（对应 yaml 里的 path）
        ├── images/            <-- 必须叫 images，存放图片
        │     ├── train/       <-- 训练集图片 (如 001.jpg, 002.jpg)
        │     └── val/         <-- 验证集图片
        └── labels/            <-- 必须叫 labels，存放标注文本
              ├── train/       <-- 训练集标签 (必须与图片同名，如 001.txt, 002.txt)
              └── val/         <-- 验证集标签
```

---

## 🚀 第二部分：YOLO 底层是怎么找标签的？（核心原理）

你可能会好奇：“在代码里，我们只告诉了 YOLO 图片的路径，我们并没有告诉它标签文件在哪个文件夹啊？它是怎么自动把图片和标签对上的？”

### 💡 底层的“自动路径替换算法”
YOLO 底层的 PyTorch 数据读取器（DataLoader）在加载图片时，会使用一套死记硬背的**字符串替换逻辑**：

1. **第一步（读图）**：它先去 `images/train/` 文件夹下读取一张图片，比如路径是：
   `D:/vision_algo_workspace/vision-bootcamp/datasets/safety_helmet/images/train/worker_001.jpg`
2. **第二步（路径替换）**：它在后台，自动将路径中的 `/images/` 替换成 `/labels/`。路径变成了：
   `D:/vision_algo_workspace/vision-bootcamp/datasets/safety_helmet/labels/train/worker_001.jpg`
3. **第三步（后缀替换）**：它自动把图片后缀（如 `.jpg`, `.png`, `.jpeg`）替换为 `.txt`。路径变成了：
   `D:/vision_algo_workspace/vision-bootcamp/datasets/safety_helmet/labels/train/worker_001.txt`
4. **第四步（加载标签）**：它去尝试读取这个自动算出来的 `.txt` 文件。如果找到了，就成功配对；如果找不到，就报错崩溃。

### 🚨 避坑指南（面试常识）：
* **坑一**：如果你把存放图片的文件夹改名叫 `pics`，把存放标签的改名叫 `txts`，YOLO 底层的路径替换就完全失效了，会直接报错找不到标签。所以，**必须死死记住叫 `images` 和 `labels`**。
* **坑二**：图片和 `.txt` 标注文件**文件名必须百分之百大小写一致**。如果图片叫 `Worker_01.jpg`，标签叫 `worker_01.txt`（一个小写一个大写），在 Windows 下可能凑合能用，但在 Linux 系统（大部分服务器部署环境）上会直接报错找不到文件，因为 Linux 严格区分大小写！

---

## 📝 第四部分：data.yaml 是怎么把这个目录结构告诉 YOLO 的

```yaml
# 你的 custom_data.yaml（在 01_helmet_detect/ 下）
path: d:\vision_algo_workspace\vision-bootcamp\datasets\safety_helmet
train: images/train    # ← 相对于 path 的路径
val: images/val        # ← YOLO 自动把 /images/ 换成 /labels/ 找标注

names:
  0: helmet
  1: no-helmet
  2: safety-vest
```

### 底层逻辑（分步骤执行）：

```python
# YOLO 内部伪代码 — 它怎么根据 data.yaml 找到标注文件
img_path = r"D:\...\datasets\safety_helmet\images\train\worker_001.jpg"

# Step 1: 把 path + train 拼接 → images/train 目录
# Step 2: 扫目录找到所有 .jpg/.png 文件
# Step 3: 对每个图片文件：
#    a. 把字符串 "/images/" 替换成 "/labels/"      ← 核心逻辑
#    b. 把后缀 .jpg 替换成 .txt
#    c. 打开这个 .txt，读取标注框
# Step 4: 如果 .txt 不存在 → 当做"负样本"（背景图，无标注）

# 结果：
# images/train/worker_001.jpg → labels/train/worker_001.txt ✓
# images/train/bg_empty.jpg   → labels/train/bg_empty.txt ✗ → 负样本
```

### 你的实际目录看一眼

```python
# 在终端跑这个，看你的数据集结构
import os

data_root = r"d:\vision_algo_workspace\vision-bootcamp\datasets\safety_helmet"
for root, dirs, files in os.walk(data_root):
    level = root.replace(data_root, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:
        continue
    for f in files[:3]:  # 只显示前 3 个文件
        print(f"{indent}  {f}")
    if len(files) > 3:
        print(f"{indent}  ... (共 {len(files)} 个文件)")
```

### 一张图看懂路径替换

```
训练时，YOLO 自动做这个转换：

读图片：    datasets/safety_helmet/images/train/worker_001.jpg
                                    ↓ 自动替换
读标注：    datasets/safety_helmet/labels/train/worker_001.txt

验证时同理：
读图片：    datasets/safety_helmet/images/val/worker_050.jpg
                                    ↓ 自动替换
读标注：    datasets/safety_helmet/labels/val/worker_050.txt
```

### 标签文件里面长什么样

```txt
# labels/train/worker_001.txt — 一张图里可能有多行（多个目标）
0 0.532 0.418 0.087 0.095    ← helmet, 中心(53.2%,41.8%), 宽8.7%, 高9.5%
1 0.345 0.612 0.102 0.133    ← no-helmet
2 0.678 0.523 0.156 0.234    ← safety-vest
#   ↑       ↑       ↑     ↑
# 类别ID  中心x    中心y  宽    高     （全部归一化到 0~1）
```

---

## 🔬 第三部分：为什么一定要分 `train` 和 `val`？

在机器学习中，这是最基础也是最重要的“三权分立”原则：

1. **`train`（训练集） ➡️ 【模拟考试题】**：
   * 占总数据的 $80\% \sim 90\%$。
   * 模型在训练过程中天天看它，根据它和答案（label）计算 Loss，并用梯度下降去修改自己的权重参数。
2. **`val`（验证集） ➡️ 【期末考试卷】**：
   * 占总数据的 $10\% \sim 20\%$。
   * 模型在训练时**绝对不能看验证集的答案**。每一轮训练结束，模型以“闭卷考试”的形式去预测验证集的图片，算出 Precision、Recall 和 mAP。
   * **作用**：用来监控模型有没有学傻（过拟合）。如果验证集考试成绩很差，我们就要手动调超参数。
3. **`test`（测试集，非必需） ➡️ 【社会实践大考】**：
   * 模型彻底训练好、参数固定后，拿完全没参与过训练的测试集做一次最终综合评估。

---

## 复习速答

- `images/labels`：YOLO 默认配对目录。
- `同名文件`：图片和标签必须同名。
- `train`：训练用。
- `val`：验证用。
- `data.yaml`：告诉 YOLO 数据在哪、类名是什么。

# Vision Transformer (ViT)：用 Transformer 做图像分类

## 这一节在学习顺序里的位置

前面你已经知道：

- Transformer 的自注意力机制：每个位置都能看到所有位置
- Encoder Block 的完整结构：自注意力 + 残差 + LayerNorm + FFN

但 Transformer 处理的是**序列**（一维的词序列）。图片是**二维的像素网格**。

怎么把图片变成 Transformer 能吃的输入？

这就是 ViT（Vision Transformer，2020 年 Google）要解决的问题。

---

## 先说最核心的一句话

**ViT 把图片切成小块（patch），每个块当成一个"词"，扔进标准 Transformer Encoder。**

如果只记一句，就记这句。

---

## 1. 问题：Transformer 吃的是序列，图片不是序列

```
NLP 的输入: "我 今天 去 北京" → 4 个词 → 4 个向量 → 序列 ✓

图片的输入: 224×224 像素 → 50176 个像素 → 怎么变成序列？
```

如果把每个像素当一个"词"，那序列长度 = 50176。自注意力的计算量 = 序列长度的平方 = 50176² ≈ 25 亿。算不动。

**ViT 的解决办法：不按像素，按"块"。**

---

## 2. Patch Embedding：把图片切成块

一张 224×224 的图，切成 16×16 的小块：

```
224 ÷ 16 = 14
→ 横向 14 块，纵向 14 块
→ 总共 14 × 14 = 196 个块
```

每个块是 16×16×3 = 768 个像素值（RGB 三通道）。

把每个块展平成一个 768 维的向量 → 196 个向量 → 序列！

```
图片 224×224
  ↓ 切成 16×16 的块
196 个块 (每个 16×16×3)
  ↓ 每个块展平
196 个 768 维向量
  ↓ 线性投影
196 个 768 维向量 (Patch Embedding)
```

**Patch Embedding 的本质：把二维图片切成块，展平，再投影到 Transformer 的维度空间。** [看 Patch Embedding 可视化](vit_patch_embedding_visual.html)

---

## 3. CLS Token：分类用的特殊标记

Transformer 的输入不是只有 196 个 patch 向量。ViT 在最前面加了一个特殊的向量：**CLS Token**。

```
输入序列: [CLS] [patch1] [patch2] ... [patch196]
           ↑
         一个可学习的向量，不对应任何图片区域
```

**CLS Token 不包含图片信息。它的作用是：经过 12 层 Transformer 后，它的输出就是整张图的分类结果。**

为什么需要 CLS Token？

自注意力让所有位置都能看到所有位置。经过多层后，CLS Token 已经"看过了"所有 patch 的信息。它就像一个"汇总员"，把所有 patch 的信息整合到自己身上。

---

## 4. 位置编码

上一节说过：Transformer 不关心顺序。"patch1 patch2 patch3" 和 "patch3 patch1 patch2" 算出来一样。

ViT 用**可学习的位置编码**：

```
输入 = Patch Embedding + 位置编码

位置编码: 197 个 768 维向量（196 个 patch + 1 个 CLS）
         这些向量是训练时学出来的，不是固定的公式
```

**位置编码告诉 Transformer 每个 patch 在图片里的位置（左上、右下、中间...）。**

---

## 5. 送入 Transformer Encoder

```
[CLS, patch1, patch2, ..., patch196]  ← 197 个 768 维向量
  ↓
Transformer Encoder × 12 层
  (每层: 自注意力 → 残差 → LayerNorm → FFN → 残差 → LayerNorm)
  ↓
取 CLS Token 的输出 → 768 维向量
  ↓
Linear → 类别概率（比如 ImageNet 的 1000 类）
```

**就这么简单。把 CNN 里的卷积层全部换成了 Transformer Encoder。**

---

## 6. ViT vs CNN

```
CNN (比如 ResNet):
  3×3 卷积 → 3×3 卷积 → 3×3 卷积 → ...
  每层只看 3×3 区域，逐步扩大感受野
  天然的平移不变性
  小数据集效果好

ViT:
  切 patch → Transformer Encoder × 12
  每层直接看所有 196 个 patch
  没有平移不变性（靠数据量学出来）
  大数据集效果更好
```

**关键结论：ViT 在大数据集（ImageNet-21K、JFT-300M）上超过 CNN，但在小数据集上不如 CNN。**

原因：CNN 有归纳偏置（局部性、平移不变性），这些先验知识在数据少时很有用。ViT 没有这些先验，需要更多数据来学。

---

## 7. DeiT：数据高效的 ViT

Facebook 在 2021 年提出了 DeiT（Data-efficient Image Transformers），证明用更好的训练策略，ViT 在 ImageNet-1K（130 万张图）上也能超过 CNN。

关键技巧：
- 知识蒸馏（用 CNN 教 ViT）
- 更强的数据增强
- 正则化（DropPath、Label Smoothing）

**现在 ViT + 足够的数据增强 = 小数据集也能用。**

---

## 8. 和 YOLO 的关系

YOLOv8/v11 的 backbone 还是 CNN。但一些新的检测模型（比如 RT-DETR）已经用 ViT 做 backbone 了。

趋势是：**CNN 提取低层特征（边缘、纹理），Transformer 提取高层特征（物体、语义）。** 混合架构是现在的主流。

---

## 9. 这一节最该记住的 5 句话

1. **ViT 把图片切成 patch，每个 patch 当成一个"词"输入 Transformer**
2. **Patch Embedding = 切块 → 展平 → 线性投影**
3. **CLS Token 是一个特殊向量，经过 Transformer 后的输出就是分类结果**
4. **位置编码告诉 Transformer 每个 patch 在图片里的位置**
5. **ViT 大数据集超 CNN，小数据集不如 CNN（靠数据增强弥补）**

---

## 10. 学完这一节后你应该能回答的问题

**Q1: ViT 怎么把图片变成 Transformer 能处理的输入**

A: 切成 16×16 的 patch → 每个 patch 展平成 768 维向量 → 线性投影 → 196 个向量当 196 个"词"输入 Transformer。

---

**Q2: Patch Embedding 做了什么**

A: 把二维图片切成块，展平，再投影到 Transformer 的维度空间。本质是把图片变成序列。

---

**Q3: CLS Token 是什么，为什么需要它**

A: 一个可学习的特殊向量，不对应任何图片区域。经过 12 层 Transformer 后，它的输出就是整张图的分类结果。它像一个"汇总员"，把所有 patch 的信息整合到自己身上。

---

**Q4: 为什么 ViT 需要位置编码**

A: 因为 Transformer 不关心顺序。位置编码告诉每个 patch 在图片里的位置（左上、右下、中间...）。

---

**Q5: ViT 和 CNN 的核心区别是什么**

A: CNN 用卷积核逐步扩大感受野，有归纳偏置（局部性、平移不变性）。ViT 用自注意力一步看全局，没有归纳偏置。ViT 大数据集超 CNN，小数据集不如 CNN。

---

**Q6: 为什么 ViT 在小数据集上不如 CNN**

A: CNN 有归纳偏置（局部性、平移不变性），这些先验知识在数据少时很有用。ViT 没有这些先验，需要更多数据来学。

---

**Q7: DeiT 怎么解决小数据集的问题**

A: 知识蒸馏（用 CNN 教 ViT）+ 更强的数据增强 + 正则化。证明用更好的训练策略，ViT 在 ImageNet-1K（130 万张图）上也能超过 CNN。

---

## 复习速答

- `ViT`：把图片切成 patch 当序列处理。
- `Patch Embedding`：把 patch 投影到 Transformer 维度。
- `CLS Token`：汇总整张图信息。
- `位置编码`：告诉模型 patch 的位置。
- `小数据集`：CNN 通常更稳，ViT 往往需要更多数据。

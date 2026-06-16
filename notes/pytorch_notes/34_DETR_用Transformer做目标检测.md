# DETR：用 Transformer 做目标检测

## 这一节在学习顺序里的位置

前面你已经知道：

- YOLO 做检测的方式：CNN backbone → 检测头 → NMS 后处理
- ViT 做分类的方式：切 patch → Transformer Encoder → CLS Token → 类别

DETR（2020，Facebook）把 Transformer 用到了**目标检测**上。

它的最大贡献：**去掉了 NMS。**

---

## 先说最核心的一句话

**DETR 把目标检测看成一个"集合预测"问题：直接输出 N 个框，每个框带类别，不需要 NMS 后处理。**

---

## 1. YOLO 的检测流程有什么问题

```
YOLO:
  图片 → CNN → 检测头 → 每个格子预测 N 个框 → 几千个候选框
  → 置信度过滤 → NMS 去重 → 最终结果
```

问题：
- **NMS 是手工设计的后处理**，不是网络学出来的
- NMS 的 IoU 阈值需要调参
- 密集物体重叠时 NMS 容易误删
- 整个流程不是端到端的

**DETR 的想法：能不能让网络直接输出最终结果，不要 NMS？**

---

## 2. DETR 的整体结构

```
输入图片
  ↓
CNN Backbone (ResNet) → 提取特征图
  ↓
Transformer Encoder → 特征进一步处理
  ↓
Transformer Decoder → 用"Object Query"去查询特征
  ↓
输出 N 个预测 (框 + 类别)
  ↓
不需要 NMS！直接就是最终结果
```

---

## 3. Object Query：DETR 的核心创新

传统检测器（YOLO）：**"图上每个位置都预测一个框"** → 几千个候选 → NMS 去重。

DETR：**"我只有 N 个查询槽位，每个槽位负责找一个物体"** → N 个结果 → 不需要去重。 [看 Object Query 可视化](detr_object_query_visual.html)

```
Object Query = N 个可学习的向量（比如 N=100）

每个 Query 负责"找一个物体":
  Query 1 → 找到 → 输出 (框1, 类别1)
  Query 2 → 找到 → 输出 (框2, 类别2)
  ...
  Query 50 → 没找到 → 输出 "no object"
  ...
  Query 100 → 没找到 → 输出 "no object"
```

**Object Query 就是"我要找物体"的提问。每个 Query 通过注意力机制去图片特征里找它负责的那个物体。**

---

## 4. Decoder 怎么用 Query 找物体

Transformer Decoder 做两件事：

### 4.1 自注意力：Query 之间互相看

```
Query 1: "我要找左边的物体"
Query 2: "我也在找左边的物体"
→ 自注意力后: 两个 Query 知道对方也在找 → 避免重复
```

**Query 之间的自注意力让不同的 Query 负责不同的物体，不会都去抢同一个。**

### 4.2 交叉注意力：Query 去看图片特征

```
Query 1 的 Q: "我在找一个红色的物体"
图片特征的 K/V: 这里有红色、那里有蓝色...

→ Query 1 的注意力集中在红色区域 → 找到了！
```

**交叉注意力就是 Query（Q 来自 Decoder）和图片特征（K/V 来自 Encoder）之间的注意力。Query 去图片里"搜索"它要找的物体。**

---

## 5. 二部图匹配：训练时怎么对应

DETR 输出 N 个预测，但一张图可能只有 3 个物体。怎么算 Loss？

**问题：哪个 Query 应该负责哪个真实物体？**

DETR 用**匈牙利算法**做二部图匹配：

```
预测: [Query1, Query2, ..., Query100]
真实: [物体A, 物体B, 物体C]

匈牙利算法: 找一种配对方式，使得总匹配成本最低
  Query2 → 物体A
  Query7 → 物体B
  Query15 → 物体C
  其他 Query → "no object"
```

**配对完成后，只对配对上的 Query 计算 Loss。没配上的 Query 被训练成输出 "no object"。**

---

## 6. DETR vs YOLO

```
YOLO:
  CNN → 几千个候选框 → NMS → 最终结果
  快，但 NMS 是手工设计，密集场景有问题

DETR:
  CNN + Transformer → N 个直接预测 → 最终结果
  端到端，没有 NMS，但训练慢（需要 500 epoch）
```

**DETR 的优势：端到端、无 NMS、全局推理。**
**DETR 的劣势：训练收敛慢（500 epoch vs YOLO 的 300 epoch）、小物体检测差。**

---

## 7. Deformable DETR：改进版

原始 DETR 的问题是：全局注意力计算量大，小物体容易被忽略。

Deformable DETR（2021）的改进：

```
原始 DETR:  每个 Query 看图片的所有位置 → 计算量大
Deformable: 每个 Query 只看几个关键点 → 计算量小、收敛快
```

**Deformable DETR 让 DETR 真正能用：训练 epoch 少了 10 倍，小物体检测也好了。**

---

## 8. RT-DETR：实时 DETR

百度在 2023 年提出了 RT-DETR，让 DETR 达到了实时速度。

```
YOLO:     ~30 FPS  (RTX 3050)
RT-DETR:  ~30 FPS  (差不多的速度)
```

**RT-DETR 证明了 Transformer 检测器也能实时。** 现在 DETR 系列和 YOLO 系列在性能上已经接近。

---

## 9. 这一节最该记住的 5 句话

1. **DETR 把检测看成"集合预测"，直接输出 N 个结果，不需要 NMS**
2. **Object Query = N 个可学习的"我要找物体"的提问**
3. **Query 之间的自注意力避免重复，Query 和图片特征的交叉注意力找到物体**
4. **训练时用匈牙利算法做 Query 和真实物体的最优配对**
5. **DETR 端到端无 NMS，但训练慢；Deformable DETR 和 RT-DETR 解决了这个问题**

---

## 10. 学完这一节后你应该能回答的问题

**Q1: DETR 和 YOLO 的核心区别是什么**

A: YOLO 输出几千个候选框 → NMS 去重 → 最终结果。DETR 直接输出 N 个预测 → 不需要 NMS。DETR 是端到端的，YOLO 不是。

---

**Q2: Object Query 是什么，为什么需要它**
"我" 的向量 = [1.0, 0.5, 0.2, 0.8]
A: N 个可学习的向量，每个 Query 负责"找一个物体"。传统检测器每个位置都预测一个框（几千个），DETR 只有 N 个 Query（比如 100 个），每个 Query 通过注意力去图片里找它负责的物体。

---

**Q3: Decoder 的自注意力和交叉注意力分别做什么**

A: 自注意力 = Query 之间互相看，避免重复（两个 Query 不要去抢同一个物体）。交叉注意力 = Query 去看图片特征，找到它负责的物体。

---

**Q4: 匈牙利算法在 DETR 训练里做什么**

A: 给 N 个预测和 M 个真实物体做最优配对。配对上的计算 Loss，没配上的训练成输出"no object"。

---

**Q5: DETR 的优势和劣势各是什么**

A: 优势 = 端到端无 NMS、全局推理、密集场景更好。劣势 = 训练收敛慢（500 epoch vs YOLO 的 300 epoch）、小物体检测差。

---

**Q6: Deformable DETR 改进了什么**

A: 每个 Query 只看几个关键点（而不是全图），计算量小了 10 倍，收敛快了 10 倍，小物体检测也好了。

---

**Q7: RT-DETR 证明了什么**

A: Transformer 检测器也能达到实时速度（~30 FPS），和 YOLO 差不多。DETR 系列和 YOLO 系列在性能上已经接近。

---

## 复习速答

- `DETR`：端到端目标检测，不靠 NMS。
- `Object Query`：每个 Query 负责找一个物体。
- `匈牙利算法`：训练时做最优匹配。
- `Deformable DETR`：只看少量关键点，收敛更快。
- `RT-DETR`：把 Transformer 检测做到了实时。

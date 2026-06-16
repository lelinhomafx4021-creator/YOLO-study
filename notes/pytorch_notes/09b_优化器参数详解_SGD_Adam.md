# 优化器参数详解：SGD vs Adam

## 一句话

SGD 手动挡，lr 设错就崩。Adam 自动挡，lr=0.001 全世界通用。weight_decay 是刹车，两个都能加。

---

## 1. SGD 参数

```python
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
```

| 参数 | 中文 | 作用 | 不设会怎样 | 典型值 |
|------|------|------|-----------|--------|
| `lr` | 学习率 | 每步走多大 | 不动/发散 | **0.01**（比 Adam 大 10 倍） |
| `momentum` | 动量 | 保留 90% 历史速度，平滑震荡 | 走得像锯齿，慢 | **0.9**（几十年没改过） |
| `weight_decay` | 权重衰减 | 每次往 0 拉一把，防过拟合 | 参数可能过大，过拟合 | 1e-4 ~ 5e-4 |

**SGD 的更新公式：**

```
v = momentum × v + gradient           ← 累积速度
θ = θ — lr × v — lr × weight_decay × θ  ← 更新参数
```

**核心认知：SGD 对所有参数用同一个 lr。** 梯度大的参数走太快会震荡，梯度小的走太慢学不到。lr 设错 = 全盘崩。

---

## 2. Adam 参数

```python
torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
```

| 参数 | 中文 | 作用 | 不设会怎样 | 典型值 |
|------|------|------|-----------|--------|
| `lr` | 学习率 | 全局缩放系数 | 基本都能跑 | **0.001** |
| `betas[0]` | 一阶动量系数 | 短期方向记忆（类似 SGD momentum） | 不稳定 | **0.9**（永远不改） |
| `betas[1]` | 二阶动量系数 | 长期速度统计（自适应学习率的关键） | 自适应失效 | **0.999**（永远不改） |
| `eps` | 防除零 | 分母加个极小值 | 除零报错 | **1e-8**（永远不改） |
| `weight_decay` | 权重衰减 | 同 SGD | 可能过拟合 | 1e-4 ~ 1e-3 |

**Adam 的更新公式（简化理解）：**

```
m = 0.9×m + 0.1×gradient          ← 跟踪梯度方向
v = 0.999×v + 0.001×gradient²     ← 跟踪梯度大小
θ = θ — lr × m / (√v + 1e-8)      ← 步长 = lr / √v（大梯度→小步，小梯度→大步）
```

**核心认知：Adam 每个参数有自己的"专属学习率"。** 梯度一直大的参数→√v大→步长自动缩小。梯度一直小的→√v小→步长自动放大。所以你 lr 写大一倍也没事。

---

## 3. weight_decay 详解

**每次更新时，额外把参数往 0 拉一把。**

```
没有 weight_decay： weight = weight - lr×梯度
有 weight_decay：   weight = weight - lr×梯度 - lr×wd×weight
                                                   └────┬────┘
                                                  往 0 拉的力
```

| weight_decay 值 | 效果 |
|-----------------|------|
| 0 | 不限制，参数可以任意大 |
| 1e-4 | 轻轻拽，防过拟合，最常用 |
| 1e-3 | 中等力度，明显过拟合时用 |
| 1e-2 | 太猛，模型学不会 |

**你的实验证据：weight_decay=0 → 69.5%，weight_decay=1e-4 → 70.2%**，加了反而更高——说明模型之前有轻微过拟合。

### Adam 要用 AdamW

普通 Adam 把 weight_decay 和自适应学习率混在一起，效果打折。**AdamW 把 weight_decay 从自适应计算里拆出来**，跟 SGD 的 weight_decay 行为一致：

```python
# 推荐用这个，不是 Adam
torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

---

## 4. 如何选择（实战流程图）

```
你要不要快速出结果？
├── 是 → Adam(lr=0.001)，什么都不用改，先跑起来
│       └── 过拟合？ → AdamW(lr=0.001, weight_decay=1e-4)
│
└── 否，追求极致精度 → SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)
                        └── 还要调 lr schedule（学习率衰减）
```

---

## 5. 常见问题诊断

| 现象 | 可能原因 | 解决 |
|------|---------|------|
| loss 震荡剧烈 | lr 太大 | 降 lr（SGD 降 10 倍, Adam 降 3 倍） |
| loss 几乎不降 | lr 太小 | 加 lr（SGD 加 10 倍, Adam 加 3 倍） |
| 训练到一半 loss=NaN | lr 太大导致溢出 | 降 lr，加梯度裁剪 |
| 训练集好，验证集差 | 过拟合 | 加 weight_decay、数据增强、减少 epochs |
| 训练集验证集都不好 | 欠拟合 | 加 epochs、换更强模型、降低 weight_decay |
| SGD 用 lr=0.001 训不动 | SGD 需要的 lr 比 Adam 大 10 倍 | 改成 lr=0.01 |

---

## 6. 最该记住的

1. **Adam lr=0.001 是全世界默认值**，90% 场景不用改
2. **SGD lr 比 Adam 大 10 倍**（0.01 vs 0.001）
3. **momentum=0.9 永远带着**，SGD 不带 momentum 基本是废的
4. **betas 永远不改**，论文作者定的 (0.9, 0.999) 就是最优解
5. **weight_decay=1e-4 是安全起点**，过拟合再加大到 1e-3
6. **用 AdamW 而不是 Adam**（weight_decay 效果更好）

---

## 7. 可视化演示

打开 `00_basics/optimizer_visual.html`，拖动滑块看：
- SGD 不同 lr 的路径差异
- Adam lr×10 仍然稳定
- momentum 平滑震荡
- weight_decay 拉近终点

---

## 复习速答

- `SGD`：统一学习率更新所有参数。
- `Adam`：给每个参数自适应步长。
- `momentum`：保留历史方向，减少震荡。
- `weight_decay`：把参数往 0 拉，防过拟合。
- `AdamW`：更推荐的带权重衰减版本。

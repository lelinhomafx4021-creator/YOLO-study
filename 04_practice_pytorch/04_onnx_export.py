# -*- coding: utf-8 -*-
"""
关卡 4 PyTorch版: torch.onnx.export → ONNX Runtime 推理

═══════════════════════════════════════════════════════════════════════════════
这一关在做什么
═══════════════════════════════════════════════════════════════════════════════

  1. 搭一个 TinyCNN 模型（随机初始化，不需要训练）
  2. 用 torch.onnx.export 导出成 .onnx 文件
  3. 同一个输入 dummp，分别用 PyTorch 和 ONNX Runtime 推理
  4. 对比两次输出 → 完全一致 → 证明 ONNX 完美还原了 PyTorch 的计算

  核心验证: PyTorch 模型 → ONNX 格式 → 不依赖 torch 也能拿到完全一样的结果。

═══════════════════════════════════════════════════════════════════════════════
ONNX 和 .pth 的区别
═══════════════════════════════════════════════════════════════════════════════

  .pth:  只存参数（权重值），不存模型结构
         → 加载方必须用完全相同的 Python 类重建模型 → 绑死 PyTorch

  .onnx: 存计算图 + 参数（权重嵌在图里）
         → 加载方不需要 Python 类定义，任何支持 ONNX 的框架都能跑
         → 解耦: 训练用 PyTorch，部署用 ONNX Runtime / TensorRT / OpenVINO...

═══════════════════════════════════════════════════════════════════════════════
和 YOLO 关卡 ONNX 的关系
═══════════════════════════════════════════════════════════════════════════════

  YOLO 版:  model.export(format="onnx") → 一行搞定，细节全封装
  PyTorch 版: torch.onnx.export(model, dummy, path, ...) → 每个参数自己写
             → 知道 YOLO 的 model.export() 底层到底做了什么
"""

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort   # pip install onnxruntime-gpu  (或 onnxruntime)
import os


# ═════════════════════════════════════════════════════════════════════════════
# 1. 建一个 SimpleCNN
# ═════════════════════════════════════════════════════════════════════════════
# 和关卡 1/2 完全一样的结构，不需要训练 — ONNX 导出只管结构 + 参数，不要求训练过
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16*8*8, 10)
        )
    def forward(self, x):
        return self.net(x)


model = TinyCNN().eval()
#            ↑ .eval(): 推理模式
#              对导出 ONNX 很重要: Dropout 关闭、BatchNorm 用全局统计量
#              虽然这个模型没有 Dropout/BN，但养成习惯

# ── dummy: 假输入, 让 PyTorch 追踪模型结构 ──
# torch.randn(1, 3, 32, 32): 标准正态分布随机数
#   形状 [batch=1, channel=3, H=32, W=32]
#   为什么叫 dummy? 它只是"样本输入"，用来走一遍 forward，让 PyTorch 记录计算图
#   值不重要，形状最重要 — 决定了 ONNX 模型接受的输入维度
dummy = torch.randn(1, 3, 32, 32)


# ═════════════════════════════════════════════════════════════════════════════
# 2. 导出 ONNX
# ═════════════════════════════════════════════════════════════════════════════
# torch.onnx.export: 把 PyTorch 模型的计算图 + 权重一起写成 .onnx 文件
#
# 参数:
#   model           → PyTorch 模型（必须是 eval 模式）
#   dummy           → 一个样例输入，让框架从输入开始追踪所有计算步骤
#                      PyTorch 会实际跑一遍 forward，记录每一步操作
#   onnx_file       → 输出路径
#   input_names     → 给输入节点起名，方便后续推理时按名传数据
#   output_names    → 给输出节点起名，方便推理时按名取结果
#   dynamic_axes    → 声明哪些维度是"动态的"（可变长度）
#                     {"input":{0:"batch"}, "output":{0:"batch"}}
#                     维 0 = batch 维 → 推理时 batch 可以是 1, 2, 8, 任意
#                     如果不声明，导出时固定为 1 → 以后只能推理 batch=1 的输入
#   opset_version   → ONNX 算子集版本号，决定了哪些算子可用
#                     17 是比较新的版本，兼容性好
#                     太小 → 某些操作不支持
#                     太大 → 部署框架可能还没适配
onnx_file = "04_practice_pytorch/tinycnn.onnx"
torch.onnx.export(model, dummy, onnx_file,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17)

# 看看导出的文件多大 — 只存了结构和参数，很小
print(f"ONNX 导出: {onnx_file}  ({os.path.getsize(onnx_file)/1024:.0f} KB)")


# ═════════════════════════════════════════════════════════════════════════════
# 3. PyTorch 推理 — 作为"标准答案"
# ═════════════════════════════════════════════════════════════════════════════
with torch.no_grad():
    pt_output = model(dummy)         # [1, 10] 的 tensor
    pt_output = pt_output.numpy()    # tensor → numpy，方便和 ONNX 输出对比
print(f"PyTorch 输出前5个值: {pt_output[0, :5]}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. ONNX Runtime 推理 — 完全不依赖 PyTorch
# ═════════════════════════════════════════════════════════════════════════════
# InferenceSession: 加载 .onnx 文件，创建一个"推理引擎"
#   内部做的事: 解析 ONNX 计算图 → 分配内存 → 准备执行
#   这里的输入输出都是 numpy，不涉及 torch.Tensor
session = ort.InferenceSession(onnx_file)

# session.run(输出名列表, 输入字典):
#   None        → 表示"我要所有输出"（等效于 ["output"]）
#   {"input": dummp.numpy()} → 输入: 名字必须和 export 时的 input_names 对应
#   [0]         → run() 返回一个列表，[0] 取第一个输出节点
ort_output = session.run(None, {"input": dummy.numpy()})[0]
print(f"ONNX   输出前5个值: {ort_output[0, :5]}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. 验证: PyTorch vs ONNX Runtime
# ═════════════════════════════════════════════════════════════════════════════
# 两个输出应该几乎完全一样（允许极小浮点误差）
# np.abs(pt_output - ort_output): 逐元素取绝对值差
# .max(): 所有差异里的最大值
diff = np.abs(pt_output - ort_output).max()
print(f"PyTorch vs ONNX 最大误差: {diff:.8f}")
# diff < 1e-5: 误差小于十万分之一 → 完美还原
print(f"结果一致: {diff < 1e-5}")

print("✅ 关卡 4 PyTorch版 完成 — 手写 ONNX 导出 + 不依赖 torch 推理")

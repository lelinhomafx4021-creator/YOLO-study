# -*- coding: utf-8 -*-
"""
关卡 4 PyTorch版: torch.onnx.export → ONNX Runtime 推理
手写 ONNX 导出 + ONNX Runtime 加载 + numpy 推理（完全不依赖 torch）
"""
import torch, torch.nn as nn, numpy as np, onnxruntime as ort, os

# ═══ 1. 建一个 SimpleCNN 并保存为 .onnx ═══
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16*8*8, 10))
    def forward(self, x): return self.net(x)

model = TinyCNN().eval()
dummy = torch.randn(1, 3, 32, 32)  # 假输入，让 PyTorch 追踪模型结构

# ═══ 2. 导出 ONNX ═══
onnx_file = "04_practice_pytorch/tinycnn.onnx"
torch.onnx.export(model, dummy, onnx_file,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}},
    opset_version=17)
print(f"ONNX 导出: {onnx_file}  ({os.path.getsize(onnx_file)/1024:.0f} KB)")

# ═══ 3. 用 PyTorch 推理一次，记录结果作为"标准答案" ═══
with torch.no_grad():
    pt_output = model(dummy).numpy()
print(f"PyTorch 输出前5个值: {pt_output[0,:5]}")

# ═══ 4. 用 ONNX Runtime 推理同一输入（完全不依赖 torch） ═══
session = ort.InferenceSession(onnx_file)
ort_output = session.run(None, {"input": dummy.numpy()})[0]
print(f"ONNX   输出前5个值: {ort_output[0,:5]}")

# ═══ 5. 验证 ═══
diff = np.abs(pt_output - ort_output).max()
print(f"PyTorch vs ONNX 最大误差: {diff:.8f}")
print(f"结果一致: {diff < 1e-5}")  # True = ONNX 完美还原了 PyTorch 的计算结果

print("✅ 关卡 4 PyTorch版 完成 — 手写 ONNX 导出 + 不依赖 torch 推理")

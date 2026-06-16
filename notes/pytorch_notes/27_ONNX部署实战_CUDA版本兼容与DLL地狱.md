# ONNX 部署实战：CUDA 版本兼容与 DLL 地狱

## 这一节在学习顺序里的位置

前面你已经知道：

- ONNX 是中间格式，导出后不依赖 PyTorch
- `model.export(format="onnx")` 可以把 `.pt` 转成 `.onnx`
- ONNX Runtime 可以推理，体积比 PyTorch 小很多

这些都没问题。但当你真的在 Windows 上跑 ONNX 推理的时候——

**大概率会报错。**

这一节不讲理论，讲**你一定会踩的坑**和怎么爬出来。

---

## 先说最核心的一句话

**ONNX Runtime 不带 CUDA DLL，PyTorch 带。你得自己配。**

如果只记一句，就记这句。

---

## 1. 为什么 PyTorch 跑 GPU 没问题，ONNX 就报错

你装 PyTorch 的时候，它自动绑了一套 CUDA DLL：

```
.venv/Lib/site-packages/torch/lib/
    cublasLt64_11.dll      ← CUDA 11
    cudart64_110.dll
    cudnn64_9.dll
    ...
```

这些 DLL 是 **CUDA 11.8** 的。PyTorch 启动时直接从这个目录加载，不用你操心。

但 ONNX Runtime 不一样。它不带 DLL。它启动时去系统路径找：

```
onnxruntime → "我要找 cublasLt64_12.dll"
系统:       "没有"
→ 报错
```

**PyTorch 自带 CUDA 11 的 DLL，ONNX Runtime 编译时绑的是 CUDA 12。两个版本对不上。**

---

## 2. 报错长什么样

```
Error loading onnxruntime_providers_cuda.dll
which depends on "cublasLt64_12.dll" which is missing
```

翻译：ONNX Runtime 想用 GPU，但 CUDA 12 的 DLL 找不到。

---

## 3. 怎么解决

最简单的办法：用 pip 装 NVIDIA 官方的 CUDA 12 运行时包。

```bash
pip install nvidia-cuda-runtime-cu12   # 核心运行时
pip install nvidia-cublas-cu12          # 矩阵计算库
pip install nvidia-cufft-cu12           # 傅里叶变换库
pip install nvidia-cudnn-cu12           # 深度学习加速库
```

这些包装完后，DLL 在 `.venv/Lib/site-packages/nvidia/` 下面。

但 ONNX Runtime 不知道去那里找。你需要在代码开头告诉它：

```python
import os
dll_dir = ".venv/Lib/site-packages/nvidia/cublas/bin"
os.add_dll_directory(dll_dir)
# 把其他几个也加上 ...
```

或者更粗暴——直接把 DLL 复制到 `onnxruntime/capi/` 目录下。

**装完这四个包，ONNX Runtime 就能用 CUDA 12 跑 GPU 推理了。**

---

## 4. 装了 CUDA 12 的包，会不会影响 PyTorch

**不会。**

两个 CUDA 版本各管各的：

```
PyTorch 启动:
    去 torch/lib/ 找 cublasLt64_11.dll → 找到 → 用 CUDA 11 ✅

ONNX Runtime 启动:
    去 nvidia/cublas/bin/ 找 cublasLt64_12.dll → 找到 → 用 CUDA 12 ✅
```

文件名不一样（`_11` vs `_12`），路径也不一样。互不干扰。

**两个 CUDA 版本可以共存，就像你电脑上可以同时装 Python 3.10 和 3.11。**

---

## 5. DLL 依赖链是什么

ONNX Runtime 加载 CUDA 不是只找一个 DLL，而是一条链：

```
onnxruntime_providers_cuda.dll
    → 需要 cublasLt64_12.dll
        → 需要 cudnn64_9.dll
            → 需要 cufft64_11.dll
                → 需要 cudart64_12.dll
```

**缺一个就全挂。** 所以那四个包都得装。

---

## 6. `half=True` 和 `half=False` 是什么

导出 ONNX 时有个 `half` 参数：

```python
model.export(format="onnx", half=True)   # FP16: 16位浮点
model.export(format="onnx", half=False)  # FP32: 32位浮点
```

FP16 的每个数字占 16 位，精度低一点，但文件小一半，GPU 推理更快。

**但 FP16 只能用 GPU。**

CPU 没有 FP16 计算单元。你给它 FP16 的数据，它内部也得先转成 FP32 再算，反而多一道转换。

所以：

```
确定跑 GPU → half=True  (模型 5MB，快)
可能跑 CPU → half=False (模型 10MB，兼容)
```

---

## 7. 模型是数字，不绑 CUDA 版本

这是最重要的一点：

```
训练时: PyTorch + CUDA 11.8 → 模型参数 = [0.23, -1.45, 0.87, ...]
导出:   这些数字写进 best.onnx
推理时: ONNX Runtime + CUDA 12 → 读同样的数字 → 在 GPU 上算
```

**CUDA 11 训出来的模型，CUDA 12 推理，完全没问题。** ONNX 是纯权重文件，不包含任何框架信息。模型就是一堆浮点数，不绑定任何 CUDA 版本。

---

## 8. `device=0` 怎么写最通用

ONNX 推理也要指定用哪张 GPU。最通用的写法：

```python
import torch
device = 0 if torch.cuda.is_available() else "cpu"
model.predict(img, device=device)
```

有 GPU 用 0，没有用 `"cpu"`。

---

## 9. 我们实测的数据

RTX 3050 + YOLO11n：

```
best.pt  (PyTorch FP32):  ~33ms/帧
best.onnx (CUDA FP16):    ~10ms/帧  → 快 3 倍
```

---

## 10. 这一节最该记住的 5 句话

1. **ONNX Runtime 不带 CUDA DLL，必须自己装，版本必须匹配**
2. **PyTorch 带自己的 CUDA 11 DLL，和 ONNX 需要的 CUDA 12 不冲突**
3. **`half=True` 的 FP16 模型只能 GPU 跑，CPU 不兼容**
4. **模型就是数字，CUDA 11 训出来的 CUDA 12 推理完全没问题**
5. **四个 nvidia pip 包装完 + `os.add_dll_directory` 就够了**

---

## 复习速答

- `ONNX Runtime`：需要自己准备 CUDA DLL。
- `PyTorch`：通常自带自己的 CUDA 依赖。
- `half=True`：FP16，快但只适合 GPU。
- `CUDA 版本`：只影响运行时，不影响 ONNX 模型文件本身。
- `DLL 问题`：本质是运行时依赖没对上。

---

## 11. 学完这一节后你应该能回答的问题

1. 为什么 PyTorch 跑 GPU 没问题，ONNX 就报 DLL 缺失
   - PyTorch 自带完整的 CUDA 依赖包（cuDNN, cuBLAS 都在 torch/lib 里）。ONNX Runtime 的 GPU 包(onnxruntime-gpu)依赖系统级 CUDA DLL，不会自动带这些文件 → 系统里没有就报错。解决方法: 装 onnxruntime-gpu + 确保 CUDA 相关 DLL 在 PATH 里。

2. 两个 CUDA 版本为什么可以共存
   - CUDA 12(系统安装)负责 PyTorch 训练; CUDA 11(手动安装)负责 ONNX Runtime；它们在不同目录，不冲突。PyTorch 找它的 CUDA 12 DLL，ONNX Runtime 找它的 CUDA 11 DLL。

3. `half=True` 和 `half=False` 的区别是什么
   - half=True: 模型导出时参数转 FP16(半精度 16-bit)。推理更快、显存省一半，但精度略微降低。需要 GPU 支持 FP16(目前所有 RTX 都支持)。
   - half=False: 导出的模型保持 FP32(全精度 32-bit)。CPU/GPU 都能跑，精度不变但慢一些。

4. FP16 为什么只能 GPU 跑
   - CPU 推理引擎只支持 FP32 和 INT8 量化。FP16 是 GPU 上的原生精度，CPU 不支持。如果 half=True 导出的模型在 CPU 上跑 → 报错或自动转 FP32 运行。

5. 训练用 CUDA 11、推理用 CUDA 12 能行吗
   - 完全能行。.onnx 文件里只有模型的参数值(纯数字)，不包含任何 CUDA 版本信息。CUDA 版本只影响推理引擎(ONNX Runtime)本身，不影响模型文件。

6. `device=0` 和 `device="cpu"` 怎么自动判断
   - try: ort.InferenceSession(..., providers=['CUDAExecutionProvider']) → 成功用 GPU。except: ort.InferenceSession(..., providers=['CPUExecutionProvider']) → 降级用 CPU。

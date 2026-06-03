# 工作区对接上下文

## 1. 这个工作区是干嘛的

这是一个面向 `视觉算法基础岗 / 培养型实习` 的学习工作区。

目标不是泛泛学 AI，而是围绕下面这条主线冲刺：

`Python -> NumPy -> OpenCV -> 深度学习基础 -> PyTorch -> YOLO 检测 -> 模型微调 -> Pose demo -> ONNX 导出`

## 2. 用户当前背景

- 当前自我定位：`AI 应用开发入门`
- 当前诉求：争取一个偏 `目标识别 / 姿态识别` 的视觉算法机会
- 当前约束：
  - 学历没有优势
  - 时间不算宽裕
  - 需要走 `实战优先` 路线

## 3. 已经明确下来的策略

- 学习方式：`B 站中文实战入门 + 官方文档纠偏 + 每周必须有产出`
- 不走纯理论路线，不先啃大而全课程
- 先做 `检测`，再做 `姿态`
- 先学 `微调`，不先学“从零手写网络”
- `ONNX` 必学，`FastAPI` 选学

## 4. 最终要拿出来的东西

- `1 个目标检测项目`
- `1 个姿态估计 demo`
- `1 个 ONNX 导出结果`
- `1 套能讲顺的项目表达`

## 5. 当前主工具和主资料

主工具：

- `Python`
- `OpenCV`
- `PyTorch`
- `Ultralytics YOLO`

主资料：

- B 站：中文实战入门
- Ultralytics 中文文档：纠偏和查参数
- PyTorch 官方教程：补训练流程基础
- OpenCV Python 教程：补图像处理基础

## 6. 已经审核过的课程结论

必看：

- `PyTorch CUDA 安装 3050 保姆级`
- `OpenCV Python 快速入门`
- `刘二大人《PyTorch 深度学习实践》前 1-9 讲`
- `YOLO 训练自己的数据集` 这类实战视频
- `MakeSense YOLO 格式数据标注教程`
- `YOLO ONNX 导出教程`

选看：

- `YOLO Pose 关键点检测实战`
- `PyTorch 猫狗分类实战`
- `FastAPI 深度学习部署`
- `吴恩达 deep learning` 的相关章节

不看：

- OpenCV 大长课
- YOLO 全家桶大长课
- MMPose / TensorRT / 源码深挖
- 过多论文推导

## 7. 版本注意事项

- B 站视频常写 `YOLOv8`，这个可以作为搜索词使用。
- 实际落地时，以当前安装的 `ultralytics` 包和官方文档为准。
- 不要把 `YOLOv8 / YOLO11 / 其他版本名` 当成三套完全不同的体系去学。
- 核心动作始终是：
  - `train`
  - `val`
  - `predict`
  - `export`

## 8. 新线程接手时应该怎么做

如果是新的助手接手，请按这个顺序继续：

1. 先读本文件。
2. 再读 [current_status.md](D:/vision_algo_workspace/vision-bootcamp/notes/current_status.md)。
3. 再读 [next_steps.md](D:/vision_algo_workspace/vision-bootcamp/notes/next_steps.md)。
4. 然后再开始实际操作。

## 9. 核心判断标准

后面的学习和实现，都要围绕下面几个问题判断是否有效：

- 能不能跑通一个小项目
- 能不能解释 `loss / optimizer / epoch / learning rate`
- 能不能完成一次 `YOLO` 自定义数据训练
- 能不能看懂 `precision / recall / mAP`
- 能不能说出一次误检和漏检的原因
- 能不能把项目讲清楚

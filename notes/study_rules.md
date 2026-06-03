# 学习规则与资源边界

## 1. 这次学习的主目标

目标不是“学完 AI”，而是：

`把自己练成一个能争取视觉算法基础岗机会的人。`

## 2. 这次必须围绕的核心能力

- 会用 `PyTorch`
- 会跑 `YOLO`
- 会做 `微调`
- 会看 `precision / recall / mAP`
- 会分析 `误检 / 漏检`
- 会讲清楚自己的项目

## 3. 什么资源优先

优先级从高到低：

1. 中文实战视频
2. 官方文档
3. 官方教程
4. 理论补充课

说明：

- 中文实战视频负责让你先跑通
- 官方文档负责纠偏
- 官方教程负责补基础
- 理论课只在卡概念时补

## 4. 什么不允许成为主线

- 大而全的 OpenCV 课程
- 冗长的 YOLO 全版本解析
- 深挖论文推导
- 源码级重构
- 和当前岗位无关的大模型应用路线

## 5. 学习时的判断标准

每看完一个资源，都问自己：

- 我能不能复现出一个结果
- 我能不能讲清楚它在整条流程里是干嘛的
- 它是不是直接服务于检测 / 微调 / 调参 / 导出

如果答案是否定的，就不要继续往那个方向投入太多时间。

## 6. 当前默认资料入口

- Ultralytics 中文文档：<https://docs.ultralytics.com/zh>
- PyTorch Basics：<https://docs.pytorch.org/tutorials/beginner/basics/intro>
- PyTorch Transfer Learning：<https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>
- OpenCV Python 教程：<https://docs.opencv.org/4.x/d0/de3/tutorial_py_intro.html>

## 7. 压缩时间时怎么取舍

如果时间不够，按这个优先级裁剪：

1. 保留 `PyTorch 基础`
2. 保留 `YOLO 检测`
3. 保留 `模型微调`
4. 保留 `指标与调参`
5. 保留 `ONNX 导出`
6. `Pose` 降级为只做 demo
7. `FastAPI` 可以直接跳过

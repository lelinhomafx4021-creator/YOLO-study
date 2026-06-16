# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A computer vision learning workspace targeting entry-level vision algorithm roles. The focus is practical: object detection, model fine-tuning, metrics analysis, ONNX export, and multi-object tracking using PyTorch + Ultralytics YOLO. Recently expanded into Transformer/ViT/DETR fundamentals.

All documentation is in **Chinese (Simplified)**. Respond in Chinese unless asked otherwise.

## Handoff Protocol

When starting a new session, read these files in order before doing anything else:

1. `README_CONTEXT.md` — full project context, user background, strategy, and resource list
2. `notes/current_status.md` — what's been done and what hasn't (may be slightly stale)
3. `notes/next_steps.md` — immediate action items
4. `notes/study_rules.md` — learning boundaries and resource priorities

Then start working from `next_steps.md`. Do not re-discuss strategy or direction.

## Directory Layout

- `00_basics/` — OpenCV, PyTorch, YOLO fundamentals (~18 scripts: image ops, training loops, CIFAR10, NMS, video detection, ByteTrack tracking)
- `01_helmet_detect/` — Core safety helmet detection pipeline: train, val, predict, render, data split
- `02_export_deploy/` — ONNX export, ONNX Runtime inference (image + video), video capture/save utilities
- `03_practice_yolo/` — **Second-round YOLO practice**: deeper dives into ONNX export/inference, video skip-frame, async threading, ByteTrack + Kalman prediction
- `04_practice_pytorch/` — **Second-round PyTorch practice**: NMS from scratch, ONNX export internals, feature map visualization, optimizer comparison, train/validate patterns
- `notes/` — Planning docs, daily logs, execution tracking
- `notes/pytorch_notes/` — 35+ concept notes covering PyTorch, YOLO, metrics, ONNX, CNN, Transformer/ViT/DETR/DINO, ByteTrack, and more. Includes HTML visualization files.
- `datasets/` — Training data (YOLO format)
- `data/` — Additional data assets
- `runs/` — Training run outputs (metrics, weights)
- `output/` — Inference output directory
- `tmp/` — Temporary/scratch files

## Key Pattern: Two-Round Learning

This workspace follows a deliberate two-round learning approach:

1. **First round** (`00_basics/`, `01_helmet_detect/`, `02_export_deploy/`): High-level YOLO API — get things running fast with Ultralytics
2. **Second round** (`03_practice_yolo/`, `04_practice_pytorch/`): Low-level understanding — revisit the same concepts with deeper implementation

Within each practice directory, files follow a naming convention:
- `0X_topic.py` — the main exercise script (well-commented, designed as a teaching tool)
- `0X_mytest.py` — user's hands-on practice/experimentation file for that topic
- Not all topics have a `mytest` variant; they're created as the user works through

## Tech Stack

- Python 3.11 + PyTorch 2.7 + CUDA 118 (NVIDIA RTX 3050 Laptop)
- Ultralytics YOLO for detection, pose, tracking (ByteTrack built-in), export
- OpenCV for image/video processing
- ONNX Runtime for standalone inference (install separately: `pip install onnxruntime-gpu`)
- pandas for metrics analysis

Dependencies managed with `pyproject.toml` + `uv`. Install with:

```
uv sync
```

Note: `onnxruntime` is not in `pyproject.toml` — scripts in `02_export_deploy/` and `03_practice_yolo/` that use it require manual installation.

PyTorch is pinned to the CUDA 118 index in `pyproject.toml`. Do not change this without confirming GPU compatibility.

## Common Commands

```bash
# ── First-round scripts ──

# Run training
python 01_helmet_detect/01_train.py

# Run validation (prints per-class Precision/Recall)
python 01_helmet_detect/03_val.py

# Run prediction on images
python 01_helmet_detect/02_predict.py

# Export model to ONNX
python 02_export_deploy/01_export.py

# ONNX inference on single image
python 02_export_deploy/02_onnx_inference.py

# ONNX inference on video
python 02_export_deploy/04_onnx_video_inference.py

# Split dataset into train/val
python 01_helmet_detect/split_data.py

# ── Second-round / deeper practice ──

# ByteTrack multi-object tracking (video)
python 00_basics/12_bytetrack_tracking.py

# Video detection with frame skipping (performance optimization)
python 00_basics/11_video_detection_skipframe.py

# NMS from scratch + visualization (no framework dependency)
python 04_practice_pytorch/03_nms_from_scratch.py

# Feature map visualization
python 04_practice_pytorch/05_feature_map_visualize.py

# Optimizer comparison (SGD vs Adam vs ...)
python 04_practice_pytorch/06_optimizer_compare.py

# CIFAR10 classifier training (pure PyTorch, no YOLO)
python 00_basics/08_cifar10_classifier.py
```

## Code Architecture

**YOLO API pattern** — All detection scripts follow the same Ultralytics pattern:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")        # load pretrained model
model.train(data=..., epochs=...)  # train
model.val(data=...)                # validate
model.predict(source=..., save=...) # inference
model.export(format="onnx")        # export
```

**ONNX deployment pipeline** (`02_export_deploy/`):
1. `01_export.py` — Converts `.pt` to `.onnx`
2. `02_onnx_inference.py` — Pure ONNX Runtime + OpenCV inference (no PyTorch dependency). Implements letterbox resize, BGR→RGB, NMS, coordinate decoding from scratch
3. `04_onnx_video_inference.py` — Same pipeline applied to video frames
4. `05_video.py`, `06_save_video.py`, `07_onnx_video_homework.py` — Video utilities and exercises

**Second-round deeper dives** (`03_practice_yolo/`, `04_practice_pytorch/`):
- Revisit ONNX export/inference with manual preprocessing alignment
- Video skip-frame detection with async threading + EMA smoothing
- ByteTrack tracking with Kalman filter prediction for skipped frames
- NMS implemented from scratch (pure numpy, no torchvision dependency)
- Feature map extraction and visualization from YOLO layers
- Optimizer behavior comparison with loss curve plotting

**Data flow**: Images in `datasets/safety_helmet/{images,labels}/{train,val}` → YOLO training → weights in `runs/safety_helmet/yolo11n_baseline_v1/weights/` → ONNX export → standalone inference

**Class names**: `helmet`, `no-helmet`, `safety-vest` (defined in `custom_data.yaml` — exists in both root and `01_helmet_detect/`; the training script references the latter via absolute path)

**Path handling**: Training scripts use hardcoded absolute Windows paths (`d:\vision_algo_workspace\...`). ONNX and second-round scripts use dynamic resolution via `Path(__file__).resolve().parent.parent`.

**AGENTS.md**: An identical copy of this file. Keep it in sync if you modify CLAUDE.md.

## Priority When Time Is Short

If cutting scope, preserve in this order:
1. PyTorch basics
2. YOLO detection
3. Model fine-tuning
4. Metrics & tuning
5. ONNX export
6. Pose → demo only
7. FastAPI → skip entirely

## What NOT To Do

- Don't start with theory-heavy courses or paper derivations
- Don't deep-dive into MMPose, TensorRT, or YOLO source code
- Don't take long comprehensive OpenCV or YOLO courses
- Don't build things unrelated to detection/pose/export
- Don't use version-specific YOLO naming as a learning framework
- Don't modify `*_mytest.py` files without the user explicitly asking — those are their personal practice space

## Key References

- Ultralytics Chinese docs: https://docs.ultralytics.com/zh
- PyTorch basics: https://docs.pytorch.org/tutorials/beginner/basics/intro
- OpenCV Python tutorial: https://docs.opencv.org/4.x/d0/de3/tutorial_py_intro.html

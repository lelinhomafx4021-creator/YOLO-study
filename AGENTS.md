# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A 6-week computer vision learning workspace targeting entry-level vision algorithm roles. The focus is practical: object detection, model fine-tuning, metrics analysis, and ONNX export using PyTorch + Ultralytics YOLO.

All documentation is in **Chinese (Simplified)**. Respond in Chinese unless asked otherwise.

## Handoff Protocol

When starting a new session, read these files in order before doing anything else:

1. `README_CONTEXT.md` — full project context, user background, strategy, and resource list
2. `notes/current_status.md` — what's been done and what hasn't
3. `notes/next_steps.md` — immediate action items
4. `notes/study_rules.md` — learning boundaries and resource priorities

Then start working from `next_steps.md`. Do not re-discuss strategy or direction.

## Directory Layout

- `00_basics/` — OpenCV, PyTorch, and YOLO fundamentals (6 scripts)
- `01_helmet_detect/` — Core detection pipeline: train, val, predict, render, data split
- `02_export_deploy/` — ONNX export, inference, video processing
- `notes/` — Planning docs, daily logs, execution tracking
- `notes/pytorch_notes/` — 25+ concept notes covering PyTorch, YOLO, metrics, ONNX, etc.
- `datasets/` — Training data (YOLO format)
- `runs/` — Training run outputs (metrics, weights)

## Tech Stack

- Python 3.11 + PyTorch 2.7 + CUDA 118 (NVIDIA RTX 3050 Laptop)
- Ultralytics YOLO for detection, pose, export
- OpenCV for image processing
- ONNX Runtime for standalone inference (install separately: `pip install onnxruntime-gpu`)

Dependencies managed with `pyproject.toml` + `uv`. Install with:

```
uv sync
```

Note: `onnxruntime` is not in `pyproject.toml` — scripts in `02_export_deploy/` that use it require manual installation.

## Common Commands

```bash
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

**Data flow**: Images in `datasets/safety_helmet/{images,labels}/{train,val}` → YOLO training → weights in `runs/safety_helmet/yolo11n_baseline_v1/weights/` → ONNX export → standalone inference

**Class names**: `helmet`, `no-helmet`, `safety-vest` (defined in `custom_data.yaml` — exists in both root and `01_helmet_detect/`; the training script references the latter via absolute path)

**Path handling**: Training scripts use hardcoded absolute Windows paths (`d:\vision_algo_workspace\...`). ONNX scripts use dynamic resolution via `Path(__file__).resolve().parent.parent`.

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

## Key References

- Ultralytics Chinese docs: https://docs.ultralytics.com/zh
- PyTorch basics: https://docs.pytorch.org/tutorials/beginner/basics/intro
- OpenCV Python tutorial: https://docs.opencv.org/4.x/d0/de3/tutorial_py_intro.html

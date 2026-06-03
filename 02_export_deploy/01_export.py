from pathlib import Path
from ultralytics import YOLO
BASE_DIR=Path(__file__).resolve().parent.parent
pt_path=BASE_DIR/"runs"/"safety_helmet"/"yolo11n_baseline_v1"/"weights"/"best.pt"
model=YOLO(pt_path)
onnx_path=model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    half=False
)
print(f"Exported to: {onnx_path}")
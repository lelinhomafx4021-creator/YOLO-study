# -*- coding: utf-8 -*-
"""
关卡 4 YOLO版: ONNX 导出 + .pt vs .onnx 全面对比
对比项: 推理时间、检测框数、置信度差异、文件大小
"""
import time, os, cv2, numpy as np, onnxruntime as ort
from pathlib import Path
from ultralytics import YOLO

# 把 CUDA 12 的 DLL 路径加进去，让 ONNX Runtime 能找到
_ort_dir = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "onnxruntime" / "capi"
if _ort_dir.exists():
    for _f in os.listdir(str(_ort_dir)):
        if _f.endswith(".dll"):
            os.add_dll_directory(str(_ort_dir))

def main():
    # ═══ 1. 导出 ONNX ═══
    pt_path = Path(r"d:\vision_algo_workspace\vision-bootcamp\runs\safety_helmet\yolo11n_baseline_v1\weights\best.pt")
    model = YOLO(str(pt_path))
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True, half=False)
    print(f"ONNX 导出: {onnx_path}")

    # ═══ 2. 准备测试图 ═══
    img = cv2.imread("result.jpg")
    if img is None:
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # ═══ 3. 预热 + 多次推理取平均 ═══
    n_runs = 20

    # 预热一次（排除 CUDA 编译、缓存初始化等开销）
    print("预热中...")
    _ = model.predict(img, conf=0.25, verbose=False, workers=0, device=0)
    model_onnx = YOLO(str(onnx_path))
    _ = model_onnx.predict(img, conf=0.25, verbose=False, workers=0, device=0)

    # --- .pt ---
    times_pt = []
    for _ in range(n_runs):
        t0 = time.time()
        r_pt = model.predict(img, conf=0.25, verbose=False, workers=0, device=0)
        times_pt.append((time.time()-t0)*1000)
    t_pt_mean = np.mean(times_pt)
    t_pt_std  = np.std(times_pt)

    # --- .onnx (warmup 时已加载) ---
    times_onnx = []
    for _ in range(n_runs):
        t0 = time.time()
        r_onnx = model_onnx.predict(img, conf=0.25, verbose=False, workers=0, device=0)
        times_onnx.append((time.time()-t0)*1000)
    t_onnx_mean = np.mean(times_onnx)
    t_onnx_std  = np.std(times_onnx)

    # ═══ 4. 详细对比 ═══
    print(f"\n{'='*60}")
    print(f"{'指标':<20} {'best.pt':>18} {'best.onnx':>18}")
    print(f"{'-'*60}")
    print(f"{'推理时间 (均值)':<20} {t_pt_mean:>14.1f} ms {t_onnx_mean:>14.1f} ms")
    print(f"{'推理时间 (标准差)':<20} {t_pt_std:>14.1f} ms {t_onnx_std:>14.1f} ms")
    print(f"{'检测框数':<20} {len(r_pt[0].boxes):>17} {len(r_onnx[0].boxes):>17}")

    pt_sz  = os.path.getsize(pt_path) / 1024**2
    onnx_sz = os.path.getsize(onnx_path) / 1024**2
    print(f"{'文件大小':<20} {pt_sz:>14.1f} MB {onnx_sz:>14.1f} MB")
    print(f"{'速度比':<20} {'-':>18} {t_onnx_mean/t_pt_mean:>14.2f}x")

    # ═══ 5. 验证一致性 ═══
    print(f"\n{'='*60}")
    print("验证 .pt vs .onnx:")
    if len(r_pt[0].boxes) == len(r_onnx[0].boxes):
        print(f"  检测框数一致: {len(r_pt[0].boxes)}")
    else:
        print(f"  ⚠️ 框数不同! .pt={len(r_pt[0].boxes)}, .onnx={len(r_onnx[0].boxes)}")

    if len(r_pt[0].boxes) > 0 and len(r_onnx[0].boxes) > 0:
        pt_confs  = r_pt[0].boxes.conf.cpu().numpy()
        onnx_confs = r_onnx[0].boxes.conf.cpu().numpy()
        max_diff = np.abs(pt_confs[:len(onnx_confs)] - onnx_confs[:len(pt_confs)]).max()
        print(f"  置信度最大误差: {max_diff:.6f} → {'✅ 一致' if max_diff < 0.01 else '⚠️ 有偏差'}")

    print(f"\n关键: ONNX 不依赖 torch (只要 100MB onnxruntime vs 3GB torch)")

    # ═══ 6. 保存标注图片 + 每个框的坐标 ═══
    out_dir = Path(r"d:\vision_algo_workspace\vision-bootcamp\03_practice_yolo\predict_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 用最后一次推理的结果（r_pt）
    result = r_pt[0]
    boxes = result.boxes

    # ── 6.1 保存画了框的图片 ──
    # result.plot(): 在原图上画框 + 写类别名和置信度
    annotated = result.plot()
    cv2.imwrite(str(out_dir / "annotated.jpg"), annotated)
    print(f"\n标注图片已保存: {out_dir / 'annotated.jpg'}")

    # ── 6.2 保存每个框的坐标 + 类别 + 置信度到 txt ──
    # boxes.xyxy:  每个框的坐标 [x1, y1, x2, y2]（像素值）
    # boxes.cls:   每个框的类别 ID
    # boxes.conf:  每个框的置信度
    with open(out_dir / "boxes.txt", "w") as f:
        f.write("# x1 y1 x2 y2 class conf\n")
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i].item())
            conf   = boxes.conf[i].item()
            name   = result.names[cls_id]
            f.write(f"{x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f}  "
                    f"{name}  {conf:.3f}\n")

    # ── 6.3 抠出每个目标单独保存 ──
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        cls_id = int(boxes.cls[i].item())
        name   = result.names[cls_id]
        conf   = boxes.conf[i].item()
        crop   = img[y1:y2, x1:x2]   # NumPy 切片抠图：先 Y 后 X！
        crop_path = out_dir / f"crop_{i}_{name}_{conf:.2f}.jpg"
        cv2.imwrite(str(crop_path), crop)

    print(f"框坐标已保存: {out_dir / 'boxes.txt'}")
    print(f"抠图已保存: {out_dir / 'crop_*.jpg'} 共 {len(boxes)} 张")
    print("✅ 关卡 4 YOLO版 完成")


if __name__ == "__main__":
    main()

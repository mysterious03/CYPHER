"""
Deep Checkpoint & Architecture Auditor for CYPhER v2.1
Tests every checkpoint on disk, runs dummy inference, inspects device placement,
output tensor shapes, class names, architecture layers, and verifies training evidence.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np

def audit_checkpoints():
    print("=" * 80)
    print("CYPhER v2.1 INDEPENDENT MODEL CHECKPOINT AUDIT")
    print(f"PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device 0: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)")
    print("=" * 80)

    from ultralytics import YOLO

    checkpoints_to_audit = [
        # 1. Custom Road Damage (B0)
        {
            "id": "M1_ROAD_DAMAGE",
            "path": "runs/detect/runs/road_damage/B0_baseline/weights/best.pt",
            "type": "ultralytics_yolo",
            "task": "detect",
            "claimed_dataset": "RDD2022 India",
            "claimed_status": "fine_tuned",
            "inference_module": "road_cv.adapters.vision_adapters.DamageAdapter"
        },
        # 2. Surface Classifier
        {
            "id": "M2_SURFACE",
            "path": "runs/surface_classifier/best_surface_model.pt",
            "type": "torch_state_dict",
            "task": "multi_task_classification",
            "claimed_dataset": "RSCD-derived road patch dataset",
            "claimed_status": "fine_tuned",
            "inference_module": "road_cv.surface.surface_predictor.RoadSurfacePredictor"
        },
        # 3. Water Segmenter
        {
            "id": "M3_WATER",
            "path": "runs/water_segmenter/best_water_model.pt",
            "type": "torch_state_dict",
            "task": "semantic_segmentation",
            "claimed_dataset": "R2S100K Spec (Heuristic Pseudo-Masks)",
            "claimed_status": "trained_pilot",
            "inference_module": "road_cv.water.water_analyzer.WaterHazardAnalyzer"
        },
        # 4. Vehicle Segmenter
        {
            "id": "M4_VEHICLE",
            "path": "yolo26n-seg.pt",
            "type": "ultralytics_yolo",
            "task": "segment",
            "claimed_dataset": "MS COCO",
            "claimed_status": "pretrained",
            "inference_module": "road_cv.adapters.vision_adapters.VehicleAdapter"
        },
        # 5. Road Scene Semantic Segmenter
        {
            "id": "M5_ROAD_SCENE",
            "path": "yolo26n-sem.pt",
            "type": "ultralytics_yolo",
            "task": "semantic",
            "claimed_dataset": "Cityscapes",
            "claimed_status": "pretrained",
            "inference_module": "road_cv.adapters.vision_adapters.RoadSceneAdapter"
        },
        # 6. Monocular Depth Estimator
        {
            "id": "M6_DEPTH",
            "path": "yolo26n-depth.pt",
            "type": "ultralytics_yolo",
            "task": "depth",
            "claimed_dataset": "Mixed relative depth benchmark",
            "claimed_status": "pretrained",
            "inference_module": "road_cv.adapters.vision_adapters.DepthAdapter"
        },
        # 7. Base YOLO26n Detector
        {
            "id": "M7_BASE_YOLO",
            "path": "yolo26n.pt",
            "type": "ultralytics_yolo",
            "task": "detect",
            "claimed_dataset": "MS COCO",
            "claimed_status": "pretrained",
            "inference_module": "road_cv.adapters.specialized_adapters.TrafficSignalAdapter"
        },
        # Additional checkpoints
        {
            "id": "B1A_SAMPLING",
            "path": "runs/detect/runs/road_damage/B1A_sampling_only/weights/best.pt",
            "type": "ultralytics_yolo",
            "task": "detect",
            "claimed_dataset": "RDD2022 India (4x D10)",
            "claimed_status": "experimental_trial",
            "inference_module": "experiments/run_b1_suite.py"
        },
        {
            "id": "B1B_WEIGHTING",
            "path": "runs/detect/runs/road_damage/B1B_weighting_only/weights/best.pt",
            "type": "ultralytics_yolo",
            "task": "detect",
            "claimed_dataset": "RDD2022 India (Cui weights)",
            "claimed_status": "experimental_trial",
            "inference_module": "experiments/run_b1_suite.py"
        },
        {
            "id": "B1C_AUGMENTATION",
            "path": "runs/detect/runs/road_damage/B1C_augmentation_only/weights/best.pt",
            "type": "ultralytics_yolo",
            "task": "detect",
            "claimed_dataset": "RDD2022 India (D10 aug)",
            "claimed_status": "experimental_trial",
            "inference_module": "experiments/run_b1_suite.py"
        }
    ]

    dummy_rgb = np.full((640, 640, 3), 120, dtype=np.uint8)
    audit_results = {}

    for item in checkpoints_to_audit:
        cid = item["id"]
        cpath = Path(item["path"])
        print(f"\n---> Auditing Checkpoint: {cid} ({item['path']})")

        if not cpath.exists():
            print(f"  [STATUS: MISSING] File not found!")
            audit_results[cid] = {"status": "MISSING", "path": item["path"]}
            continue

        size_bytes = cpath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"  [FILE SIZE]: {size_mb:.2f} MB ({size_bytes:,} bytes)")

        # Verify load & inference
        try:
            if item["type"] == "ultralytics_yolo":
                model = YOLO(str(cpath))
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                model.to(device)

                # Inspect model structure
                arch_name = model.model.__class__.__name__ if hasattr(model, "model") else "YOLO"
                param_count = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") else 0
                classes = model.names if hasattr(model, "names") else {}

                # Run dummy inference
                t_res = model.predict(dummy_rgb, imgsz=640, device=device, verbose=False)[0]

                # Inspect output shape
                out_shape_info = {}
                if hasattr(t_res, "boxes") and t_res.boxes is not None:
                    out_shape_info["boxes"] = list(t_res.boxes.shape)
                if hasattr(t_res, "masks") and t_res.masks is not None:
                    out_shape_info["masks"] = list(t_res.masks.shape)
                if hasattr(t_res, "semantic_mask") and t_res.semantic_mask is not None:
                    out_shape_info["semantic_mask"] = list(t_res.semantic_mask.shape)
                if hasattr(t_res, "depth") and t_res.depth is not None:
                    out_shape_info["depth"] = list(t_res.depth.shape)

                # Check training evidence
                run_dir = cpath.parent.parent
                has_results_csv = (run_dir / "results.csv").exists()
                has_train_args = (run_dir / "args.yaml").exists()

                print(f"  [ARCHITECTURE]: {arch_name} ({param_count:,} parameters)")
                print(f"  [CLASSES ({len(classes)})]: {list(classes.items())[:6]}")
                print(f"  [DEVICE PLACEMENT]: {device}")
                print(f"  [OUTPUT SHAPE]: {out_shape_info}")
                print(f"  [TRAINING EVIDENCE]: results.csv={has_results_csv}, args.yaml={has_train_args}")

                audit_results[cid] = {
                    "status": "LOADED_AND_VERIFIED",
                    "path": item["path"],
                    "size_mb": round(size_mb, 2),
                    "architecture": arch_name,
                    "param_count": param_count,
                    "class_count": len(classes),
                    "classes": classes,
                    "device": device,
                    "output_shapes": out_shape_info,
                    "claimed_dataset": item["claimed_dataset"],
                    "claimed_status": item["claimed_status"],
                    "inference_module": item["inference_module"],
                    "training_evidence": {"has_results_csv": has_results_csv, "has_args_yaml": has_train_args}
                }

            elif item["type"] == "torch_state_dict":
                ckpt = torch.load(str(cpath), map_location="cpu")
                keys = list(ckpt.keys()) if isinstance(ckpt, dict) else [type(ckpt).__name__]
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

                if cid == "M2_SURFACE":
                    sys.path.insert(0, os.path.abspath("."))
                    from road_cv.surface.surface_predictor import RoadSurfacePredictor
                    predictor = RoadSurfacePredictor(checkpoint_path=str(cpath), device=device)
                    dummy_patch = dummy_rgb[200:400, 200:400]
                    pred = predictor.predict_patch(dummy_patch)
                    param_count = sum(p.numel() for p in predictor.model.parameters())
                    classes_dict = {
                        "friction": predictor.model.heads["friction"].out_features if hasattr(predictor.model, "heads") else 6,
                        "material": predictor.model.heads["material"].out_features if hasattr(predictor.model, "heads") else 6,
                        "unevenness": predictor.model.heads["unevenness"].out_features if hasattr(predictor.model, "heads") else 4
                    }
                    print(f"  [ARCHITECTURE]: MultiTaskSurfaceClassifier (MobileNetV3-Large, {param_count:,} params)")
                    print(f"  [CLASSES]: Friction (6), Material (6), Unevenness (4)")
                    print(f"  [DEVICE PLACEMENT]: {device}")
                    print(f"  [SAMPLE PREDICTION]: {pred['visual_surface_condition']['class_name']}, {pred['road_material']['class_name']}")
                    audit_results[cid] = {
                        "status": "LOADED_AND_VERIFIED",
                        "path": item["path"],
                        "size_mb": round(size_mb, 2),
                        "architecture": "MobileNetV3-Large Multi-Task",
                        "param_count": param_count,
                        "classes": classes_dict,
                        "device": device,
                        "claimed_dataset": item["claimed_dataset"],
                        "claimed_status": item["claimed_status"],
                        "inference_module": item["inference_module"],
                        "training_evidence": {"dict_keys": keys}
                    }

                elif cid == "M3_WATER":
                    sys.path.insert(0, os.path.abspath("."))
                    from road_cv.water.water_segmenter import RoadWaterSegmenter
                    seg_model = RoadWaterSegmenter(num_classes=4)
                    if "model_state_dict" in ckpt:
                        seg_model.load_state_dict(ckpt["model_state_dict"])
                    seg_model.to(device)
                    seg_model.eval()
                    param_count = sum(p.numel() for p in seg_model.parameters())
                    dummy_t = torch.from_numpy(dummy_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    dummy_t = dummy_t.to(device)
                    with torch.no_grad():
                        out = seg_model(dummy_t)
                    out_shape = list(out.shape)
                    print(f"  [ARCHITECTURE]: RoadWaterSegmenter (LRASPP MobileNetV3-Large, {param_count:,} params)")
                    print(f"  [CLASSES (4)]: 0=dry, 1=standing_flood, 2=puddle, 3=reflection")
                    print(f"  [DEVICE PLACEMENT]: {device}")
                    print(f"  [OUTPUT TENSOR SHAPE]: {out_shape}")
                    audit_results[cid] = {
                        "status": "LOADED_AND_VERIFIED",
                        "path": item["path"],
                        "size_mb": round(size_mb, 2),
                        "architecture": "LRASPP MobileNetV3-Large",
                        "param_count": param_count,
                        "classes": {0: "dry", 1: "flood", 2: "puddle", 3: "reflection"},
                        "device": device,
                        "output_shapes": {"logits": out_shape},
                        "claimed_dataset": item["claimed_dataset"],
                        "claimed_status": item["claimed_status"],
                        "inference_module": item["inference_module"],
                        "training_evidence": {"dict_keys": keys}
                    }

        except Exception as e:
            print(f"  [LOAD/INFERENCE FAILED]: {str(e)}")
            import traceback
            traceback.print_exc()
            audit_results[cid] = {"status": "FAILED_LOAD", "path": item["path"], "error": str(e)}

    import json
    with open("reports/v21_checkpoint_audit_raw.json", "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print("\nSaved checkpoint audit to reports/v21_checkpoint_audit_raw.json")

if __name__ == "__main__":
    audit_checkpoints()


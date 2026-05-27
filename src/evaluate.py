"""
Evaluate and benchmark YOLO26-seg on CholecSeg8k.

Usage:
    python src/evaluate.py --weights results/cholecseg_yolo26n-seg/weights/best.pt
    python src/evaluate.py --ablation   # Runs all model variants
"""

import argparse
import json
import time
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


CLASS_NAMES = [
    "background", "abdominal_wall", "liver", "gi_tract", "fat",
    "grasper", "connective_tissue", "blood", "cystic_duct",
    "l_hook", "gallbladder", "hepatoduodenal_ligament", "clipper"
]


def evaluate_model(weights_path: str, data_yaml: str = "configs/data.yaml"):
    model = YOLO(weights_path)

    print(f"\n📊 Evaluating: {weights_path}")
    metrics = model.val(data=data_yaml, split="test", plots=True, save_json=True)

    results = {
        "model": Path(weights_path).stem,
        "mAP50_mask": round(metrics.seg.map50, 4),
        "mAP50_95_mask": round(metrics.seg.map, 4),
        "mAP50_box": round(metrics.box.map50, 4),
    }

    # Measure inference speed (fps)
    import cv2
    dummy = "data/processed/images/val"
    if Path(dummy).exists():
        test_imgs = list(Path(dummy).glob("*.png"))[:50]
        start = time.time()
        for img in test_imgs:
            model(str(img), verbose=False)
        elapsed = time.time() - start
        results["fps"] = round(len(test_imgs) / elapsed, 1)
    else:
        results["fps"] = "N/A"

    print(f"  mAP50-Mask:    {results['mAP50_mask']}")
    print(f"  mAP50-95-Mask: {results['mAP50_95_mask']}")
    print(f"  FPS:           {results['fps']}")
    return results


def run_ablation():
    """Compare YOLO26 variants vs YOLOv11 baseline."""
    model_variants = [
        "yolo11n-seg",
        "yolo11s-seg",
        "yolo26n-seg",
        "yolo26s-seg",
    ]

    all_results = []
    for variant in model_variants:
        weights = f"results/cholecseg_{variant}/weights/best.pt"
        if not Path(weights).exists():
            print(f"⚠️  Weights not found for {variant}, skipping...")
            continue
        results = evaluate_model(weights)
        all_results.append(results)

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n📊 Ablation Study Results:")
        print(df.to_string(index=False))
        df.to_csv("results/ablation_study.csv", index=False)
        print("\n✅ Saved to results/ablation_study.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="Path to model weights (.pt)")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    args = parser.parse_args()

    if args.ablation:
        run_ablation()
    elif args.weights:
        evaluate_model(args.weights, args.data)
    else:
        print("Provide --weights or use --ablation flag")

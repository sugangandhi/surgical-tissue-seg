"""
Train YOLO26-seg on CholecSeg8k dataset.

Usage:
    python src/train.py --model yolo26n-seg --epochs 100 --batch 16
    python src/train.py --model yolo26s-seg --epochs 150 --batch 8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(model_name: str, epochs: int, batch: int, imgsz: int, device: str):
    print(f"\n🚀 Training {model_name} for {epochs} epochs...")

    model = YOLO(f"{model_name}.pt")

    results = model.train(
        data="configs/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=f"cholecseg_{model_name}",
        project="results",
        patience=20,
        save=True,
        save_period=10,
        val=True,
        augment=True,
        # Augmentation tweaks for surgical video
        degrees=10.0,
        flipud=0.0,       # Surgical frames are upright
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        # Optimizer
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,
        cos_lr=True,
        # Logging
        plots=True,
        verbose=True,
    )

    best_weights = Path(f"results/cholecseg_{model_name}/weights/best.pt")
    print(f"\n✅ Training complete. Best weights: {best_weights}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n-seg",
                        choices=["yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo11n-seg", "yolo11s-seg"],
                        help="Model variant to train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="GPU device (0, 1, cpu)")
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch, args.imgsz, args.device)

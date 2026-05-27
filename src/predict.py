"""
Run inference on a single image, video, or webcam.

Usage:
    python src/predict.py --source path/to/video.mp4 --weights models/best.pt
    python src/predict.py --source path/to/image.jpg --weights models/best.pt --sam2
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, SAM

CLASS_NAMES = [
    "background", "abdominal_wall", "liver", "gi_tract", "fat",
    "grasper", "connective_tissue", "blood", "cystic_duct",
    "l_hook", "gallbladder", "hepatoduodenal_ligament", "clipper"
]

# Class-specific colors (BGR)
CLASS_COLORS = {
    0: (50, 50, 50), 1: (0, 255, 127), 2: (0, 200, 255),
    3: (255, 100, 0), 4: (255, 200, 100), 5: (100, 100, 255),
    6: (200, 255, 0), 7: (0, 0, 255), 8: (255, 0, 200),
    9: (150, 0, 255), 10: (0, 255, 0), 11: (255, 150, 0),
    12: (200, 200, 200)
}


def predict_with_sam2(image_path: str, yolo_model: YOLO, sam_model: SAM):
    """YOLO26 detection → SAM2 mask refinement pipeline."""
    det_results = yolo_model(image_path, conf=0.3, verbose=False)
    boxes = det_results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return det_results[0].plot()
    sam_results = sam_model(image_path, bboxes=boxes)
    return sam_results[0].plot()


def run_inference(source: str, weights: str, use_sam2: bool = False,
                  conf: float = 0.3, save: bool = True):
    yolo = YOLO(weights)
    sam = SAM("sam2.1_b.pt") if use_sam2 else None

    source_path = Path(source)
    output_dir = Path("results/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Image inference
    if source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        if use_sam2 and sam:
            annotated = predict_with_sam2(source, yolo, sam)
            out_path = output_dir / f"sam2_{source_path.name}"
            cv2.imwrite(str(out_path), annotated)
        else:
            results = yolo(source, conf=conf, save=save, project=str(output_dir))
            annotated = results[0].plot()
            out_path = output_dir / source_path.name
            cv2.imwrite(str(out_path), annotated)
        print(f"✅ Saved prediction to: {out_path}")

    # Video inference
    elif source_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
        cap = cv2.VideoCapture(source)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = output_dir / f"pred_{source_path.name}"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = yolo(frame, conf=conf, verbose=False)
            annotated = results[0].plot()
            writer.write(annotated)
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  Processed {frame_count} frames...")

        cap.release()
        writer.release()
        print(f"✅ Saved video prediction to: {out_path} ({frame_count} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--sam2", action="store_true", help="Use SAM2 refinement")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    run_inference(args.source, args.weights, args.sam2, args.conf)

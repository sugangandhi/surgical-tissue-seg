"""
Convert CholecSeg8k color-coded masks → YOLO polygon segmentation format.

Usage:
    python scripts/convert_masks.py --input data/raw/ --output data/processed/
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
import random

# CholecSeg8k color → class ID mapping (BGR format for OpenCV)
COLOR_MAP = {
    (0,   0,   0):   0,   # Background
    (0,   255, 0):   1,   # Abdominal wall (green)
    (0,   255, 255): 2,   # Liver (yellow)
    (255, 0,   0):   3,   # GI Tract (blue)
    (255, 0,   255): 4,   # Fat (magenta)
    (255, 255, 0):   5,   # Grasper (cyan)
    (0,   0,   255): 6,   # Connective tissue (red)
    (255, 165, 0):   7,   # Blood (orange)
    (128, 0,   128): 8,   # Cystic duct (purple)
    (0,   128, 0):   9,   # L-hook electrocautery
    (128, 128, 0):   10,  # Gallbladder
    (0,   0,   128): 11,  # Hepatoduodenal ligament
    (128, 0,   0):   12,  # Clipper
}

IMG_W, IMG_H = 854, 480
MIN_CONTOUR_AREA = 100


def mask_to_yolo_polygons(mask_path: str) -> list[str]:
    """Convert a color mask to YOLO segmentation label lines."""
    mask = cv2.imread(mask_path)
    if mask is None:
        return []

    lines = []
    for bgr_color, class_id in COLOR_MAP.items():
        if class_id == 0:
            continue  # skip background
        lower = np.array(bgr_color, dtype=np.uint8)
        binary = cv2.inRange(mask, lower, lower)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            # Simplify contour
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            pts = approx.reshape(-1, 2).astype(float)
            pts[:, 0] /= IMG_W
            pts[:, 1] /= IMG_H
            pts = np.clip(pts, 0, 1)

            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            lines.append(f"{class_id} {coords}")
    return lines


def split_dataset(image_paths: list, ratios=(0.8, 0.1, 0.1)):
    """Split into train/val/test."""
    random.seed(42)
    random.shuffle(image_paths)
    n = len(image_paths)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    return image_paths[:train_end], image_paths[train_end:val_end], image_paths[val_end:]


def convert_dataset(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Collect all image paths (assumes paired image + mask_visicolor.png)
    image_paths = sorted(input_path.rglob("*.png"))
    image_paths = [p for p in image_paths if "visicolor" not in p.name]

    print(f"Found {len(image_paths)} images")
    train_imgs, val_imgs, test_imgs = split_dataset(image_paths)

    for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        img_out = output_path / "images" / split
        lbl_out = output_path / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"\nConverting {split} split ({len(imgs)} images)...")
        for img_path in tqdm(imgs):
            # Find paired mask
            mask_path = str(img_path).replace(".png", "_visicolor.png")
            if not os.path.exists(mask_path):
                continue

            # Copy image
            shutil.copy(img_path, img_out / img_path.name)

            # Convert + save label
            label_lines = mask_to_yolo_polygons(mask_path)
            label_file = lbl_out / img_path.with_suffix(".txt").name
            with open(label_file, "w") as f:
                f.write("\n".join(label_lines))

    print(f"\n✅ Dataset converted to YOLO format at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CholecSeg8k data")
    parser.add_argument("--output", required=True, help="Output path for YOLO-format data")
    args = parser.parse_args()
    convert_dataset(args.input, args.output)

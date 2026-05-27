# 🔬 Surgical Tissue Segmentation with YOLO26

> Real-time 13-class tissue segmentation on laparoscopic surgery videos using **YOLO26-seg + SAM2** refinement.

[![Hugging Face](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/YOLO26-Ultralytics-orange)](https://ultralytics.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Overview

This project fine-tunes **YOLO26-seg** on the [CholecSeg8k](https://arxiv.org/abs/2012.12453) dataset for real-time semantic tissue segmentation in laparoscopic cholecystectomy videos. A **SAM2** refinement stage improves mask precision on small and complex tissue boundaries.

### 13 Tissue Classes
| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Background | 7 | Blood |
| 1 | Abdominal Wall | 8 | Cystic Duct |
| 2 | Liver | 9 | L-Hook Electrocautery |
| 3 | Gastrointestinal Tract | 10 | Gallbladder |
| 4 | Fat | 11 | Hepatoduodenal Ligament |
| 5 | Grasper | 12 | Clipper |
| 6 | Connective Tissue | | |

---

## 📊 Results

| Model | mAP50-Mask | mAP50-95-Mask | FPS (RTX 3060) |
|-------|-----------|---------------|----------------|
| YOLOv11n-seg (baseline) | - | - | - |
| YOLO26n-seg | - | - | - |
| YOLO26s-seg | - | - | - |
| YOLO26s-seg + SAM2 | - | - | - |


---

## 🗂️ Project Structure

```
surgical-tissue-seg/
├── app/
│   └── gradio_demo.py          # Hugging Face Spaces demo
├── configs/
│   └── data.yaml               # YOLO dataset config
├── data/
│   ├── raw/                    # Original CholecSeg8k images + masks
│   └── processed/              # Converted YOLO-format labels
├── models/                     # Saved checkpoints (.pt files)
├── notebooks/
│   └── 01_explore_dataset.ipynb
├── results/                    # Evaluation metrics, plots
├── scripts/
│   └── convert_masks.py        # CholecSeg8k → YOLO polygon format
├── src/
│   ├── dataset.py              # Dataset loading utilities
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation & benchmarking
│   └── predict.py              # Inference on video/image
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/surgical-tissue-seg.git
cd surgical-tissue-seg
pip install -r requirements.txt
```

### Download Dataset

```bash
# Option 1: Hugging Face
python -c "from datasets import load_dataset; ds = load_dataset('minwoosun/CholecSeg8k', trust_remote_code=True)"

# Option 2: Kaggle
kaggle datasets download newslab/cholecseg8k -p data/raw/
```

---

##  Usage

### 1. Convert Dataset
```bash
python scripts/convert_masks.py --input data/raw/ --output data/processed/
```

### 2. Train
```bash
python src/train.py --model yolo26n-seg --epochs 100 --batch 16
# Scale up:
python src/train.py --model yolo26s-seg --epochs 150 --batch 8
```

### 3. Evaluate
```bash
python src/evaluate.py --weights models/best.pt --data configs/data.yaml
```

### 4. Run Demo Locally
```bash
python app/gradio_demo.py
```

---

##  Ablation Study

Run the full ablation comparing YOLO26 variants vs YOLOv11 baseline:
```bash
python src/evaluate.py --ablation
```

---

##  Hugging Face Deployment

```bash
# Push to HF Spaces
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='app/', repo_id='YOUR_HF_USERNAME/surgical-tissue-seg', repo_type='space')
"
```

---

##  References

- [CholecSeg8k Paper](https://arxiv.org/abs/2012.12453)
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)
- [SAM2 (Meta)](https://github.com/facebookresearch/segment-anything-2)
- [Surgical Video Segmentation Review (2025)](https://www.sciencedirect.com/science/article/pii/S0010482525008339)

---

## 📄 License
MIT License — see [LICENSE](LICENSE)

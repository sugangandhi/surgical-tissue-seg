"""
Dataset utilities for CholecSeg8k.
"""
import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


def download_cholecseg8k(output_dir: str = "data/raw"):
    """Download CholecSeg8k from Hugging Face."""
    try:
        from datasets import load_dataset
        print("📥 Downloading CholecSeg8k from Hugging Face...")
        ds = load_dataset("minwoosun/CholecSeg8k", trust_remote_code=True)
        print(f"✅ Dataset loaded: {ds}")
        return ds
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Try: kaggle datasets download newslab/cholecseg8k -p data/raw/")


class CholecSeg8kDataset(Dataset):
    """PyTorch Dataset for CholecSeg8k images."""
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = Path(img_dir)
        self.images = sorted(self.img_dir.glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)


if __name__ == "__main__":
    download_cholecseg8k()

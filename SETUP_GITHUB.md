# 🚀 Push to GitHub — One-Time Setup

Follow these steps to create the repo and push all code:

## Step 1: Create GitHub Repo
Go to https://github.com/new and create a repo named `surgical-tissue-seg`
- Set it to **Public**
- Do NOT initialize with README (we have one)

## Step 2: Push Code

```bash
cd surgical-tissue-seg

git init
git add .
git commit -m "🔬 Initial commit: YOLO26 surgical tissue segmentation"

# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/surgical-tissue-seg.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy to Hugging Face Spaces

```bash
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login

# Create a new Space (Gradio SDK)
# Go to https://huggingface.co/new-space
# Name it: surgical-tissue-seg
# SDK: Gradio

# Push app files
cd app/
git init
git add .
git commit -m "Add Gradio demo"
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/surgical-tissue-seg
git push space main
```

## Step 4: Upload Model Weights to HF Hub

```bash
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="results/cholecseg_yolo26s-seg/weights/best.pt",
    path_in_repo="best.pt",
    repo_id="YOUR_HF_USERNAME/surgical-tissue-seg-weights",
    repo_type="model"
)
```

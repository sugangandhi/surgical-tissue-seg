"""
Hugging Face Spaces — Real-Time Surgical Tissue Segmentation
Deploy: push this file to a HF Space (Gradio SDK)
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Model will be auto-downloaded or loaded from local weights
MODEL_PATH = "models/best.pt"  # replace with your HF model path if uploaded

CLASS_NAMES = [
    "background", "abdominal_wall", "liver", "gi_tract", "fat",
    "grasper", "connective_tissue", "blood", "cystic_duct",
    "l_hook", "gallbladder", "hepatoduodenal_ligament", "clipper"
]

model = None


def load_model():
    global model
    if model is None:
        try:
            model = YOLO(MODEL_PATH)
            print("✅ Model loaded")
        except Exception as e:
            print(f"⚠️  Using pretrained YOLO26n-seg (no fine-tuned weights found): {e}")
            model = YOLO("yolo26n-seg.pt")
    return model


def segment_tissue(image: Image.Image, confidence: float = 0.3):
    """Run YOLO26 tissue segmentation on an input image."""
    m = load_model()
    img_array = np.array(image)
    results = m(img_array, conf=confidence, verbose=False)

    annotated = results[0].plot(
        labels=True,
        boxes=True,
        masks=True,
        conf=True,
        line_width=2,
    )
    annotated_pil = Image.fromarray(annotated[..., ::-1])

    # Build detection summary
    boxes = results[0].boxes
    summary_lines = []
    if boxes is not None and len(boxes):
        for cls_id, conf_val in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            name = CLASS_NAMES[int(cls_id)]
            summary_lines.append(f"• {name}: {conf_val:.2f}")
        summary = "\n".join(summary_lines)
    else:
        summary = "No tissue regions detected above confidence threshold."

    return annotated_pil, summary


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Surgical Tissue Segmentation — YOLO26",
    theme=gr.themes.Soft(primary_hue="teal"),
) as demo:
    gr.Markdown("""
    # 🔬 Real-Time Surgical Tissue Segmentation
    **YOLO26-seg fine-tuned on CholecSeg8k** | 13 tissue/instrument classes | Laparoscopic cholecystectomy
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="Upload Surgical Frame")
            conf_slider = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence Threshold")
            run_btn = gr.Button("🔍 Segment", variant="primary")

        with gr.Column(scale=1):
            out_image = gr.Image(type="pil", label="Segmentation Output")
            out_text = gr.Textbox(label="Detected Tissue Classes", lines=8)

    run_btn.click(
        fn=segment_tissue,
        inputs=[inp_image, conf_slider],
        outputs=[out_image, out_text],
    )

    gr.Examples(
        examples=[],  # Add sample frame paths here
        inputs=inp_image,
    )

    gr.Markdown("""
    ---
    **Classes:** Background · Abdominal Wall · Liver · GI Tract · Fat · Grasper ·
    Connective Tissue · Blood · Cystic Duct · L-Hook · Gallbladder · Hepatoduodenal Ligament · Clipper

    📂 [GitHub](https://github.com/YOUR_USERNAME/surgical-tissue-seg) |
    📄 [CholecSeg8k Paper](https://arxiv.org/abs/2012.12453) |
    🤖 [YOLO26 Docs](https://docs.ultralytics.com/models/yolo26)
    """)

if __name__ == "__main__":
    demo.launch(share=False)

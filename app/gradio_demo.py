"""
Gradio Demo: Upload an image → three models predict simultaneously + Grad-CAM heatmaps.

Usage:
    python -m app.gradio_demo
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from data.dataset import get_val_transform, IMAGENET_MEAN, IMAGENET_STD
from models.build_model import build_model, get_model_family
from visualization.grad_cam import GradCAM, get_target_layer


# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# Models to load (update paths to your trained checkpoints)
MODEL_CONFIGS = [
    {"name": "resnet18", "display": "ResNet-18 (CNN)", "checkpoint": "outputs/resnet18/best.pth"},
    {"name": "deit_tiny_patch16_224", "display": "DeiT-Tiny (ViT)", "checkpoint": "outputs/deit_tiny_patch16_224/best.pth"},
    {"name": "vim_tiny", "display": "Vim-Tiny (Mamba)", "checkpoint": "outputs/vim_tiny/best.pth"},
]

# CIFAR-100 class names (loaded lazily)
CLASS_NAMES = None


def load_class_names():
    global CLASS_NAMES
    if CLASS_NAMES is None:
        try:
            from data.dataset import get_class_names
            CLASS_NAMES = get_class_names("cifar100")
        except Exception:
            CLASS_NAMES = [str(i) for i in range(100)]
    return CLASS_NAMES


def load_models():
    """Load all three models."""
    models = {}
    for cfg in MODEL_CONFIGS:
        model = build_model(cfg["name"], num_classes=100, pretrained=False)
        ckpt_path = Path(cfg["checkpoint"])
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  [WARN] No checkpoint found at {ckpt_path}, using random weights")
        model = model.to(DEVICE)
        model.eval()
        models[cfg["display"]] = model
    return models


def generate_heatmap(model, model_name_key, img_tensor):
    """Generate Grad-CAM heatmap for a single model."""
    # Determine original model name from display name
    for cfg in MODEL_CONFIGS:
        if cfg["display"] == model_name_key:
            original_name = cfg["name"]
            break
    else:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    try:
        target_layer = get_target_layer(model, original_name)
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(img_tensor.to(DEVICE))

        # Resize to image size
        if heatmap.ndim == 2:
            from PIL import Image as PILImage
            heatmap = np.array(
                PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR
                )
            ) / 255.0
        return heatmap
    except Exception as e:
        print(f"  Grad-CAM failed for {model_name_key}: {e}")
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE))


def predict(image):
    """
    Main prediction function for Gradio.

    Args:
        image: PIL Image from user upload.

    Returns:
        results_text: Formatted prediction results.
        heatmap_figure: Matplotlib figure with heatmap comparison.
    """
    if image is None:
        return "Please upload an image.", None

    models = load_models()
    class_names = load_class_names()
    transform = get_val_transform(IMAGE_SIZE)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    # Denormalized image for display
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img_display = img_tensor.squeeze().permute(1, 2, 0).numpy() * std + mean
    img_display = np.clip(img_display, 0, 1)

    # Predict with each model
    results_lines = []
    predictions = {}

    for display_name, model in models.items():
        with torch.no_grad():
            output = model(img_tensor.to(DEVICE))
            probs = F.softmax(output, dim=1)[0]
            top5_prob, top5_idx = probs.topk(5)

        pred_class = class_names[top5_idx[0].item()]
        pred_conf = top5_prob[0].item()
        predictions[display_name] = (pred_class, pred_conf)

        family = display_name.split("(")[1].rstrip(")")
        results_lines.append(f"[{family}] {display_name}:")
        for i in range(5):
            cls = class_names[top5_idx[i].item()]
            prob = top5_prob[i].item()
            bar = "█" * int(prob * 20)
            results_lines.append(f"  {i+1}. {cls:<20s} {prob:6.2%}  {bar}")
        results_lines.append("")

    results_text = "\n".join(results_lines)

    # Generate Grad-CAM heatmaps
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))

    axes[0].imshow(img_display)
    axes[0].set_title("Input Image", fontsize=11)
    axes[0].axis("off")

    for j, (display_name, model) in enumerate(models.items()):
        heatmap = generate_heatmap(model, display_name, img_tensor)
        axes[j + 1].imshow(img_display)
        axes[j + 1].imshow(heatmap, cmap="jet", alpha=0.5)
        pred_class, pred_conf = predictions[display_name]
        short_name = display_name.split("(")[0].strip()
        axes[j + 1].set_title(f"{short_name}\n{pred_class} ({pred_conf:.1%})", fontsize=10)
        axes[j + 1].axis("off")

    plt.suptitle("Grad-CAM: What does each architecture look at?", fontsize=13, y=1.02)
    plt.tight_layout()

    return results_text, fig


def create_demo():
    """Build and return the Gradio interface."""
    with gr.Blocks(title="CNN vs ViT vs Mamba — Architecture Benchmark") as demo:
        gr.Markdown(
            """
            # 🔬 CNN vs Vision Transformer vs Vision Mamba
            ### Three Generations of Visual Architectures — Side-by-Side Comparison

            Upload an image to see how **ResNet (CNN)**, **DeiT (ViT)**, and **Vim (Mamba)**
            classify it, and visualize what each model attends to via **Grad-CAM**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Image")
                predict_btn = gr.Button("🚀 Classify with All Three Models", variant="primary")

            with gr.Column(scale=2):
                output_text = gr.Textbox(label="Prediction Results (Top-5 per model)", lines=20)
                output_plot = gr.Plot(label="Grad-CAM Heatmap Comparison")

        predict_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[output_text, output_plot],
        )

        gr.Markdown(
            """
            ---
            **Architecture Legend:**
            - 🔵 **CNN (ResNet-18)**: Local convolutions → hierarchical feature extraction
            - 🟠 **ViT (DeiT-Tiny)**: Self-Attention over image patches → global context
            - 🟢 **Mamba (Vim-Tiny)**: Selective State Space Model → linear-time sequence modeling
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False)

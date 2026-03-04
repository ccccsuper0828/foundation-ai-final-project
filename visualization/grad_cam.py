"""
Grad-CAM visualization for CNN, ViT, and Mamba models.
Generates heatmap overlays showing what each model attends to.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Works with any model by hooking into a specified target layer.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) input image.
            target_class: Target class index (None = predicted class).

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Compute Grad-CAM
        gradients = self.gradients  # (1, C, ...)
        activations = self.activations  # (1, C, ...)

        if gradients.dim() == 4:
            # CNN: (1, C, H, W) → spatial Grad-CAM
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
        elif gradients.dim() == 3:
            # Transformer / Mamba: (1, N_tokens, D) → reshape to spatial
            weights = gradients.mean(dim=1, keepdim=True)  # (1, 1, D)
            cam = (weights * activations).sum(dim=-1)  # (1, N_tokens)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            # Reshape to 2D (assume square grid of patches)
            n_patches = cam.shape[0]
            side = int(np.sqrt(n_patches))
            if side * side == n_patches:
                cam = cam.reshape(side, side)
            else:
                cam = cam.reshape(1, -1)  # fallback
        else:
            cam = gradients.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


def get_target_layer(model, model_name: str):
    """Get the appropriate target layer for Grad-CAM based on model family."""
    if "resnet" in model_name:
        return model.layer4[-1]  # last residual block
    elif "convnext" in model_name:
        return model.stages[-1]  # last stage
    elif "deit" in model_name or "vit" in model_name:
        return model.blocks[-1].norm1  # last transformer block norm
    elif "vim" in model_name:
        return model.blocks[-1]  # last mamba block
    else:
        raise ValueError(f"No target layer defined for {model_name}")


def plot_gradcam_comparison(
    images,
    models_dict,
    device,
    class_names=None,
    save_path=None,
):
    """
    Plot Grad-CAM heatmaps side-by-side for multiple models.

    Args:
        images: List of (1, C, H, W) tensors.
        models_dict: dict of {model_name: (model, target_layer)}.
        device: torch device.
        class_names: Optional list of class names.
        save_path: Path to save the figure.
    """
    n_images = len(images)
    n_models = len(models_dict)

    fig, axes = plt.subplots(n_images, n_models + 1, figsize=(4 * (n_models + 1), 4 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]

    for i, img_tensor in enumerate(images):
        img_tensor = img_tensor.to(device)

        # Show original image
        img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original", fontsize=12)
        axes[i, 0].axis("off")

        for j, (name, (model, target_layer)) in enumerate(models_dict.items()):
            model = model.to(device)
            cam_gen = GradCAM(model, target_layer)
            heatmap = cam_gen.generate(img_tensor)

            # Resize heatmap to image size
            if heatmap.ndim == 2:
                from PIL import Image
                heatmap_resized = np.array(
                    Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                        (img_np.shape[1], img_np.shape[0]), Image.BILINEAR
                    )
                ) / 255.0
            else:
                heatmap_resized = heatmap

            # Overlay
            axes[i, j + 1].imshow(img_np)
            axes[i, j + 1].imshow(heatmap_resized, cmap="jet", alpha=0.5)

            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()
                conf = F.softmax(output, dim=1)[0, pred].item()

            label = class_names[pred] if class_names else str(pred)
            axes[i, j + 1].set_title(f"{name}\n{label} ({conf:.1%})", fontsize=10)
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grad-CAM figure saved to {save_path}")
    plt.close(fig)
    return fig

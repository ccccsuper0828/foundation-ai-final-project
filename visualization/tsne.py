"""
t-SNE visualization of feature spaces from different model architectures.
Shows how each model organizes its learned representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, loader, device, max_samples=2000):
    """
    Extract features from the penultimate layer of a model.

    Args:
        model: Trained model.
        loader: DataLoader.
        device: torch device.
        max_samples: Max number of samples to extract.

    Returns:
        features: (N, D) numpy array.
        labels: (N,) numpy array.
    """
    model.eval()
    all_features = []
    all_labels = []
    total = 0

    # Try to get features from the model
    # For timm models, use forward_features
    # For VisionMamba, use get_features
    use_forward_features = hasattr(model, "forward_features")
    use_get_features = hasattr(model, "get_features")

    for images, targets in tqdm(loader, desc="Extracting features", leave=False):
        if total >= max_samples:
            break

        images = images.to(device)

        if use_get_features:
            feats = model.get_features(images)
        elif use_forward_features:
            feats = model.forward_features(images)
            if feats.dim() == 3:
                feats = feats.mean(dim=1)  # (B, N, D) → (B, D) for ViT
        else:
            # Fallback: hook into the last layer before head
            feats = _hook_features(model, images)

        all_features.append(feats.cpu().numpy())
        all_labels.append(targets.numpy())
        total += images.size(0)

    features = np.concatenate(all_features, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    return features, labels


def _hook_features(model, images):
    """Fallback feature extraction using a forward hook on the avgpool or norm layer."""
    features = {}

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            features["out"] = output

    # Try common layer names
    for name in ["global_pool", "avgpool", "norm", "head_drop"]:
        layer = getattr(model, name, None)
        if layer is not None:
            handle = layer.register_forward_hook(hook_fn)
            model(images)
            handle.remove()
            if "out" in features:
                out = features["out"]
                if out.dim() > 2:
                    out = out.flatten(1)
                return out

    # Last resort: just use model output
    return model(images)


def plot_tsne_comparison(
    models_dict: dict,
    loader,
    device,
    max_samples=1500,
    n_classes_to_show=10,
    save_path=None,
):
    """
    Plot t-SNE visualizations side-by-side for multiple models.

    Args:
        models_dict: {model_name: model}
        loader: Test DataLoader.
        device: torch device.
        max_samples: Number of samples to use.
        n_classes_to_show: Number of classes to color (for clarity).
        save_path: Path to save figure.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("tab10", n_classes_to_show)

    for idx, (model_name, model) in enumerate(models_dict.items()):
        features, labels = extract_features(model, loader, device, max_samples)

        # Only show first N classes for visual clarity
        mask = labels < n_classes_to_show
        features_sub = features[mask]
        labels_sub = labels[mask]

        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings = tsne.fit_transform(features_sub)

        # Plot
        scatter = axes[idx].scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=labels_sub, cmap=cmap, s=8, alpha=0.6,
        )
        axes[idx].set_title(model_name.replace("_patch16_224", ""), fontsize=13)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.suptitle("t-SNE Feature Space Comparison", fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig

"""
Unified model builder — pass a model name, get back a nn.Module.
"""

from models.cnn_models import build_resnet18, build_resnet50, build_convnext_tiny
from models.vit_models import build_deit_tiny, build_deit_small
from models.mamba_models import build_vim_tiny, build_vim_small


MODEL_REGISTRY = {
    # CNN family
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
    "convnext_tiny": build_convnext_tiny,
    # ViT family
    "deit_tiny_patch16_224": build_deit_tiny,
    "deit_small_patch16_224": build_deit_small,
    # Mamba family
    "vim_tiny": build_vim_tiny,
    "vim_small": build_vim_small,
}

# Map each model to its architecture family
MODEL_FAMILIES = {
    "resnet18": "cnn",
    "resnet50": "cnn",
    "convnext_tiny": "cnn",
    "deit_tiny_patch16_224": "vit",
    "deit_small_patch16_224": "vit",
    "vim_tiny": "mamba",
    "vim_small": "mamba",
}


def build_model(model_name: str, num_classes: int = 100, pretrained: bool = False, **kwargs):
    """
    Build a model by name.

    Args:
        model_name: One of the keys in MODEL_REGISTRY.
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained weights (timm models only).

    Returns:
        nn.Module
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](num_classes=num_classes, pretrained=pretrained, **kwargs)


def get_model_family(model_name: str) -> str:
    """Return 'cnn', 'vit', or 'mamba'."""
    return MODEL_FAMILIES.get(model_name, "unknown")


def list_models():
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

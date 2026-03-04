"""
Vision Transformer model family: DeiT-Tiny, DeiT-Small.
Built via `timm` for consistency with CNN baselines.
"""

import timm


def build_deit_tiny(num_classes: int = 100, pretrained: bool = False):
    """DeiT-Tiny (~5.7M params) — Data-efficient Image Transformer."""
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def build_deit_small(num_classes: int = 100, pretrained: bool = False):
    """DeiT-Small (~22M params)."""
    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model

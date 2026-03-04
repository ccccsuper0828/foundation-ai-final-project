"""
CNN model family: ResNet-18, ResNet-50, ConvNeXt-Tiny.
All models are built via the `timm` library for consistency.
"""

import timm


def build_resnet18(num_classes: int = 100, pretrained: bool = False):
    """ResNet-18 (~11.2M params)."""
    model = timm.create_model(
        "resnet18",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def build_resnet50(num_classes: int = 100, pretrained: bool = False):
    """ResNet-50 (~25.6M params)."""
    model = timm.create_model(
        "resnet50",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def build_convnext_tiny(num_classes: int = 100, pretrained: bool = False):
    """ConvNeXt-Tiny (~28.6M params) — a 'modernized' CNN (2022)."""
    model = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model

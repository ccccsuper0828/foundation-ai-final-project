"""
Optimizer and scheduler utilities.
"""

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def build_optimizer(model, lr: float = 1e-3, weight_decay: float = 0.05):
    """AdamW optimizer with weight decay applied only to non-bias, non-norm params."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr, betas=(0.9, 0.999))


def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
    """Cosine annealing with linear warmup."""

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        progress = (current_epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

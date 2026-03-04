"""
Robustness evaluation: FGSM adversarial attack and common corruptions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def fgsm_attack(model, images, targets, epsilon, device):
    """
    Fast Gradient Sign Method (FGSM) attack.

    Args:
        model: Trained model.
        images: Input images (B, C, H, W).
        targets: True labels (B,).
        epsilon: Attack strength (0 = no attack).
        device: torch device.

    Returns:
        Perturbed images.
    """
    images = images.clone().detach().to(device).requires_grad_(True)
    targets = targets.to(device)

    outputs = model(images)
    loss = F.cross_entropy(outputs, targets)
    model.zero_grad()
    loss.backward()

    # Create perturbation
    perturbed = images + epsilon * images.grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)  # keep in valid range

    return perturbed.detach()


@torch.no_grad()
def evaluate_fgsm_robustness(model, loader, device, epsilons=(0, 0.01, 0.02, 0.05, 0.1)):
    """
    Evaluate model accuracy under FGSM attacks at different epsilon levels.

    Returns:
        dict: {epsilon: accuracy}
    """
    results = {}

    for eps in epsilons:
        correct = 0
        total = 0

        for images, targets in tqdm(loader, desc=f"FGSM eps={eps}", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            if eps > 0:
                # Need gradients for attack
                model.eval()
                images_adv = fgsm_attack(model, images, targets, eps, device)
                outputs = model(images_adv)
            else:
                outputs = model(images)

            _, preds = outputs.max(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        acc = 100.0 * correct / total
        results[eps] = acc
        print(f"  FGSM eps={eps:.3f}: {acc:.2f}%")

    return results


def add_gaussian_noise(images, severity=1):
    """Add Gaussian noise to images (corruption test)."""
    sigma = [0.04, 0.06, 0.08, 0.10, 0.12][severity - 1]
    noise = torch.randn_like(images) * sigma
    return torch.clamp(images + noise, 0, 1)


def add_gaussian_blur(images, severity=1):
    """Apply Gaussian blur (corruption test)."""
    kernel_sizes = [3, 5, 7, 9, 11]
    k = kernel_sizes[severity - 1]
    # Simple box blur approximation
    blur = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
    return blur(images)


@torch.no_grad()
def evaluate_corruptions(model, loader, device, severities=(1, 3, 5)):
    """
    Evaluate model accuracy under common image corruptions.

    Returns:
        dict: {corruption_type: {severity: accuracy}}
    """
    corruption_fns = {
        "gaussian_noise": add_gaussian_noise,
        "gaussian_blur": add_gaussian_blur,
    }

    results = {}
    model.eval()

    for name, fn in corruption_fns.items():
        results[name] = {}
        for severity in severities:
            correct = 0
            total = 0

            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                corrupted = fn(images, severity=severity)
                outputs = model(corrupted)
                _, preds = outputs.max(dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

            acc = 100.0 * correct / total
            results[name][severity] = acc
            print(f"  {name} (severity={severity}): {acc:.2f}%")

    return results

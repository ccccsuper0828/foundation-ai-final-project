"""
Core evaluation metrics: accuracy, per-class accuracy, confusion matrix.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


@torch.no_grad()
def compute_accuracy(model, loader, device, topk=(1, 5)):
    """Compute Top-1 and Top-5 accuracy on a dataloader."""
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)

        _, preds = outputs.topk(max(topk), dim=1, largest=True, sorted=True)
        preds_t = preds.t()
        correct_mask = preds_t.eq(targets.view(1, -1).expand_as(preds_t))

        for k in topk:
            correct[k] += correct_mask[:k].reshape(-1).float().sum(0).item()

        total += targets.size(0)
        all_preds.extend(preds[:, 0].cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    acc = {f"top{k}": 100.0 * correct[k] / total for k in topk}
    return acc, np.array(all_preds), np.array(all_targets)


def compute_confusion_matrix(all_preds, all_targets, class_names=None):
    """Return confusion matrix and per-class report."""
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return cm, report

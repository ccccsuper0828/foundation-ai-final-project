"""
Unified training loop for all model architectures.
Supports: mixed precision, gradient clipping, label smoothing, model checkpointing.

Usage:
    python -m training.trainer --config resnet18
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_loader import load_config, Config
from data.dataset import get_cifar100_dataloaders
from models.build_model import build_model, count_parameters, get_model_family
from training.optimizer import build_optimizer, build_scheduler


class AverageMeter:
    """Track running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch, config,
):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward
        use_amp = config.training.mixed_precision and device.type == "cuda"
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward
        if use_amp:
            scaler.scale(loss).backward()
            if config.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.training.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()

        # Metrics
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1, batch_size)
        acc5_meter.update(acc5, batch_size)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc1=f"{acc1_meter.avg:.2f}%")

    return {
        "train_loss": loss_meter.avg,
        "train_acc1": acc1_meter.avg,
        "train_acc5": acc5_meter.avg,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, config):
    """Evaluate on val/test set."""
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_amp = config.training.mixed_precision and device.type == "cuda"
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1, batch_size)
        acc5_meter.update(acc5, batch_size)

    return {
        "val_loss": loss_meter.avg,
        "val_acc1": acc1_meter.avg,
        "val_acc5": acc5_meter.avg,
    }


def train(config_name: str):
    """Full training pipeline for one model."""
    # Load config
    cfg_dict = load_config(config_name)
    config = Config(cfg_dict)

    model_name = config.model.name
    family = get_model_family(model_name)
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  |  Family: {family}")
    print(f"{'='*60}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Data
    train_loader, val_loader, test_loader, num_classes = get_cifar100_dataloaders(
        data_dir=config.data.data_dir,
        image_size=config.data.image_size,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
    )
    print(f"  Dataset: CIFAR-100 | Train: {len(train_loader.dataset)} | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")

    # Model
    model = build_model(model_name, num_classes=num_classes, pretrained=config.model.pretrained)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)

    # Optimizer + Scheduler
    optimizer = build_optimizer(model, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=config.training.epochs, warmup_epochs=config.training.warmup_epochs)
    scaler = GradScaler(enabled=config.training.mixed_precision and device.type == "cuda")

    # Output directory
    save_dir = Path(config.logging.save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_acc1 = 0.0
    history = []

    print(f"\n  Starting training for {config.training.epochs} epochs...\n")
    total_start = time.time()

    for epoch in range(1, config.training.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, config)
        val_metrics = evaluate(model, val_loader, criterion, device, config)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        log = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
            **train_metrics,
            **val_metrics,
        }
        history.append(log)

        is_best = val_metrics["val_acc1"] > best_acc1
        if is_best:
            best_acc1 = val_metrics["val_acc1"]
            torch.save(model.state_dict(), save_dir / "best.pth")

        print(
            f"  Epoch {epoch:3d}/{config.training.epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Train Acc@1: {train_metrics['train_acc1']:.2f}% | "
            f"Val Acc@1: {val_metrics['val_acc1']:.2f}% | "
            f"Val Acc@5: {val_metrics['val_acc5']:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {epoch_time:.1f}s"
            f"{'  ★ BEST' if is_best else ''}"
        )

    total_time = time.time() - total_start
    print(f"\n  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best Val Acc@1: {best_acc1:.2f}%")

    # Test evaluation
    model.load_state_dict(torch.load(save_dir / "best.pth", weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device, config)
    print(f"  Test Acc@1: {test_metrics['val_acc1']:.2f}% | Test Acc@5: {test_metrics['val_acc5']:.2f}%")

    # Save results
    results = {
        "model_name": model_name,
        "family": family,
        "num_params": n_params,
        "best_val_acc1": best_acc1,
        "test_acc1": test_metrics["val_acc1"],
        "test_acc5": test_metrics["val_acc5"],
        "total_train_time": total_time,
        "epochs": config.training.epochs,
        "history": history,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {save_dir}/results.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train a vision model")
    parser.add_argument("--config", type=str, required=True, help="Config name (e.g. resnet18)")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

"""
Unified data loading for CIFAR-100 (and extensible to Tiny-ImageNet).
Provides consistent train/val/test splits and augmentation across all models.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path


# CIFAR-100 channel statistics
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# ImageNet statistics (used when resizing to 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(image_size: int = 224):
    """Standard training augmentation matching timm defaults."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=image_size // 8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def get_val_transform(image_size: int = 224):
    """Deterministic validation/test transform."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_cifar100_dataloaders(
    data_dir: str = "./data/datasets",
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1,
):
    """
    Returns train, val, test DataLoaders for CIFAR-100.

    Args:
        data_dir: Where to download/store the dataset.
        image_size: Target image size (all models use 224).
        batch_size: Batch size for all loaders.
        num_workers: DataLoader workers.
        val_split: Fraction of training set to hold out for validation.

    Returns:
        (train_loader, val_loader, test_loader, num_classes)
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)

    # Download CIFAR-100
    full_train_dataset = datasets.CIFAR100(
        root=str(data_path), train=True, download=True, transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=str(data_path), train=False, download=True, transform=val_transform,
    )

    # Split training into train + validation
    num_train = len(full_train_dataset)
    num_val = int(num_train * val_split)
    num_train = num_train - num_val

    train_dataset, val_dataset_raw = random_split(
        full_train_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Val split should use val_transform (no augmentation)
    # We wrap it with the val transform via a simple wrapper
    val_dataset = ValSubset(val_dataset_raw, val_transform, full_train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_classes = 100
    return train_loader, val_loader, test_loader, num_classes


class ValSubset(torch.utils.data.Dataset):
    """Wraps a Subset to apply a different transform (val instead of train)."""

    def __init__(self, subset, transform, original_dataset):
        self.subset = subset
        self.transform = transform
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get the original index in the full dataset
        original_idx = self.subset.indices[idx]
        # Access the raw PIL image and label from the full dataset
        img, label = self.original_dataset.data[original_idx], self.original_dataset.targets[original_idx]
        # CIFAR images are numpy arrays, convert to PIL for transforms
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_class_names(dataset_name: str = "cifar100", data_dir: str | None = None, download: bool = True):
    """Return human-readable class names.

    Args:
        dataset_name: Dataset identifier.
        data_dir: Optional dataset root. If None, use project-root/data/datasets.
        download: Whether to auto-download if dataset files are missing.
    """
    if dataset_name == "cifar100":
        # Resolve dataset path relative to project root, not current working directory.
        if data_dir is None:
            data_path = Path(__file__).resolve().parent.parent / "data" / "datasets"
        else:
            data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        ds = datasets.CIFAR100(root=str(data_path), train=False, download=download)
        return ds.classes
    return [str(i) for i in range(100)]

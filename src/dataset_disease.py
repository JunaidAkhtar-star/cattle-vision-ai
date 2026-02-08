# src/dataset_disease.py

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_disease import (
    IMG_SIZE,
    MEAN,
    STD,
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    BATCH_SIZE,
)


# -------------------------
# Transforms
# -------------------------

def get_disease_transforms():
    """Get image transformations for disease dataset (classification)."""

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )

    return train_transform, valid_transform, test_transform


# -------------------------
# Dataloaders
# -------------------------

def get_disease_dataloaders(batch_size: int = None):
    """
    Get training, validation, and test dataloaders using folder-based dataset:

    data/disease_dataset/train/
        healthy/, JERSEY/, lumpy/, sahiwal/
    data/disease_dataset/valid/
        healthy/, JERSEY/, lumpy/, sahiwal/
    data/disease_dataset/test/
        healthy/, JERSEY/, lumpy/, sahiwal/
    """

    if batch_size is None:
        batch_size = BATCH_SIZE

    train_transform, valid_transform, test_transform = get_disease_transforms()

    # ImageFolder expects root/train/<class_name>/image.jpg
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"TRAIN_DIR not found: {TRAIN_DIR}")
    if not VALID_DIR.exists():
        raise FileNotFoundError(f"VALID_DIR not found: {VALID_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=str(VALID_DIR), transform=valid_transform)
    test_dataset  = datasets.ImageFolder(root=str(TEST_DIR),  transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Class names as discovered by ImageFolder (alphabetical)
    class_names = train_dataset.classes  # e.g. ['JERSEY', 'healthy', 'lumpy', 'sahiwal']

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Discovered classes (ImageFolder): {class_names}")

    # Optional: show training distribution
    counts = {name: 0 for name in class_names}
    for _, label in train_dataset.samples:
        label_name = class_names[label]
        counts[label_name] = counts.get(label_name, 0) + 1

    print("\nTraining set distribution:")
    for disease, count in sorted(counts.items()):
        print(f"  {disease}: {count}")

    return train_loader, valid_loader, test_loader, class_names

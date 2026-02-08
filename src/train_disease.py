# src/train_disease.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import copy
from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_disease import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    DEVICE,
    DISEASE_MODEL_PATH,      # models/best_disease_model.pth
    CHECKPOINT_PATH,
    EARLY_STOPPING_PATIENCE,
)
from src.dataset_disease import get_disease_dataloaders
from torchvision import models
from torchvision.models import ResNet18_Weights


def create_disease_model(num_classes: int) -> nn.Module:
    """Create ResNet model for disease classification."""
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_disease_model():
    """Train the disease classification model."""
    print("Starting disease model training...")

    # Get dataloaders and discovered class names
    train_loader, val_loader, test_loader, class_names = get_disease_dataloaders(
        batch_size=BATCH_SIZE
    )
    num_classes = len(class_names)

    # Create model
    model = create_disease_model(num_classes=num_classes)
    model = model.to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Early stopping variables
    best_loss = float("inf")
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Classes: {class_names}")
    print(f"Training on   {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    print(f"Testing on    {len(test_loader.dataset)} samples")
    print(f"Number of epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 50)

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # --------------------
        # Training phase
        # --------------------
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)              # [batch, num_classes]
                loss = criterion(outputs, labels)    # scalar loss

                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # --------------------
        # Validation phase
        # --------------------
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # --------------------
        # Check for best model
        # --------------------
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "class_names": class_names,
                "num_classes": num_classes,
            }
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"  -> Saved checkpoint (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        scheduler.step()

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save final/best model for inference
    DISEASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
        },
        DISEASE_MODEL_PATH,
    )

    print("-" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best model saved to: {DISEASE_MODEL_PATH}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    return model


if __name__ == "__main__":
    train_disease_model()

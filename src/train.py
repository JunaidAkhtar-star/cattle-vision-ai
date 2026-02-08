import os
import torch
from torch import nn, optim
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from dataset import CattleBreedDataset

# Create 'models' directory if it doesn't exist to avoid save error
if not os.path.exists("models"):
    os.makedirs("models")

# Parameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# Data transforms (resize, tensor, normalization)
data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Create datasets
train_dataset = CattleBreedDataset(
    "data/train/images",
    "data/train/_classes.csv",
    transform=data_transform,
)
valid_dataset = CattleBreedDataset(
    "data/valid/images",
    "data/valid/_classes.csv",
    transform=data_transform,
)

# DataLoaders with num_workers=0 for stable loading
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# Number of classes from dataset (now includes Holstein Fresian + Red Dane)
num_classes = len(train_dataset.breed_columns)

# Load pretrained model using new weights argument
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    print(f"Validation Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "models/best_model.pth")
print("Saved trained model to models/best_model.pth")

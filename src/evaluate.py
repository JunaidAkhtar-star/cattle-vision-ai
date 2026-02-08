import torch
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from dataset import CattleBreedDataset

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define test data transform (same as train transform)
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create test dataset and DataLoader
test_dataset = CattleBreedDataset('data/test/images', 'data/test/_classes.csv', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
weights = None  # No pretrained weights because loading own weights
model = resnet18(weights=weights)
num_classes = len(test_dataset.breed_columns)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('models/best_model.pth'))
model = model.to(device)
model.eval()

# Variables for predictions and labels
all_preds = []
all_labels = []
misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Collect misclassified samples
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified.append((images[i].cpu(), labels[i].cpu().item(), preds[i].cpu().item()))

# Calculate accuracy
correct = sum([1 for true, pred in zip(all_labels, all_preds) if true == pred])
total = len(all_labels)
print(f'Test Accuracy: {correct / total:.4f}')

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Classification report: precision, recall, F1-score per class
report = classification_report(all_labels, all_preds, target_names=test_dataset.breed_columns)
print("Classification Report:\n", report)

# Plot the confusion matrix heatmap
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_dataset.breed_columns,
            yticklabels=test_dataset.breed_columns)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Cattle Breed Classification')
plt.show()

# Visualize misclassified images
if misclassified:
    fig, axes = plt.subplots(2, 5, figsize=(16,7))
    axes = axes.flatten()
    for i, (img, true_label, pred_label) in enumerate(misclassified[:10]):
        # Unnormalize image for display
        img = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"True: {test_dataset.breed_columns[true_label]}\nPred: {test_dataset.breed_columns[pred_label]}")
        axes[i].axis('off')
    plt.suptitle('Sample Misclassified Images')
    plt.tight_layout()
    plt.show()
else:
    print("No misclassified images found!")



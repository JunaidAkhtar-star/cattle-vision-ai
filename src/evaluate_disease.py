import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_disease_model(
    model,
    dataloader: DataLoader,
    class_names,
) -> Dict[str, Any]:
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    return metrics


if __name__ == "__main__":
    # Example usage; adjust paths and class list to your project
    from src.dataset_disease import DiseaseDatasetCSV, get_disease_transforms
    from src.train_disease import create_disease_model
    from torch.utils.data import DataLoader
    from pathlib import Path

    class_names = [
        "Foot Infection",
        "Lumpy Skin Disease",
        "Mouth Infection",
    ]

    # Assuming test CSV exists; if not, create it using utils_disease.py
    test_csv = Path("data/disease_dataset/test/_classes.csv")
    test_dir = Path("data/disease_dataset/test")

    if not test_csv.exists():
        print(f"Test CSV not found: {test_csv}")
        print("Please create test CSV using src.utils_disease.create_disease_csv_manual")
        exit(1)

    _, valid_transform = get_disease_transforms()

    test_dataset = DiseaseDatasetCSV(
        images_dir=str(test_dir),
        csv_path=str(test_csv),
        transform=valid_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
    )

    model = create_disease_model(num_classes=len(class_names))
    checkpoint = torch.load("models/best_disease_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    evaluate_disease_model(model, test_loader, class_names)

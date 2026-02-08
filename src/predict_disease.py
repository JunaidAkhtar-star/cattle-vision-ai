# src/predict_disease.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from pathlib import Path
from typing import Dict, List

from src.config_disease import (
    DISEASE_MODEL_PATH,
    IMG_SIZE,
    MEAN,
    STD,
    DEVICE,
)


class DiseasePredictor:
    def __init__(
        self,
        model_path: Path = DISEASE_MODEL_PATH,
        class_names: List[str] = None,
        image_size: int = IMG_SIZE,
    ):
        self.model_path = Path(model_path)
        self.image_size = image_size

        self.model, self.class_names = self._load_model_and_classes(class_names)
        self.transform = self._build_transform()

    def _build_model(self, num_classes: int) -> nn.Module:
        """Create same ResNet architecture as in training."""
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _load_model_and_classes(self, class_names_override: List[str] = None):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)

        # class_names and num_classes from training
        saved_class_names = checkpoint.get("class_names", None)
        num_classes = checkpoint.get("num_classes", None)

        if class_names_override is not None:
            class_names = class_names_override
        elif saved_class_names is not None:
            class_names = saved_class_names
        else:
            raise ValueError(
                "No class_names found in checkpoint and no override provided."
            )

        if num_classes is None:
            num_classes = len(class_names)

        model = self._build_model(num_classes=num_classes)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(DEVICE)
        model.eval()

        return model, class_names

    def _build_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    def predict_image(self, image_path: str, confidence_threshold: float = 0.6) -> Dict:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        confidence_score = confidence.item()
        predicted_disease = self.class_names[pred_idx.item()]
        is_healthy = predicted_disease.lower() == "healthy" or confidence_score < confidence_threshold

        all_predictions = {
            self.class_names[i]: float(probs[0, i].item())
            for i in range(len(self.class_names))
        }

        return {
            "image_path": str(image_path),
            "disease": "Healthy" if is_healthy else predicted_disease,
            "confidence": round(confidence_score * 100, 2),
            "healthy": is_healthy,
            "all_predictions": all_predictions,
        }


if __name__ == "__main__":
    # Example usage
    predictor = DiseasePredictor()
    result = predictor.predict_image("data/disease_dataset/test/healthy/some_image.jpg")
    print(result)

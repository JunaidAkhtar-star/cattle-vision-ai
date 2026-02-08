"""
Combined Breed + Disease Predictor
Integrates breed prediction with disease detection
"""

from typing import Dict, List
import logging

from src.predict_disease import DiseasePredictor

logger = logging.getLogger(__name__)


class CombinedPredictor:
    """
    Combines breed and disease prediction.
    Keeps breed model intact while adding disease detection.
    """

    def __init__(self, breed_predictor, disease_predictor: DiseasePredictor, breed_names: List[str]):
        """
        Initialize combined predictor.

        Args:
            breed_predictor: function(image_path) -> List[float] (probabilities over breeds)
            disease_predictor: DiseasePredictor instance
            breed_names: list of breed names in the SAME order as model outputs
        """
        self.breed_predictor = breed_predictor
        self.disease_predictor = disease_predictor
        self.breed_names = breed_names

    def predict_all(self, image_path: str) -> Dict:
        """
        Predict breed AND disease for a given image path.
        """
        # Step 1: Breed prediction
        breed_probabilities = self.breed_predictor(image_path)  # list of length len(self.breed_names)

        if not breed_probabilities:
            raise ValueError("Breed predictor returned an empty probability list.")

        if len(breed_probabilities) != len(self.breed_names):
            raise ValueError(
                f"Length mismatch: got {len(breed_probabilities)} breed probabilities "
                f"but {len(self.breed_names)} breed names."
            )

        # Top breed
        max_idx = max(range(len(breed_probabilities)), key=lambda i: breed_probabilities[i])
        max_prob = breed_probabilities[max_idx]
        top_breed = self.breed_names[max_idx]

        # Step 2: Disease prediction
        disease_result = self.disease_predictor.predict_image(image_path)

        # Step 3: Combine results
        combined_result = {
            # Breed info
            "breed": top_breed,
            "breed_confidence": round(max_prob * 100, 2),
            "breed_probabilities": breed_probabilities,
            "species": "Buffalo" if top_breed in ["Bhadawari", "Banni"] else "Cattle",

            # Disease info
            "disease": disease_result.get("disease", "Unknown"),
            "disease_confidence": disease_result.get("confidence", 0.0),
            "health_status": "Healthy" if disease_result.get("healthy", False) else "Unhealthy",

            # Metadata
            "raw_disease_predictions": disease_result.get("all_predictions", {}),
        }

        return combined_result

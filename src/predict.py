"""
Combined Breed and Disease Prediction Module
Maintains existing breed prediction while adding disease detection
Processes images from data/train/images, data/valid/images, data/test/images
Author: Digital Dominators
"""

import os
from collections import Counter
from PIL import Image
import torch
from torchvision import transforms, models
from typing import Dict, Tuple, List
import logging
from disease_predictor import DiseasePredictor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DEVICE SETUP
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ============================================================================
# IMAGE TRANSFORMS (Your existing code - UNCHANGED)
# ============================================================================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================================
# BREED MODEL SETUP (Your existing code - UNCHANGED)
# ============================================================================
model = models.resnet18(pretrained=False)
num_classes = 12  # Number of breeds

# Modify the final layer
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load your trained model
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()
model.to(device)


# ============================================================================
# BREED NAMES (Your existing code - UNCHANGED)
# ============================================================================
breed_names = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni',
    'Bargur', 'Bhadawari', 'Brown Swiss', 'Dangi',
    'Deoni', 'Gir', 'Guernsey', 'Hallikar'
]


# ============================================================================
# BREED TRAITS DICTIONARY (Complete with all 12 breeds)
# ============================================================================
breed_traits = {
    'Alambadi': {
        'species': 'Cattle',
        'origin': 'Karnataka, India',
        'characteristics': 'Hardy, good milk production, adaptable to drought',
        'milk_yield': '4-6 liters/day',
        'color': 'Red to dark red',
        'size': 'Medium to large',
        'purpose': 'Dairy'
    },
    'Amritmahal': {
        'species': 'Cattle',
        'origin': 'Karnataka, India',
        'characteristics': 'Dual purpose (milk & draft), strong build, excellent draught',
        'milk_yield': '3-5 liters/day',
        'color': 'Gray to dark gray',
        'size': 'Large',
        'purpose': 'Dual purpose'
    },
    'Ayrshire': {
        'species': 'Cattle',
        'origin': 'Scotland',
        'characteristics': 'High milk production, refined features, good feed conversion',
        'milk_yield': '6-8 liters/day',
        'color': 'Red and white',
        'size': 'Medium',
        'purpose': 'Dairy'
    },
    'Banni': {
        'species': 'Cattle',
        'origin': 'Gujarat, India',
        'characteristics': 'Good milk production, salt-tolerant, heat resistant',
        'milk_yield': '5-7 liters/day',
        'color': 'Black with white markings',
        'size': 'Large',
        'purpose': 'Dairy'
    },
    'Bargur': {
        'species': 'Cattle',
        'origin': 'Tamil Nadu, India',
        'characteristics': 'Draught purpose, strong muscular build, good walker',
        'milk_yield': '2-3 liters/day',
        'color': 'Black',
        'size': 'Large',
        'purpose': 'Draft'
    },
    'Bhadawari': {
        'species': 'Buffalo',
        'origin': 'Madhya Pradesh, India',
        'characteristics': 'High milk production, dual purpose, excellent butterfat',
        'milk_yield': '6-8 liters/day',
        'color': 'Black',
        'size': 'Large',
        'purpose': 'Dairy'
    },
    'Brown Swiss': {
        'species': 'Cattle',
        'origin': 'Switzerland',
        'characteristics': 'High milk production, strong legs, excellent health',
        'milk_yield': '6-8 liters/day',
        'color': 'Brown',
        'size': 'Large',
        'purpose': 'Dairy'
    },
    'Dangi': {
        'species': 'Cattle',
        'origin': 'Maharashtra, India',
        'characteristics': 'Draught purpose, hardy, good walking ability, drought resistant',
        'milk_yield': '2-3 liters/day',
        'color': 'Gray',
        'size': 'Medium',
        'purpose': 'Draft'
    },
    'Deoni': {
        'species': 'Cattle',
        'origin': 'Telangana, India',
        'characteristics': 'Dual purpose, moderate milk, strong draught, heat tolerant',
        'milk_yield': '4-5 liters/day',
        'color': 'Gray and white',
        'size': 'Large',
        'purpose': 'Dual purpose'
    },
    'Gir': {
        'species': 'Cattle',
        'origin': 'Gujarat, India',
        'characteristics': 'High milk production, heat-tolerant, hump visible, excellent immune',
        'milk_yield': '6-8 liters/day',
        'color': 'Red and white',
        'size': 'Large',
        'purpose': 'Dairy'
    },
    'Guernsey': {
        'species': 'Cattle',
        'origin': 'United Kingdom',
        'characteristics': 'High butterfat milk production, refined features, good feed conversion',
        'milk_yield': '5-7 liters/day',
        'color': 'Golden and white',
        'size': 'Medium',
        'purpose': 'Dairy'
    },
    'Hallikar': {
        'species': 'Cattle',
        'origin': 'Karnataka, India',
        'characteristics': 'Draught purpose, hardy, good walking ability, strong constitution',
        'milk_yield': '2-3 liters/day',
        'color': 'Gray or black',
        'size': 'Large',
        'purpose': 'Draft'
    }
}


# ============================================================================
# DATASET FOLDERS (Your existing code - NOW PROPERLY USED)
# ============================================================================
folders = {
    'train': 'data/train/images',
    'valid': 'data/valid/images',
    'test': 'data/test/images'
}


# ============================================================================
# BREED PREDICTION FUNCTION (Your existing code - MODIFIED for export)
# ============================================================================
def predict_breed(image_path: str) -> Dict:
    """
    Predict breed from image
    
    Args:
        image_path: Path to cattle/buffalo image
        
    Returns:
        Dict with breed prediction results:
        {
            "breed": str,
            "confidence": float,
            "traits": dict,
            "species": str,
            "error": str or None
        }
    """
    try:
        if not os.path.exists(image_path):
            return {
                "breed": "Error",
                "confidence": 0.0,
                "traits": {},
                "species": "Unknown",
                "error": f"Image file not found: {image_path}"
            }
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = data_transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, 1)
        
        # Get breed name and confidence
        breed = breed_names[pred.item()]
        confidence_score = confidence.item() * 100
        
        # Get breed traits
        traits = breed_traits.get(breed, {})
        species = traits.get('species', 'Unknown')
        
        logger.info(f"Breed prediction: {breed} ({confidence_score:.2f}%) - {image_path}")
        
        return {
            "breed": breed,
            "confidence": round(confidence_score, 2),
            "traits": traits,
            "species": species,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Breed prediction error: {str(e)}"
        logger.error(error_msg)
        return {
            "breed": "Error",
            "confidence": 0.0,
            "traits": {},
            "species": "Unknown",
            "error": error_msg
        }


# ============================================================================
# COMBINED PREDICTION FUNCTION (NEW - MAIN FUNCTION)
# ============================================================================
def predict_breed_and_disease(image_path: str, disease_predictor: DiseasePredictor = None) -> Dict:
    """
    Predict BOTH breed and disease for an image

    Args:
        image_path: Path to cattle/buffalo image
        disease_predictor: Optional DiseasePredictor instance for disease detection

    Returns:
        Dict with combined prediction results:
        {
            "breed": str,
            "breed_confidence": float,
            "traits": dict,
            "species": str,
            "disease": str,
            "disease_confidence": float,
            "health_status": str,
            "error": str or None
        }
    """
    try:
        # Get breed prediction
        breed_result = predict_breed(image_path)
        if breed_result["error"]:
            return {
                "breed": "Error",
                "breed_confidence": 0.0,
                "traits": {},
                "species": "Unknown",
                "disease": "Unknown",
                "disease_confidence": 0.0,
                "health_status": "Unknown",
                "error": breed_result["error"]
            }

        # Get disease prediction if predictor provided
        if disease_predictor:
            disease_result = disease_predictor.predict(image_path)
            disease = disease_result.get("disease", "Unknown")
            disease_confidence = disease_result.get("confidence", 0.0)
            health_status = disease_result.get("status", "Unknown")
        else:
            disease = "No predictor provided"
            disease_confidence = 0.0
            health_status = "Unknown"

        logger.info(f"Combined prediction: {breed_result['breed']} ({breed_result['confidence']}%) - Disease: {disease} ({disease_confidence}%) - {image_path}")

        return {
            "breed": breed_result["breed"],
            "breed_confidence": breed_result["confidence"],
            "traits": breed_result["traits"],
            "species": breed_result["species"],
            "disease": disease,
            "disease_confidence": round(disease_confidence, 2),
            "health_status": health_status,
            "error": None
        }

    except Exception as e:
        error_msg = f"Combined prediction error: {str(e)}"
        logger.error(error_msg)
        return {
            "breed": "Error",
            "breed_confidence": 0.0,
            "traits": {},
            "species": "Unknown",
            "disease": "Error",
            "disease_confidence": 0.0,
            "health_status": "Unknown",
            "error": error_msg
        }


# ============================================================================
# MAIN EXECUTION (For testing)
# ============================================================================
if __name__ == "__main__":
    # Example usage with disease predictor
    test_image = "data/test/images/00000002_jpg.rf.cbf5fbb94199ebfc89296e0e1c1e1ed6.jpg"

    # Initialize disease predictor
    disease_predictor = DiseasePredictor()

    # Run combined prediction
    result = predict_breed_and_disease(test_image, disease_predictor)
    print("Prediction Result:")
    print(result)

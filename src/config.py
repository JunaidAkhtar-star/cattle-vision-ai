"""
Configuration file for breed and disease prediction
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """App configuration"""
    
    # Flask settings
    FLASK_ENV = "development"
    DEBUG = True
    SECRET_KEY = "your-secret-key-change-in-production"
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = "static/uploads"
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    
    # Model paths
    BREED_MODEL_PATH = "models/best_model.pth"
    
    # Roboflow API
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
    ROBOFLOW_MODEL_ID = "cattle-disease-pnjdc/3"
    ROBOFLOW_API_URL = "https://serverless.roboflow.com"
    
    # Breed configuration
    BREED_NAMES = [
        'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni',
        'Bargur', 'Bhadawari', 'Brown Swiss', 'Dangi',
        'Deoni', 'Gir', 'Guernsey', 'Hallikar'
    ]
    
    # Breed traits (keep your existing data)
    BREED_TRAITS = {
        'Alambadi': {'species': 'Cattle', 'trait': 'Hardy, good milk production'},
        'Amritmahal': {'species': 'Cattle', 'trait': 'Draft purpose, strong build'},
        # ... add all 12 breeds
    }

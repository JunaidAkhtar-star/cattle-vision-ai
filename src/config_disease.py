# src/config_disease.py

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "disease_dataset"

# Dataset paths (folder-based, no CSV)
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR  = DATA_DIR / "test"   # create this folder and put test images in it

# Model paths
MODEL_DIR = BASE_DIR / "models"
DISEASE_MODEL_PATH = MODEL_DIR / "best_disease_model.pth"

# Disease classes (folder names used by ImageFolder, order will be alphabetical)
# You can keep this list for reference / app display
DISEASE_CLASSES = ["healthy", "JERSEY", "lumpy", "sahiwal"]

# Training parameters
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Image parameters
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Device
DEVICE = "cpu"  # change to "cuda" if you have GPU

# Checkpoint path
CHECKPOINT_PATH = MODEL_DIR / "disease_checkpoint.pth"

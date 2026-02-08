# make_disease_csv.py

from src.utils_disease import create_disease_csv_manual
from src.config_disease import TRAIN_DIR, VALID_DIR, TRAIN_CSV, VALID_CSV

if __name__ == "__main__":
    print("Creating disease CSVs...")
    create_disease_csv_manual(TRAIN_DIR, TRAIN_CSV)
    create_disease_csv_manual(VALID_DIR, VALID_CSV)
    print("Done. CSV files created:")
    print(f"  Train CSV: {TRAIN_CSV}")
    print(f"  Valid CSV: {VALID_CSV}")

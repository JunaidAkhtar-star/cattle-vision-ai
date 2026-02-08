# src/utils_disease.py

import csv
from pathlib import Path
from PIL import Image

from src.config_disease import DISEASE_CLASSES


def create_disease_csv_manual(dataset_dir, csv_path):
    """
    Manually label images as one of the DISEASE_CLASSES and
    write a detection-style CSV with full-image bounding boxes.

    Output CSV columns:
    filename,width,height,class,xmin,ymin,xmax,ymax
    """
    dataset_dir = Path(dataset_dir)
    images = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))

    if not images:
        print(f"No images found in {dataset_dir}")
        return

    print(f"\nFound {len(images)} images")
    print("\nDisease classes:")
    for idx, disease in enumerate(DISEASE_CLASSES):
        print(f"  {idx}: {disease}")

    rows = []

    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] Image: {img_path.name}")

        try:
            img = Image.open(img_path)
            width, height = img.size
            print(f"  Size: {width}x{height}")
        except Exception as e:
            print(f"  (Could not load image: {e})")
            continue

        while True:
            try:
                disease_idx = int(
                    input(f"  Enter disease index (0-{len(DISEASE_CLASSES)-1}): ")
                )
                if 0 <= disease_idx < len(DISEASE_CLASSES):
                    disease_name = DISEASE_CLASSES[disease_idx]
                    row = {
                        "filename": img_path.name,
                        "width": width,
                        "height": height,
                        "class": disease_name,
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": width,
                        "ymax": height,
                    }
                    rows.append(row)
                    print(f"  ✓ Labeled as: {disease_name}")
                    break
                else:
                    print(
                        f"  Invalid index! Enter integer between 0 and {len(DISEASE_CLASSES)-1}"
                    )
            except ValueError:
                print("  Invalid input! Enter a number")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "filename",
            "width",
            "height",
            "class",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Detection CSV saved to: {csv_path}")
    return rows
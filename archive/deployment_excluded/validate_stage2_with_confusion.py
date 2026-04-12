# ml_service/validate_stage2_with_confusion.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from config.nutrients_ranges import NUTRIENT_RANGES
from app.model_loader import STAGE2_MODELS, STAGE2_FEATURES

# -----------------------------
# Helpers
# -----------------------------

def classify_value(value, ranges):
    for label, (lo, hi) in ranges.items():
        if lo <= value < hi:
            return label
    return "unknown"

# -----------------------------
# Load dataset
# -----------------------------

DATA_PATH = Path("data/soil_satellite_lab_data.csv")
df = pd.read_csv(DATA_PATH)

print(f"Loaded dataset with {len(df)} rows")

# -----------------------------
# Build input matrix
# -----------------------------

X = df[STAGE2_FEATURES]

# -----------------------------
# Validation per nutrient
# -----------------------------

for nutrient, model in STAGE2_MODELS.items():
    if nutrient not in NUTRIENT_RANGES:
        print(f"\n⚠ Skipping {nutrient} (no ICAR ranges)")
        continue

    print("\n" + "="*50)
    print(f" Nutrient: {nutrient}")
    print("="*50)

    # True numeric values from dataset
    y_true_numeric = df[nutrient]

    # Predict numeric values
    y_pred_numeric = model.predict(X)

    # Convert to ICAR classes
    y_true_class = y_true_numeric.apply(
        lambda v: classify_value(v, NUTRIENT_RANGES[nutrient])
    )

    y_pred_class = pd.Series(y_pred_numeric).apply(
        lambda v: classify_value(v, NUTRIENT_RANGES[nutrient])
    )

    # Remove unknowns (safety)
    mask = (y_true_class != "unknown") & (y_pred_class != "unknown")
    y_true_class = y_true_class[mask]
    y_pred_class = y_pred_class[mask]

    # Confusion matrix
    labels = sorted(NUTRIENT_RANGES[nutrient].keys())
    cm = confusion_matrix(y_true_class, y_pred_class, labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion Matrix:")
    print(cm_df)

    # Accuracy
    accuracy = (y_true_class == y_pred_class).mean()
    print(f"\nClass Accuracy: {accuracy:.2%}")

    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_true_class, y_pred_class))

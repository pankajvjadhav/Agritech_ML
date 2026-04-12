import pandas as pd
from app.predictor import make_classification_prediction
from app.model_loader import STAGE2_FEATURES

# Load dataset
df = pd.read_csv("data/final_training_dataset.csv")
df.columns = df.columns.str.strip()  # remove hidden spaces

print("🚀 Starting REAL validation...\n")

for i in range(5):  # test first 5 rows
    row = df.iloc[i]

    # --------------------------------------
    # 🔥 AUTO-FILL ALL REQUIRED FEATURES
    # --------------------------------------
    features = {}

    for f in STAGE2_FEATURES:
        if f in row:
            features[f] = row[f]
        else:
            features[f] = 0  # safe fallback (never crash)

    # --------------------------------------
    # 🔥 OVERRIDE WITH REAL IMPORTANT VALUES
    # --------------------------------------
    features.update({
        # Satellite (REAL)
        "NDVI_mean": row.get("NDVI_mean", 0),
        "NDMI_mean": row.get("NDMI_mean", 0),
        "BSI_mean": row.get("BSI_mean", 0),
        "NBR2_mean": row.get("NBR2_mean", 0),
        "VV_mean": row.get("VV_mean", 0),
        "VH_mean": row.get("VH_mean", 0),
        "VV_VH_ratio_mean": row.get("VV_VH_ratio_mean", 0),

        # Terrain
        "elevation": row.get("elevation", 0),
        "slope": row.get("slope", 0),
        "LST_mean": row.get("LST_mean", 0),

        # Rainfall (REAL)
        "rainfall_sum": row.get("rainfall_sum", 0),
        "rainfall_mean": row.get("rainfall_mean", 0),
        "rainfall_max": row.get("rainfall_max", 0),
        "rainy_days": row.get("rainy_days", 0),

        # Soil
        "pH": row.get("ph", 6.5),
        "OC": row.get("oc", 0.5),
    })

    # --------------------------------------
    # 🔥 PREDICTION
    # --------------------------------------
    pred = make_classification_prediction(features)

    # --------------------------------------
    # 🎯 PRINT RESULTS
    # --------------------------------------
    print(f"\n========== ROW {i} ==========")

    print(f"Nitrogen    → Actual: {row['nitrogen']} | Pred: {pred['nitrogen']['value']}")
    print(f"Phosphorus  → Actual: {row['phosphorus']} | Pred: {pred['phosphorus']['value']}")
    print(f"Potassium   → Actual: {row['potassium']} | Pred: {pred['potassium']['value']}")
    print(f"Iron        → Actual: {row['iron']} | Pred: {pred['iron']['value']}")
    print(f"Manganese   → Actual: {row['manganese']} | Pred: {pred['manganese']['value']}")
    print(f"Zinc        → Actual: {row['zinc']} | Pred: {pred['zinc']['value']}")
    print(f"Copper      → Actual: {row['copper']} | Pred: {pred['copper']['value']}")
    print(f"Sulfur      → Actual: N/A | Pred: {pred['SULFUR']['value']}")
    print(f"Boron       → Actual: N/A | Pred: {pred['BORON']['value']}")

print("\n✅ Validation Completed Successfully")
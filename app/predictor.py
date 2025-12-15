import numpy as np
import pandas as pd

NUTRIENT_ORDER = ["N","P","K","OC","pH","EC","S","Fe","Zn","Cu","B","Mn"]

# Feature order must match the order used when training the model in train_model.py
FEATURE_ORDER = [
    "ndvi_mean_90d",
    "ndvi_trend_30d",
    "pH_0_30",
    "soc_0_30",
    "clay",
    "silt",
    "ndvi_std_90d",
    "ndre_mean_90d",
    "bsi_mean_90d",
    "valid_obs_count",
    "cloud_pct",
    "area_ha",
    "elevation",
    "rainfall_30d",
    "sand"
]


def make_prediction(model, features: dict):
    if model is None:
        # fallback if model missing
        return {
            code: {
                "value": None,
                "unit": "mg/kg",
                "confidence": 0.0,
                "method": "ml"
            } for code in NUTRIENT_ORDER
        }
    # Ensure `sand` exists; compute if we have clay & silt
    if "sand" not in features and "clay" in features and "silt" in features:
        try:
            features = dict(features)  # copy
            features["sand"] = float(100 - (features.get("clay", 0) + features.get("silt", 0)))
        except Exception:
            pass

    # Create DataFrame with the exact column order expected by the trained model
    try:
        df = pd.DataFrame([[features.get(k, None) for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
    except Exception:
        # Fallback to numpy array if DataFrame creation fails
        df = np.array([list(features.values())])

    y_pred = model.predict(df)[0]

    result = {}
    for i, code in enumerate(NUTRIENT_ORDER):
        result[code] = {
            "value": float(y_pred[i]),
            "unit": "mg/kg",
            "confidence": 0.8,       # placeholder until calibration
            "method": "ml"
        }

    return result

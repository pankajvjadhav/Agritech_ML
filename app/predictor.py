import numpy as np
import pandas as pd
import warnings
from sklearn.multioutput import MultiOutputRegressor

# Suppress sklearn warnings about feature names mismatch for tree estimators
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but DecisionTreeRegressor was fitted without feature names"
)

NUTRIENT_ORDER = ["N","P","K","OC","pH","EC","S","Fe","Zn","Cu","B","Mn"]

# Unit map aligned to soil reporting standards
NUTRIENT_UNITS = {
    "N": "kg/ha",
    "P": "kg/ha",     # P2O5 as requested
    "K": "kg/ha",     # K2O as requested
    "OC": "%",
    "pH": "unitless",
    "EC": "dS/m",#(deciSiemens per meter)
    "S": "mg/kg",
    "Fe": "mg/kg",
    "Zn": "mg/kg",
    "Cu": "mg/kg",
    "B": "mg/kg",
    "Mn": "mg/kg",
}

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

def _ensure_feature_names(model, feature_names):
    """
    Ensure model and its base estimators expose feature_names_in_ to silence sklearn warnings.
    """
    try:
        if not hasattr(model, "feature_names_in_"):
            model.feature_names_in_ = np.array(feature_names)
        # Propagate to sub-estimators if present
        if hasattr(model, "estimators_"):
            for est in model.estimators_:
                if not hasattr(est, "feature_names_in_"):
                    est.feature_names_in_ = np.array(feature_names)
    except Exception:
        # Best effort; ignore if model structure differs
        pass

def compute_confidence_from_model(model, x, nutrient_keys=None):
    """
    Returns list of confidences for each nutrient (0..1).
    Shared logic for both local and API predictions.
    """
    if nutrient_keys is None:
        nutrient_keys = NUTRIENT_ORDER

    try:
        # Case A: single ensemble regressor that returns vector predictions per estimator
        if hasattr(model, "estimators_") and not isinstance(model, MultiOutputRegressor):
            preds = []
            for est in model.estimators_:
                p = np.array(est.predict(x), dtype=float).reshape(-1)
                preds.append(p)
            preds = np.stack(preds, axis=0)  # (n_estimators, n_targets)
            mean = preds.mean(axis=0)
            std = preds.std(axis=0)
            rel = np.abs(std) / (np.abs(mean) + 1e-6)
            alpha = 4.0
            conf_raw = 1.0 / (1.0 + alpha * rel)
            conf = np.clip(conf_raw, 0.2, 0.99)
            return [float(c) for c in conf]

        # Case B: MultiOutputRegressor => model.estimators_ is list of per-target estimators
        if isinstance(model, MultiOutputRegressor) and hasattr(model, "estimators_"):
            per_target_conf = []
            for est in model.estimators_:
                # If per-target estimator is itself an ensemble (e.g., RandomForest), use its estimators_
                if hasattr(est, "estimators_") and len(getattr(est, "estimators_")) > 0:
                    tree_preds = []
                    for tree in est.estimators_:
                        p = np.array(tree.predict(x), dtype=float).reshape(-1)  # shape (1,)
                        tree_preds.append(p[0])
                    arr = np.array(tree_preds, dtype=float)  # shape (n_trees,)
                    mean = arr.mean()
                    std = arr.std()
                    rel = abs(std) / (abs(mean) + 1e-6)
                    alpha = 4.0
                    conf_raw = 1.0 / (1.0 + alpha * rel)
                    conf_val = float(np.clip(conf_raw, 0.2, 0.99))
                    per_target_conf.append(conf_val)
                else:
                    # no internal ensemble for this target -> fallback
                    per_target_conf.append(0.8)
            if len(per_target_conf) < len(nutrient_keys):
                per_target_conf += [0.8] * (len(nutrient_keys) - len(per_target_conf))
            return [float(c) for c in per_target_conf[:len(nutrient_keys)]]

    except Exception:
        # If anything fails, fall back to fixed confidences
        pass

    return [0.8] * len(nutrient_keys)


def make_prediction(model, features: dict):
    if model is None:
        # fallback if model missing
        return {
            code: {
                "value": None,
                "unit": NUTRIENT_UNITS.get(code, "mg/kg"),
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

    # Best-effort fix for feature-name warnings
    _ensure_feature_names(model, FEATURE_ORDER)

    # Create DataFrame with the exact column order expected by the trained model
    try:
        base_df = pd.DataFrame([[features.get(k, None) for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
        if hasattr(model, "feature_names_in_"):
            # Align to training feature order to avoid sklearn feature-name warnings
            ordered_cols = list(model.feature_names_in_)
            df = base_df.reindex(columns=ordered_cols)
            model_input = df
        else:
            # Fall back to numpy array (no feature names used during training)
            model_input = base_df.values
    except Exception:
        # Fallback to numpy array if DataFrame creation fails
        model_input = np.array([list(features.values())])

    y_pred = model.predict(model_input)[0]

    # Compute per-nutrient confidences based on model variability
    confidences = compute_confidence_from_model(model, model_input, nutrient_keys=NUTRIENT_ORDER)

    result = {}
    for i, code in enumerate(NUTRIENT_ORDER):
        conf = confidences[i] if i < len(confidences) else 0.8
        result[code] = {
            "value": float(y_pred[i]),
            "unit": NUTRIENT_UNITS.get(code, "mg/kg"),
            "confidence": float(conf),
            "method": "ml"
        }

    return result

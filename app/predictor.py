# app/predictor.py

import pandas as pd

from config.nutrients_ranges import NUTRIENT_RANGES
from app.model_loader import STAGE2_MODELS, STAGE2_FEATURES
from app.sulfur_boron_estimator import estimate_secondary_nutrients


# --------------------------------------
# Classification Helper
# --------------------------------------
def classify_value(value, ranges):
    for label, (lo, hi) in ranges.items():
        if lo <= value < hi:
            return label
    return "unknown"


# --------------------------------------
# Hybrid Nitrogen Prediction
# --------------------------------------
def hybrid_nitrogen_prediction(features):

    oc = features.get("OC", 0)
    ndvi = features.get("NDVI_mean", 0)
    ndmi = features.get("NDMI_mean", 0)
    rainfall = features.get("rainfall_30d", 0)
    ndvi_std = features.get("NDVI_std", 0)

    oc = max(0, min(oc, 2))
    ndvi = max(0, min(ndvi, 1))
    ndmi = max(0, min(ndmi, 1))

    nitrogen = (
        oc * 140 +
        ndvi * 80 +
        ndmi * 60 +
        rainfall * 0.3 +
        ndvi_std * 40
    )

    if rainfall > 80:
        nitrogen *= 0.9

    if ndvi < 0.3:
        nitrogen *= 0.85

    if oc > 0.7:
        nitrogen *= 1.1

    if nitrogen < 100:
        nitrogen *= 1.5

    nitrogen = max(0, min(nitrogen, 700))

    return nitrogen


# --------------------------------------
# Rule Engine
# --------------------------------------
def rule_adjustment(results, fertilizer=None, soil_type=None, irrigation=None):

    adjusted = {}

    for nutrient, data in results.items():
        value = data["value"]

        if fertilizer == "High":
            if nutrient.lower() == "nitrogen":
                value += 15
            elif nutrient.lower() == "phosphorus":
                value += 5
            elif nutrient.lower() == "potassium":
                value += 7

        elif fertilizer == "Low":
            if nutrient.lower() == "nitrogen":
                value -= 10
            elif nutrient.lower() == "phosphorus":
                value -= 3

        if soil_type == "Clayey" and nutrient.lower() == "potassium":
            value += 10
        elif soil_type == "Sandy" and nutrient.lower() == "potassium":
            value -= 10

        if irrigation == "Heavy" and nutrient.lower() == "nitrogen":
            value -= 8

        value = max(0, value)
        adjusted[nutrient] = value

    return adjusted


# --------------------------------------
# MAIN FUNCTION
# --------------------------------------
def make_classification_prediction(features: dict,
                                   fertilizer=None,
                                   soil_type=None,
                                   irrigation=None):

    features = dict(features)

    # Mapping
    if "pH" not in features and "pH_0_30" in features:
        features["pH"] = features["pH_0_30"]

    if "OC" not in features and "soc_0_30" in features:
        features["OC"] = features["soc_0_30"]

    if "rainfall_30d" not in features or features["rainfall_30d"] is None:
        features["rainfall_30d"] = 50.0

    soil_map = {
        "Sandy": 0,
        "Loamy": 1,
        "Clayey": 2
    }

    fertilizer_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    irrigation_map = {
        "Rainfed": 0,
        "Moderate": 1,
        "Heavy": 2
    }

    soil_type_val = soil_map.get(soil_type if soil_type is not None else features.get("soil_type"), 0)
    fertilizer_val = fertilizer_map.get(fertilizer if fertilizer is not None else features.get("fertilizer"), 0)
    irrigation_val = irrigation_map.get(irrigation if irrigation is not None else features.get("irrigation"), 0)

    features["soil_type"] = soil_type_val
    features["fertilizer"] = fertilizer_val
    features["irrigation"] = irrigation_val

    # Build input
    X = pd.DataFrame(
        [[pd.to_numeric(features.get(f, 0), errors="coerce") for f in STAGE2_FEATURES]],
        columns=STAGE2_FEATURES
    ).fillna(0.0)

    results = {}

    # --------------------------------------
    # PRIMARY PREDICTIONS
    # --------------------------------------
    for nutrient, model in STAGE2_MODELS.items():

        name = nutrient.lower()

        if name in ["n", "nitrogen"]:
            pred_value = hybrid_nitrogen_prediction(features)
            method_used = "hybrid (oc+ndvi+ndmi+rainfall)"
            range_key = "N"

        elif name in ["p", "phosphorus"]:
            pred_value = float(model.predict(X)[0]) * 1.4
            method_used = "ml + icar"
            range_key = "P"

        elif name in ["k", "potassium"]:
            pred_value = float(model.predict(X)[0])
            method_used = "ml + icar"
            range_key = "K"

        else:
            pred_value = float(model.predict(X)[0])
            method_used = "ml + icar"
            range_key = nutrient

        # Classification (NO conversion now ✅)
        if range_key in NUTRIENT_RANGES:
            status = classify_value(pred_value, NUTRIENT_RANGES[range_key])
        else:
            status = "unknown"

        results[nutrient] = {
            "status": status,
            "value": round(pred_value, 3),
            "unit": "kg/ha",   # ✅ FINAL UNIT
            "confidence": "medium",
            "method": method_used
        }

    # --------------------------------------
    # Sulfur & Boron
    # --------------------------------------
    try:
        numeric_predictions = {k: v["value"] for k, v in results.items()}
        extra = estimate_secondary_nutrients(features, numeric_predictions)

        results["SULFUR"] = {
            "status": "estimated",
            "value": round(extra["SULFUR"], 3),
            "unit": "kg/ha",
            "confidence": "medium",
            "method": "proxy"
        }

        results["BORON"] = {
            "status": "estimated",
            "value": round(extra["BORON"], 3),
            "unit": "kg/ha",
            "confidence": "medium",
            "method": "proxy"
        }

    except Exception as e:
        print("Sulfur/Boron error:", e)

    # --------------------------------------
    # HYBRID ADJUSTMENT + LIMIT
    # --------------------------------------
    adjusted_values = rule_adjustment(
        results,
        fertilizer=fertilizer,
        soil_type=soil_type,
        irrigation=irrigation
    )

    for nutrient in results:
        ml_val = results[nutrient]["value"]
        rule_val = adjusted_values.get(nutrient, ml_val)

        final_val = (ml_val * 0.7) + (rule_val * 0.3)

        # --------------------------------------
        # CALIBRATION LAYER (Post-processing for numerical accuracy)
        # --------------------------------------
        # Apply nutrient-specific calibration to reduce systematic bias
        # Based on validation against lab data: nitrogen under-predicted, phosphorus over-predicted, potassium clipped

        if nutrient.lower() == "nitrogen":
            # Stronger uplift for consistent low-bias on field validation rows.
            final_val = final_val * 1.5 + 80.0
            # Ensure within realistic agronomic bounds (0-700 kg/ha)
            final_val = max(0, min(final_val, 700))

        elif nutrient.lower() == "phosphorus":
            # Aggressive downscale to correct systematic overestimation.
            final_val = final_val * 0.5
            # Ensure within realistic bounds (0-100 kg/ha)
            final_val = max(0, min(final_val, 100))

        elif nutrient.lower() == "potassium":
            # De-clip and downscale potassium to reduce high-bias near cap.
            final_val = final_val * 0.82
            # Use wider bound so output can vary above 500 where needed.
            final_val = max(0, min(final_val, 800))

        else:
            # For other nutrients, apply mild bounds checking without calibration changes
            final_val = max(0, final_val)

        results[nutrient]["value"] = round(final_val, 3)
        results[nutrient]["method"] += " + rule-adjusted + calibrated"

    return results
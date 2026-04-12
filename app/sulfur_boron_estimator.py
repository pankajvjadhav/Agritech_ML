"""
Sulfur & Boron Estimator Module
--------------------------------
This module estimates Sulfur (S) and Boron (B) using
agronomic proxy-based formulas derived from:

- NDVI_mean (vegetation)
- NDMI_mean (moisture)
- BSI_mean (bare soil index)
- Micronutrients (Zn, Fe)

This is a hybrid scientific + heuristic model used when
ground truth data is unavailable.
"""


# ✅ -------------------------------
# Utility: Safe normalization
# -------------------------------
def normalize(val, min_val, max_val):
    try:
        val = float(val)
        if max_val - min_val == 0:
            return 0.0
        normalized = (val - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    except Exception:
        return 0.0


# ✅ -------------------------------
# Sulfur Estimation (FIXED)
# -------------------------------
def estimate_sulfur(features):
    try:
        # ✅ FIX: using *_mean keys
        ndmi = normalize(features.get("NDMI_mean", 0), -1, 1)
        ndvi = normalize(features.get("NDVI_mean", 0), -1, 1)
        bsi  = normalize(features.get("BSI_mean", 0), -1, 1)

        sulfur_index = (
            0.5 * ndmi +
            0.3 * ndvi +
            0.2 * (1 - bsi)
        )

        sulfur_ppm = 5 + sulfur_index * 25
        sulfur_ppm = max(5, min(30, sulfur_ppm))

        return round(sulfur_ppm, 2)

    except Exception:
        return 10.0


# ✅ -------------------------------
# Boron Estimation (FIXED)
# -------------------------------
def estimate_boron(features, predictions):
    try:
        # ✅ FIX: using *_mean keys
        ndmi = normalize(features.get("NDMI_mean", 0), -1, 1)
        bsi  = normalize(features.get("BSI_mean", 0), -1, 1)

        # ✅ FIX: correct keys (Zn, Fe)
        zn = float(predictions.get("Zn", 0))
        fe = float(predictions.get("Fe", 0))

        zn_norm = normalize(zn, 0, 5)
        fe_norm = normalize(fe, 0, 10)

        boron_index = (
            0.4 * ndmi +
            0.3 * zn_norm +
            0.2 * fe_norm +
            0.1 * (1 - bsi)
        )

        boron_ppm = 0.1 + boron_index * 1.9
        boron_ppm = max(0.1, min(2.0, boron_ppm))

        return round(boron_ppm, 3)

    except Exception:
        return 0.5


# ✅ -------------------------------
# Combined Function
# -------------------------------
def estimate_secondary_nutrients(features, predictions):
    sulfur = estimate_sulfur(features)
    boron = estimate_boron(features, predictions)

    return {
        "SULFUR": sulfur,
        "BORON": boron
    }
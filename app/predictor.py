import numpy as np

NUTRIENT_ORDER = ["N","P","K","OC","pH","EC","S","Fe","Zn","Cu","B","Mn"]

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

    # Convert input dict → numpy array
    X = np.array([list(features.values())])

    y_pred = model.predict(X)[0]

    result = {}
    for i, code in enumerate(NUTRIENT_ORDER):
        result[code] = {
            "value": float(y_pred[i]),
            "unit": "mg/kg",
            "confidence": 0.8,       # placeholder until calibration
            "method": "ml"
        }

    return result

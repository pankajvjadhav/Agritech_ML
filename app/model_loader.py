# app/model_loader.py

import joblib
import os
import json
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STAGE2_MODEL_DIR = os.path.join(BASE_DIR, "models", "stage2")
STAGE2_META_PATH = os.path.join(STAGE2_MODEL_DIR, "meta.json")

# ✅ Correct nutrients (ML 2.0)
NUTRIENTS = [
    "nitrogen", "phosphorus", "potassium",
    "iron", "manganese", "zinc", "copper"
]


def load_stage2_models():
    models = {}

    for nutrient in NUTRIENTS:
        model_path = os.path.join(
            STAGE2_MODEL_DIR,
            f"{nutrient}_model.pkl"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing model: {model_path}"
            )

        models[nutrient] = joblib.load(model_path)
        logger.info(f"Loaded model: {nutrient}")

    return models


def load_stage2_metadata():
    if not os.path.exists(STAGE2_META_PATH):
        raise FileNotFoundError("Stage-2 metadata file missing")

    with open(STAGE2_META_PATH, "r") as f:
        meta = json.load(f)

    logger.info("Loaded Stage-2 metadata")
    return meta


# ✅ Load at startup
STAGE2_MODELS = load_stage2_models()
STAGE2_META = load_stage2_metadata()
STAGE2_FEATURES = STAGE2_META["features"]
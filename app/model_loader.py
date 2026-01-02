import joblib
import os
import json
import sklearn
import logging
import subprocess
import sys

current_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.join(current_dir, "..", "models", "nutrient_model_v1.pkl")
META_PATH = os.path.join(current_dir, "..", "models", "nutrient_model_v1.meta.json")

logger = logging.getLogger(__name__)

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model not found. Training model locally...")
        subprocess.check_call(
            [sys.executable, "tests/train_model.py"],
            cwd=os.path.join(current_dir, "..")
        )

def load_model():
    ensure_model_exists()
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)

    # Optional metadata validation
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        trained_version = meta.get("sklearn_version")
        if trained_version != sklearn.__version__:
            logger.warning(
                "Model trained with sklearn %s but runtime is %s",
                trained_version, sklearn.__version__
            )

    return model

model = load_model()

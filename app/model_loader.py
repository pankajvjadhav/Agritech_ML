import joblib
import os
import json
import sklearn
import logging

# Get the directory of this file
current_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.join(current_dir, "..", "models", "nutrient_model_v1.pkl")

meta_path = os.path.join(current_dir, "..", "models", "nutrient_model_v1.meta.json")
logger = logging.getLogger(__name__)
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from: %s", MODEL_PATH)
    # attempt to load metadata
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            sklearn_used = meta.get("sklearn_version")
            sklearn_runtime = sklearn.__version__
            if sklearn_used != sklearn_runtime:
                logger.warning("Model was trained with scikit-learn %s but runtime is %s. Consider re-training or pinning versions.", sklearn_used, sklearn_runtime)
            else:
                logger.info("Model sklearn version matches runtime: %s", sklearn_runtime)
        except Exception as me:
            print("Failed to read model metadata:", me)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None

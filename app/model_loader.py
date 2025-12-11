import joblib
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/nutrient_model_v1.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("Loaded model from:", MODEL_PATH)
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

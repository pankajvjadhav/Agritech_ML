import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
import sklearn

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------

DATA_PATH = Path("data/final_training_dataset.csv")
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# 2. FEATURES & TARGETS (ML 2.0)
# -------------------------------------------------

FEATURES = [
    "NDVI_mean", "NDMI_mean", "BSI_mean", "NBR2_mean",
    "VV_mean", "VH_mean", "VV_VH_ratio_mean",
    "elevation", "slope",
    "LST_mean",
    "rainfall_sum", "rainfall_mean", "rainfall_max", "rainy_days",
    "soil_type", "month"
]

TARGETS = ["ph", "ec", "oc"]

# -------------------------------------------------
# 3. PREPARE DATA
# -------------------------------------------------

X = df[FEATURES]
Y = df[TARGETS]

# Safety
assert not X.isnull().any().any(), "Missing values in features"
assert not Y.isnull().any().any(), "Missing values in targets"

# -------------------------------------------------
# 4. SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# 5. TRAIN MODELS
# -------------------------------------------------

models = {}
metrics = {}

for target in TARGETS:
    print(f"\nTraining Stage-1 model for: {target}")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train[target])

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test[target], y_pred)
    r2 = r2_score(y_test[target], y_pred)

    metrics[target] = {
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    }

    models[target] = model

    print(f"MAE: {mae:.4f}")
    print(f"R2 : {r2:.4f}")

# -------------------------------------------------
# 6. SAVE MODELS
# -------------------------------------------------

MODEL_DIR = Path("models/stage1")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

for target, model in models.items():
    joblib.dump(model, MODEL_DIR / f"{target}_model.pkl")

# -------------------------------------------------
# 7. SAVE META
# -------------------------------------------------

meta = {
    "stage": "stage1",
    "created_at": datetime.utcnow().isoformat(),
    "features": FEATURES,
    "targets": TARGETS,
    "sklearn_version": sklearn.__version__
}

with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Stage 1 training completed")
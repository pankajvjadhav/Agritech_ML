import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

# 🔥 Create nitrogen classes
def classify_nitrogen(n):
    if n < 280:
        return 0
    elif n < 560:
        return 1
    else:
        return 2

df["nitrogen_class"] = df["nitrogen"].apply(classify_nitrogen)

# -------------------------------------------------
# 2. FEATURES & TARGETS
# -------------------------------------------------

FEATURES = [
    "NDVI_mean", "NDMI_mean", "BSI_mean", "NBR2_mean",
    "VV_mean", "VH_mean", "VV_VH_ratio_mean",
    "elevation", "slope",
    "LST_mean",
    "rainfall_sum", "rainfall_mean", "rainfall_max", "rainy_days",
    "soil_type", "month",
    "ph", "ec", "oc"
]

TARGETS = [
    "nitrogen", "phosphorus", "potassium",
    "iron", "manganese", "zinc", "copper"
]

# -------------------------------------------------
# 3. PREPARE DATA
# -------------------------------------------------

X = df[FEATURES].copy()

# 🔥 Add interaction features
X["NDVI_rainfall"] = df["NDVI_mean"] * df["rainfall_sum"]
X["OC_rainfall"] = df["oc"] * df["rainfall_sum"]
X["pH_OC"] = df["ph"] * df["oc"]
X["slope_rainfall"] = df["slope"] * df["rainfall_sum"]
X["temp_moisture"] = df["LST_mean"] * df["NDMI_mean"]
X["NDVI_OC"] = df["NDVI_mean"] * df["oc"]
X["NDMI_rain"] = df["NDMI_mean"] * df["rainfall_sum"]
# 🔥 IMPORTANT: include nitrogen_class in Y
Y = df[TARGETS].copy()
Y["nitrogen_class"] = df["nitrogen_class"]

# Safety
assert not X.isnull().any().any()
assert not Y.isnull().any().any()

# -------------------------------------------------
# 4. SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# -------------------------------------------------
# 5. TRAIN MODELS
# -------------------------------------------------

models = {}
metrics = {}

for target in TARGETS:
    print(f"\nTraining Stage-2 model for: {target}")

    if target == "nitrogen":
        # 🔥 Classification model
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train["nitrogen_class"])

        y_pred = model.predict(X_test)
        y_true = y_test["nitrogen_class"]

        accuracy = (y_pred == y_true).mean()
        print(f"Accuracy: {accuracy:.2%}")

        metrics[target] = {
            "Accuracy": round(float(accuracy), 4)
        }

    else:
        # 🔥 Regression model
        model = RandomForestRegressor(
            n_estimators=900,
            max_depth=30,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)

        # Feature importance
        importances = model.feature_importances_
        feature_importance = dict(zip(X.columns, importances))

        print("\nTop Features:")
        for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{k}: {v:.4f}")

        mae = mean_absolute_error(y_test[target], y_pred)
        r2 = r2_score(y_test[target], y_pred)

        print(f"MAE: {mae:.4f}")
        print(f"R2 : {r2:.4f}")

        metrics[target] = {
            "MAE": round(mae, 4),
            "R2": round(r2, 4)
        }

    models[target] = model

# -------------------------------------------------
# 6. SAVE MODELS
# -------------------------------------------------

MODEL_DIR = Path("models/stage2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

for target, model in models.items():
    joblib.dump(model, MODEL_DIR / f"{target}_model.pkl")

# -------------------------------------------------
# 7. SAVE META
# -------------------------------------------------

meta = {
    "stage": "stage2",
    "created_at": datetime.utcnow().isoformat(),
    "features": list(X.columns),
    "targets": TARGETS,
    "sklearn_version": sklearn.__version__
}

with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Stage 2 training completed")
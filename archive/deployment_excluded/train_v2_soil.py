import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# ==============================
# 📁 Setup
# ==============================

DATA_PATH = "data/final_training_dataset.csv"
MODEL_DIR = "models/v2_soil"

os.makedirs(MODEL_DIR, exist_ok=True)

print("🚀 Starting Soil Model Training (V2)...\n")

# ==============================
# 📊 Load Dataset
# ==============================

df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded:", df.shape)

# ==============================
# 🧹 Clean Columns
# ==============================

# Drop non-useful columns
drop_cols = ["longitude", "latitude", "date"]
df = df.drop(columns=drop_cols, errors="ignore")

# Targets
targets = ["ph", "ec", "oc"]

# Remove rows with missing target values
df = df.dropna(subset=targets)

# ==============================
# 🔀 Split Data
# ==============================

X = df.drop(columns=targets)
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("📊 Training samples:", X_train.shape[0])
print("📊 Testing samples:", X_test.shape[0])

# ==============================
# 🤖 Models
# ==============================

models = {
    "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.1)
}

results = {}

# ==============================
# 🚀 Training Loop
# ==============================

for name, model in models.items():
    print(f"\n🔹 Training {name}...")

    try:
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results[name] = r2

        print(f"✅ {name} Results:")
        print(f"   MAE: {mae:.4f}")
        print(f"   R2 : {r2:.4f}")

        # Save model
        model_path = f"{MODEL_DIR}/{name}_soil_model.pkl"
        joblib.dump(model, model_path)

        print(f"💾 Saved: {model_path}")

    except Exception as e:
        print(f"❌ Error in {name}: {e}")

# ==============================
# 🏆 Best Model
# ==============================

if results:
    best_model = max(results, key=results.get)
    print("\n🏆 BEST MODEL:", best_model)
else:
    print("\n❌ No model trained successfully")

print("\n🎉 Soil Model Training Completed!")
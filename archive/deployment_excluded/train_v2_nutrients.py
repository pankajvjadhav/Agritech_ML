import pandas as pd
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor

# ==============================
# 📁 Setup
# ==============================

DATA_PATH = "data/final_training_dataset.csv"
MODEL_DIR = "models/v2_nutrients"

os.makedirs(MODEL_DIR, exist_ok=True)

print("🚀 Starting Nutrient Model Training (V2 - XGBoost)...\n")

# ==============================
# 📊 Load Dataset
# ==============================

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded:", df.shape)

# ==============================
# 🧹 Clean Columns
# ==============================

df = df.drop(columns=["longitude", "latitude", "date"], errors="ignore")

# ==============================
# 🚀 Feature Engineering (FINAL)
# ==============================

# -------- NDVI features --------
if all(col in df.columns for col in ["NDVI_max", "NDVI_min"]):
    df["NDVI_range"] = df["NDVI_max"] - df["NDVI_min"]

if "NDVI_std" in df.columns:
    df["NDVI_variability"] = df["NDVI_std"]

if all(col in df.columns for col in ["NDVI_max", "NDVI_mean"]):
    df["NDVI_trend"] = df["NDVI_max"] - df["NDVI_mean"]

# -------- Rainfall features --------
if all(col in df.columns for col in ["rainfall_max", "rainfall_mean"]):
    df["rainfall_intensity"] = df["rainfall_max"] - df["rainfall_mean"]
    df["rainfall_stability"] = df["rainfall_mean"] / (df["rainfall_max"] + 1)

# -------- Soil texture --------
if all(col in df.columns for col in ["sand", "clay", "silt"]):
    df["clay_ratio"] = df["clay"] / (df["sand"] + df["silt"] + 1)
    df["sand_clay_balance"] = df["sand"] - df["clay"]

if all(col in df.columns for col in ["sand", "clay"]):
    df["texture_index"] = df["clay"] / (df["sand"] + 1)

# -------- Organic matter --------
if all(col in df.columns for col in ["oc", "clay"]):
    df["oc_clay_interaction"] = df["oc"] * df["clay"]

# -------- Interaction features --------
if all(col in df.columns for col in ["NDVI_mean", "oc"]):
    df["NDVI_OC_interaction"] = df["NDVI_mean"] * df["oc"]

if all(col in df.columns for col in ["NDMI_mean", "rainfall_mean"]):
    df["NDMI_rainfall"] = df["NDMI_mean"] * df["rainfall_mean"]

if all(col in df.columns for col in ["slope", "rainfall_mean"]):
    df["slope_rainfall"] = df["slope"] * df["rainfall_mean"]

# -------- Temperature stress --------
if all(col in df.columns for col in ["LST", "NDVI_variability"]):
    df["LST_stress"] = df["LST"] * df["NDVI_variability"]

# ==============================
# 🔥 PROXY FEATURES
# ==============================

if all(col in df.columns for col in ["oc", "ec"]):
    df["clay_proxy"] = (df["oc"] * 100) / (df["ec"] + 1)

if "NDVI_mean" in df.columns:
    df["soil_density_proxy"] = 1 - df["NDVI_mean"]

if all(col in df.columns for col in ["rainfall_mean", "slope"]):
    df["water_retention"] = df["rainfall_mean"] / (df["slope"] + 1)

# ==============================
# 🔥 POTASSIUM FEATURES (PROXY)
# ==============================

if all(col in df.columns for col in ["clay_proxy", "rainfall_max"]):
    df["k_retention"] = df["clay_proxy"] / (df["rainfall_max"] + 1)

if all(col in df.columns for col in ["slope", "rainfall_max"]):
    df["k_leaching"] = df["slope"] * df["rainfall_max"]

if all(col in df.columns for col in ["oc", "NDVI_mean"]):
    df["k_bio_availability"] = df["oc"] * df["NDVI_mean"]

# ==============================
# 🔥 SWIR FEATURES (CORRECTED)
# ==============================

if all(col in df.columns for col in ["B11_mean", "B12_mean"]):
    df["clay_ratio_swir"] = df["B11_mean"] / (df["B12_mean"] + 1e-6)

if all(col in df.columns for col in ["B8_mean", "B11_mean"]):
    df["NDWI_swir"] = (df["B8_mean"] - df["B11_mean"]) / (df["B8_mean"] + df["B11_mean"] + 1e-6)

if all(col in df.columns for col in ["B11_mean", "NDVI_mean"]):
    df["soil_dryness_swir"] = df["B11_mean"] * (1 - df["NDVI_mean"])

# 🔥 Use SWIR in potassium
if all(col in df.columns for col in ["clay_ratio_swir", "rainfall_max"]):
    df["k_retention_swir"] = df["clay_ratio_swir"] / (df["rainfall_max"] + 1)

if all(col in df.columns for col in ["NDWI_swir", "rainfall_max"]):
    df["k_stress"] = (1 - df["NDWI_swir"]) * df["rainfall_max"]

# Remove raw bands
df = df.drop(columns=["B8_mean", "B11_mean", "B12_mean"], errors="ignore")

# ==============================
# 🔥 TOPO FEATURES
# ==============================

if all(col in df.columns for col in ["elevation", "slope"]):
    df["topo_trap"] = df["elevation"] / (df["slope"] + 1)

if all(col in df.columns for col in ["rainfall_max", "slope"]):
    df["k_leaching_risk"] = df["rainfall_max"] * df["slope"]

# ==============================
# 🔥 IRON-SPECIFIC FEATURES (SAFE ADD)
# ==============================

# 1. pH + vegetation interaction (iron availability indicator)
if all(col in df.columns for col in ["ph", "NDVI_mean"]):
    df["Fe_pH_effect"] = df["ph"] * (1 - df["NDVI_mean"])

# 2. Redox condition (moisture + organic carbon)
if all(col in df.columns for col in ["NDMI_mean", "oc"]):
    df["Fe_redox"] = df["NDMI_mean"] * df["oc"]

# 3. Aeration / drainage (oxygen effect)
if all(col in df.columns for col in ["slope", "rainfall_mean"]):
    df["Fe_aeration"] = df["slope"] / (df["rainfall_mean"] + 1)

# 4. Water stress using SWIR (very important)
if all(col in df.columns for col in ["NDWI_swir", "rainfall_mean"]):
    df["Fe_water_stress"] = (1 - df["NDWI_swir"]) * df["rainfall_mean"]    


# ==============================
# 🎯 Targets
# ==============================

nutrients = [
    "nitrogen", "phosphorus", "potassium",
    "iron", "manganese", "zinc", "copper"
]

# ==============================
# 🚀 Training
# ==============================

results = {}

for nutrient in nutrients:
    print(f"\n🔹 Training model for {nutrient.upper()}...")

    df_clean = df.dropna(subset=[nutrient])

    X = df_clean.drop(columns=nutrients)
    y = df_clean[nutrient]

    y = y.clip(lower=y.quantile(0.05), upper=y.quantile(0.95))

    # log transform for potassium
    if nutrient == "potassium":
        y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model selection
    if nutrient == "potassium":
        model = XGBRegressor(
            n_estimators=800,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42
        )
    else:
        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.3,
            reg_lambda=1.2,
            random_state=42
        )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    if nutrient == "potassium":
        pred = np.expm1(pred)
        y_test = np.expm1(y_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    results[nutrient] = r2

    print(f"✅ {nutrient.upper()} → MAE: {mae:.4f}, R2: {r2:.4f}")

    importance = pd.Series(model.feature_importances_, index=X.columns)
    print("🔝 Top Features:")
    print(importance.sort_values(ascending=False).head(10))

    joblib.dump(model, f"{MODEL_DIR}/{nutrient}_model.pkl")
    joblib.dump(None, f"{MODEL_DIR}/{nutrient}_scaler.pkl")

# ==============================
# 📊 Summary
# ==============================

print("\n📊 FINAL RESULTS SUMMARY:")
for nutrient, score in results.items():
    print(f"{nutrient.upper()} → R2: {score:.4f}")

print("\n🎉 All nutrient models trained successfully!")

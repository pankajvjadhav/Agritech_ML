# ml_service/train_model.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import joblib
import json
import sklearn
from datetime import datetime

# Load real lab data
df = pd.read_csv("data/soil_data.csv")

print("Columns:", df.columns)

# INPUT features (model will learn from these)
X = df[
 ['pH','EC','OC','CacO3','N','P','K','S',
  'Fe','Mn','Zn','Cu','Depth']
]

# TARGETS (what model will predict)
Y = df[
 ['N','P','K','OC','pH','EC',
  'S','Fe','Zn','Cu','Mn']
]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
model = MultiOutputRegressor(rf)
model.fit(X_train, y_train)

out = Path("models")
out.mkdir(exist_ok=True)
joblib.dump(model, out / "nutrient_model_v2.pkl")
print("Saved model to:", (out / "nutrient_model_v2.pkl").resolve())
meta = {
    "model_file": "nutrient_model_v2.pkl",
    "created_at": datetime.utcnow().isoformat() + "Z",
    "sklearn_version": sklearn.__version__,
    "feature_order": list(X.columns),
    "notes": "synthetic dataset, random forest multioutput"
}
with open(out / "nutrient_model_v2.meta.json", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2)
print("Saved metadata to:", (out / "nutrient_model_v2.meta.json").resolve())

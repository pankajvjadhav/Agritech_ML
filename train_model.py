# ml_service/train_model.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import joblib

# create synthetic dataset (same features order your service expects)
n_samples = 2500
rng = np.random.default_rng(42)

X = pd.DataFrame({
    "ndvi_mean_90d": rng.normal(0.45, 0.12, n_samples).clip(-1, 1),
    "ndvi_trend_30d": rng.normal(0.0, 0.03, n_samples),
    "pH_0_30": rng.normal(6.5, 0.8, n_samples).clip(3.5, 9.0),
    "soc_0_30": rng.exponential(0.8, n_samples).clip(0.1, 5.0),
    "clay": rng.normal(25, 10, n_samples).clip(0, 100),
    "silt": rng.normal(35, 10, n_samples).clip(0, 100),
    "ndvi_std_90d": np.abs(rng.normal(0.08, 0.04, n_samples)),
    "ndre_mean_90d": rng.normal(0.18, 0.06, n_samples).clip(-1, 1),
    "bsi_mean_90d": rng.normal(0.05, 0.04, n_samples).clip(-1, 1),
    "valid_obs_count": rng.integers(1, 12, n_samples),
    "cloud_pct": rng.normal(15, 12, n_samples).clip(0, 100),
    "area_ha": rng.exponential(0.5, n_samples).clip(0.01, 100),
    "elevation": rng.normal(300, 150, n_samples).clip(-50, 5000),
    "rainfall_30d": rng.normal(60, 50, n_samples).clip(0, 1000)
})
X["sand"] = (100 - (X["clay"] + X["silt"]) + rng.normal(0,5,n_samples)).clip(0,100)

def gen_targets(row):
    ndvi=row["ndvi_mean_90d"]; soc=row["soc_0_30"]; clay=row["clay"]; pH=row["pH_0_30"]
    N=(50+soc*80+ndvi*100+rng.normal(0,25)).clip(5,400)
    P=(10+(7.0-abs(pH-6.5))*10+rng.normal(0,6)).clip(1,200)
    K=(80+clay*2+rng.normal(0,30)).clip(5,600)
    OC=soc.clip(0.05,10)
    pH_t=(pH+rng.normal(0,0.2)).clip(3.5,9.0)
    EC=(0.2+(clay/100)*0.8+rng.normal(0,0.2)).clip(0.01,5.0)
    S=(5+soc*8+rng.normal(0,3)).clip(0.1,60)
    Fe=(20+clay*0.3+soc*3+rng.normal(0,8)).clip(0.1,500)
    Zn=(0.5+soc*0.6+rng.normal(0,0.5)).clip(0.01,10)
    Cu=(0.3+soc*0.4+rng.normal(0,0.3)).clip(0.01,10)
    B=(0.2+soc*0.2+rng.normal(0,0.1)).clip(0.01,5)
    Mn=(10+clay*0.4+rng.normal(0,6)).clip(0.1,800)
    return [N,P,K,OC,pH_t,EC,S,Fe,Zn,Cu,B,Mn]

Y = np.array([gen_targets(X.iloc[i]) for i in range(len(X))])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
model = MultiOutputRegressor(rf)
model.fit(X_train, y_train)

out = Path("models")
out.mkdir(exist_ok=True)
joblib.dump(model, out / "nutrient_model_v1.pkl")
print("Saved model to:", (out / "nutrient_model_v1.pkl").resolve())

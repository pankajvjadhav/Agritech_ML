import requests
import json

URL = "http://localhost:8000/predict"

payload = {
    "ndvi_mean_90d": 0.45,
    "ndvi_trend_30d": 0.02,
    "pH_0_30": 6.7,
    "soc_0_30": 0.9,
    "clay": 28,
    "silt": 32,
    "sand": 40,
    "ndvi_std_90d": 0.08,
    "ndre_mean_90d": 0.18,
    "bsi_mean_90d": 0.04,
    "valid_obs_count": 6,
    "cloud_pct": 12,
    "area_ha": 1.2,
    "elevation": 260,
    "rainfall_30d": 55
}

print("\n=== Running Manual API Test ===")
try:
    response = requests.post(URL, json=payload, timeout=10)
    print("Status:", response.status_code)
    data = response.json()
    print("Success:", data.get("success"))
    print("Model Version:", data.get("model_version"))
    print("\nPredictions:")
    for nutrient, info in data.get("predictions", {}).items():
        print(f"  {nutrient}: {info['value']} {info['unit']} (conf={info['confidence']})")
except Exception as e:
    print("ERROR:", e)

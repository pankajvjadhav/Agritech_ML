import requests, json

url = "http://localhost:8000/predict"

payload = {
    "ndvi_mean_90d":0.45,
    "ndvi_trend_30d":0.02,
    "pH_0_30":6.7,
    "soc_0_30":0.9,
    "clay":28,
    "silt":32,
    "sand":40,
    "ndvi_std_90d":0.08,
    "ndre_mean_90d":0.18,
    "bsi_mean_90d":0.04,
    "valid_obs_count":6,
    "cloud_pct":12,
    "area_ha":1.2,
    "elevation":260,
    "rainfall_30d":55
}

r = requests.post(url, json=payload)
data = r.json()

print("N =", data["predictions"]["N"]["value"])

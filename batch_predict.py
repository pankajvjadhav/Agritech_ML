import requests, json, csv, time

URL = "http://localhost:8000/predict"
INPUT_FILE = "batch_inputs.json"   # optional: list of payload JSON objects
OUTPUT_CSV = "batch_preds.csv"

# If you don't have a file, we create a small sample list:
sample_inputs = [
    {"ndvi_mean_90d":0.45,"ndvi_trend_30d":0.02,"pH_0_30":6.7,"soc_0_30":0.9,"clay":28,"silt":32,"sand":40,"ndvi_std_90d":0.08,"ndre_mean_90d":0.18,"bsi_mean_90d":0.04,"valid_obs_count":6,"cloud_pct":12,"area_ha":1.2,"elevation":260,"rainfall_30d":55},
    {"ndvi_mean_90d":0.30,"ndvi_trend_30d":-0.01,"pH_0_30":5.9,"soc_0_30":0.7,"clay":22,"silt":30,"sand":48,"ndvi_std_90d":0.05,"ndre_mean_90d":0.12,"bsi_mean_90d":0.02,"valid_obs_count":4,"cloud_pct":20,"area_ha":0.8,"elevation":300,"rainfall_30d":10},
    # add more payloads or load from file
]

# load file if present
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        inputs = json.load(fh)
except Exception:
    inputs = sample_inputs

# prepare output CSV header
header = [
    "timestamp", "input", "model_version", "predictions_json"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(header)
    for payload in inputs:
        try:
            r = requests.post(URL, json=payload, timeout=10)
            data = r.json()
            row = [int(time.time()), json.dumps(payload), data.get("model_version"), json.dumps(data.get("predictions"))]
        except Exception as e:
            row = [int(time.time()), json.dumps(payload), None, json.dumps({"error": str(e)})]
        w.writerow(row)
        print("Done:", payload.get("ndvi_mean_90d"), "->", (data.get("predictions").get("N").get("value") if data and data.get("predictions") else "ERR"))

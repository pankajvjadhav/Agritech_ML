import requests

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

def test_predict_schema():
    r = requests.post(URL, json=payload)
    assert r.status_code == 200
    
    data = r.json()
    assert data["success"] is True
    
    preds = data["predictions"]
    assert isinstance(preds, dict)
    
    required_keys = ["N","P","K","OC","pH","EC","S","Fe","Zn","Cu","B","Mn"]
    for key in required_keys:
        assert key in preds
        assert "value" in preds[key]
        assert "unit" in preds[key]
        assert "confidence" in preds[key]
        assert "method" in preds[key]

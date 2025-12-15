import requests
import json

URL = "http://localhost:8001/predict/realtime"

# Example location: Replace with actual lat, lon
lat = 28.6139  # Delhi, India
lon = 77.2090
area_ha = 1.2
satellite_api_key = "753fcd334b364c2041ad8d63ef8ff608"  

payload = {
    "lat": lat,
    "lon": lon,
    "area_ha": area_ha,
    "satellite_api_key": satellite_api_key
}

try:
    r = requests.post(URL, json=payload, timeout=30)
    if r.status_code == 200:
        data = r.json()
        print("Real-time prediction successful:")
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {r.status_code} - {r.text}")
except Exception as e:
    print(f"Exception: {e}")
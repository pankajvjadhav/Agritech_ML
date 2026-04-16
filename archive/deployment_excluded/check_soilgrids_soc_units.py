import requests
import json

coordinates = [
    {"lat": 28.7229, "lon": 76.170628},
    {"lat": 28.74679, "lon": 76.167808},
    {"lat": 29.5752, "lon": 74.674534},
    {"lat": 29.96136, "lon": 74.736078},
    {"lat": 29.57292, "lon": 75.095042},
    {"lat": 29.88033, "lon": 76.383249},
]

for i, coord in enumerate(coordinates, 1):
    lat = coord["lat"]
    lon = coord["lon"]
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["soc"],
    }
    print(f"\n--- Coordinate {i}: lat={lat}, lon={lon} ---")
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        layers = data.get("properties", {}).get("layers", [])
        for layer in layers:
            if layer.get("name") == "soc":
                print("SOC layer units:", layer.get("unit"))
                for depth in layer.get("depths", []):
                    print(f"  Depth: {depth.get('label')}, Mean: {depth.get('values', {}).get('mean')}")
    except Exception as e:
        print("Error fetching SOC for this coordinate:", e)

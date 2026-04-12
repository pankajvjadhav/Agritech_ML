import requests
import time

def get_clay_value(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=clay"

    for _ in range(3):  # retry 3 times
        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data["properties"]["layers"][0]["depths"][1]["values"]["mean"]

            else:
                print("Server busy, retrying...")
                time.sleep(2)

        except Exception as e:
            print("Error:", e)
            time.sleep(2)

    return None


print(get_clay_value(18.5204, 73.8567))
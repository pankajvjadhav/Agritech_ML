import traceback
import app.data_fetcher as df
from app.data_fetcher import fetch_real_time_data
from app.predictor import make_classification_prediction

# Coordinates and date for prediction
coordinates = [
    {"lat": 28.7229, "lon": 76.170628},
    {"lat": 28.74679, "lon": 76.167808},
    {"lat": 29.5752, "lon": 74.674534},
    {"lat": 29.96136, "lon": 74.736078},
    {"lat": 29.57292, "lon": 75.095042},
    {"lat": 29.88033, "lon": 76.383249},
]

# Set the month for 09-Apr-26
PREDICTION_MONTH = 4  # April

def main():
    old_timeout = df.REQUEST_TIMEOUT
    old_retries = df.MAX_RETRIES
    df.REQUEST_TIMEOUT = 45
    df.MAX_RETRIES = 2

    for i, row in enumerate(coordinates, start=1):
        try:
            features, _ = fetch_real_time_data(row["lat"], row["lon"])
            features["month"] = PREDICTION_MONTH
            pred = make_classification_prediction(features)
            print(f"\nResult for Coordinate {i} (lat={row['lat']}, lon={row['lon']}):")
            for key in ["nitrogen", "phosphorus", "potassium", "SULFUR", "BORON"]:
                if key in pred:
                    print(f"  {key.capitalize()}: {pred[key]['value']} {pred[key]['unit']} (status: {pred[key]['status']})")
            # Print pH, OC, EC from features if available
            print(f"  pH: {features.get('ph', 'N/A')}")
            print(f"  OC: {features.get('oc', 'N/A')}")
            print(f"  EC: {features.get('ec', 'N/A')}")
        except Exception as exc:
            print(f"Failed for Coordinate {i} (lat={row['lat']}, lon={row['lon']}): {exc}")

    df.REQUEST_TIMEOUT = old_timeout
    df.MAX_RETRIES = old_retries

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

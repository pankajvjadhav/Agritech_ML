import math
import traceback

import app.data_fetcher as df
from app.data_fetcher import fetch_real_time_data
from app.predictor import make_classification_prediction


def main():
    old_timeout = df.REQUEST_TIMEOUT
    old_retries = df.MAX_RETRIES
    df.REQUEST_TIMEOUT = 45
    df.MAX_RETRIES = 2

    lab_rows = [
        {"lon": 73.8439, "lat": 18.5394, "N": 163.07, "P": 12.51, "K": 433.44, "Fe": 4.42, "Mn": 4.96, "Zn": 2.58, "Cu": 3.62, "S": 19.42},
        {"lon": 73.8444, "lat": 18.5394, "N": 175.62, "P": 15.95, "K": 484.96, "Fe": 4.24, "Mn": 5.77, "Zn": 2.35, "Cu": 3.18, "S": 20.13},
        {"lon": 73.8447, "lat": 18.5397, "N": 131.98, "P": 11.89, "K": 444.64, "Fe": 3.84, "Mn": 6.64, "Zn": 2.57, "Cu": 2.85, "S": 20.80},
        {"lon": 73.8447, "lat": 18.5403, "N": 163.07, "P": 14.08, "K": 341.60, "Fe": 3.44, "Mn": 6.46, "Zn": 0.98, "Cu": 2.87, "S": 19.33},
        {"lon": 73.8456, "lat": 18.5400, "N": 150.53, "P": 17.52, "K": 315.84, "Fe": 3.26, "Mn": 6.67, "Zn": 1.48, "Cu": 2.59, "S": 19.58},
        {"lon": 73.8469, "lat": 18.5394, "N": 225.40, "P": 13.76, "K": 535.36, "Fe": 3.32, "Mn": 6.78, "Zn": 1.35, "Cu": 2.41, "S": 17.68},
        {"lon": 73.8467, "lat": 18.5400, "N": 175.62, "P": 11.89, "K": 460.32, "Fe": 3.49, "Mn": 6.96, "Zn": 0.96, "Cu": 2.56, "S": 21.47},
        {"lon": 73.8472, "lat": 18.5386, "N": 263.42, "P": 14.08, "K": 697.76, "Fe": 3.64, "Mn": 6.54, "Zn": 1.78, "Cu": 3.52, "S": 21.72},
        {"lon": 73.8467, "lat": 18.5389, "N": 225.79, "P": 18.14, "K": 640.64, "Fe": 3.58, "Mn": 7.06, "Zn": 1.64, "Cu": 3.35, "S": 22.70},
        {"lon": 73.8469, "lat": 18.5406, "N": 175.20, "P": 10.02, "K": 430.08, "Fe": 3.88, "Mn": 7.14, "Zn": 2.47, "Cu": 3.23, "S": 20.04},
        {"lon": 73.8467, "lat": 18.5408, "N": 163.07, "P": 10.51, "K": 593.60, "Fe": 4.15, "Mn": 6.85, "Zn": 2.69, "Cu": 3.65, "S": 19.52},
        {"lon": 73.8469, "lat": 18.5381, "N": 163.07, "P": 8.06, "K": 393.12, "Fe": 4.49, "Mn": 8.32, "Zn": 2.89, "Cu": 3.98, "S": 23.31},
        {"lon": 73.8414, "lat": 18.5381, "N": 150.53, "P": 11.40, "K": 335.52, "Fe": 4.44, "Mn": 8.02, "Zn": 2.92, "Cu": 3.48, "S": 22.36},
        {"lon": 73.8414, "lat": 18.5386, "N": 200.70, "P": 18.20, "K": 495.52, "Fe": 4.26, "Mn": 7.87, "Zn": 2.69, "Cu": 3.62, "S": 24.14},
        {"lon": 73.8411, "lat": 18.5392, "N": 225.99, "P": 29.41, "K": 334.88, "Fe": 4.11, "Mn": 8.65, "Zn": 2.84, "Cu": 2.64, "S": 25.76},
        {"lon": 73.8408, "lat": 18.5386, "N": 150.53, "P": 22.21, "K": 397.60, "Fe": 4.44, "Mn": 8.87, "Zn": 2.76, "Cu": 2.83, "S": 24.44},
    ]

    pred_rows = []
    valid_lab_rows = []
    try:
        for i, row in enumerate(lab_rows, start=1):
            try:
                features, _ = fetch_real_time_data(row["lat"], row["lon"])
                features["month"] = 5
                pred = make_classification_prediction(features)
                pred_rows.append({
                    "N": pred["nitrogen"]["value"],
                    "P": pred["phosphorus"]["value"],
                    "K": pred["potassium"]["value"],
                    "Fe": pred["iron"]["value"],
                    "Mn": pred["manganese"]["value"],
                    "Zn": pred["zinc"]["value"],
                    "Cu": pred["copper"]["value"],
                    "S": pred["SULFUR"]["value"],
                })
                valid_lab_rows.append(row)
                print(f"ok {i}")
            except Exception as exc:
                print(f"failed {i}: {exc}")

        keys = ["N", "P", "K", "Fe", "Mn", "Zn", "Cu", "S"]
        print("SUMMARY_METRICS")
        for key in keys:
            y = [row[key] for row in valid_lab_rows]
            yhat = [row[key] for row in pred_rows]
            n = len(y)
            mae = sum(abs(a - b) for a, b in zip(y, yhat)) / n
            rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, yhat)) / n)
            mape = 100.0 * sum(abs((a - b) / a) for a, b in zip(y, yhat) if a != 0) / n
            bias = sum((b - a) for a, b in zip(y, yhat)) / n
            print(f"{key},MAE={mae:.3f},RMSE={rmse:.3f},MAPE={mape:.2f}%,BIAS={bias:.3f}")
    finally:
        df.REQUEST_TIMEOUT = old_timeout
        df.MAX_RETRIES = old_retries


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

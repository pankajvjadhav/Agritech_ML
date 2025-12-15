import argparse
import json
import os
import requests
import pandas as pd
import numpy as np

url = "http://localhost:8000/predict"

# Load static dataset and use first row as default payload
def load_static_payload(dataset_path="static_dataset.csv", row_index=0):
    """Load payload from static dataset."""
    try:
        df = pd.read_csv(dataset_path)
        if len(df) == 0:
            raise ValueError("Dataset is empty")
        if row_index < 0 or row_index >= len(df):
            row_index = 0  # Use first row if index out of range
        
        row = df.iloc[row_index]
        return {
            "ndvi_mean_90d": float(row["ndvi_mean_90d"]),
            "ndvi_trend_30d": float(row["ndvi_trend_30d"]),
            "pH_0_30": float(row["pH_0_30"]),
            "soc_0_30": float(row["soc_0_30"]),
            "clay": float(row["clay"]),
            "silt": float(row["silt"]),
            "sand": float(row["sand"]),
            "ndvi_std_90d": float(row["ndvi_std_90d"]),
            "ndre_mean_90d": float(row["ndre_mean_90d"]),
            "bsi_mean_90d": float(row["bsi_mean_90d"]),
            "valid_obs_count": int(row["valid_obs_count"]),
            "cloud_pct": float(row["cloud_pct"]),
            "area_ha": float(row["area_ha"]),
            "elevation": float(row["elevation"]),
            "rainfall_30d": float(row["rainfall_30d"])
        }
    except Exception as e:
        print(f"Warning: Could not load from dataset ({e}), using default values")
        # Fallback to original hardcoded values
        return {
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

# Load default payload from static dataset (first row)
payload = load_static_payload()


def call_server(payload, url="http://localhost:8000/predict"):
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def call_local(payload):
    try:
        from app.model_loader import model
        from app.predictor import make_prediction
    except Exception as e:
        raise RuntimeError("Failed to import local modules: {}".format(e))
    return {"success": True, "predictions": make_prediction(model, payload)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test prediction via HTTP or local model')
    parser.add_argument('--http', dest='use_http', action='store_true', help='POST to running server')
    parser.add_argument('--url', dest='url', type=str, default=url, help='Override target URL')
    parser.add_argument('--realtime', dest='realtime', action='store_true', help='Call realtime endpoint (requires lat/lon)')
    parser.add_argument('--lat', type=float, default=None, help='Latitude for realtime fetch')
    parser.add_argument('--lon', type=float, default=None, help='Longitude for realtime fetch')
    parser.add_argument('--area', type=float, default=1.0, help='Area in hectares for realtime fetch')
    parser.add_argument('--api-key', dest='api_key', type=str, default=os.getenv('753fcd334b364c2041ad8d63ef8ff608'), help='Satellite API key; can be provided via env AGROMONITOR_API_KEY')
    parser.add_argument('--format', dest='out_format', choices=['json', 'pretty', 'csv', 'shc', 'list'], default='pretty', help='Output format')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Show debug info including fetched realtime features')
    parser.add_argument('--sample-id', dest='sample_id', type=str, default='', help='Sample ID to include in SHC CSV export')
    parser.add_argument('--dataset-row', dest='dataset_row', type=int, default=0, help='Row index from static_dataset.csv to use (default: 0)')
    args = parser.parse_args()
    
    # Load payload from dataset if not using realtime
    if not args.realtime:
        payload = load_static_payload(row_index=args.dataset_row)

    if args.use_http:
        try:
            if args.realtime:
                if args.lat is None or args.lon is None:
                    raise ValueError('Realtime mode requires --lat and --lon')
                realtime_url = args.url.rstrip('/') + '/realtime'
                # If user didn't pass --api-key, check env var and show which is used
                satellite_key_in_use = args.api_key or os.getenv('AGROMONITOR_API_KEY')
                if args.debug and args.out_format != 'list':
                    print(f"Using satellite API key: {'<present>' if satellite_key_in_use else '<none>'}")
                loc_payload = {
                    "lat": args.lat,
                    "lon": args.lon,
                    "area_ha": args.area,
                    "satellite_api_key": satellite_key_in_use
                }
                data = call_server(loc_payload, realtime_url)
            else:
                data = call_server(payload, args.url)
        except Exception as e:
            print('Server call failed:', e)
            raise
    else:
        if args.realtime:
            # Local realtime: use dataset (randomly selected row)
            try:
                if args.lat is None or args.lon is None:
                    raise ValueError('Realtime mode requires --lat and --lon')
                from app.data_fetcher import fetch_real_time_data
                # use_dataset=True means it will randomly select from static dataset
                loc_features, loc_meta = fetch_real_time_data(args.lat, args.lon, args.area, None, use_dataset=True)
                if args.debug and args.out_format != 'list':
                    print('\nFetched features from static dataset:')
                    try:
                        import json
                        print(json.dumps(loc_features, indent=2))
                        if loc_meta:
                            print('\nDataset metadata:')
                            print(json.dumps(loc_meta, indent=2))
                    except Exception:
                        print(loc_features)
                        print('Meta:', loc_meta)
                data = call_local(loc_features)
            except Exception as e:
                print('Local realtime fetch failed:', e)
                raise
        else:
            data = call_local(payload)

    # Output according to chosen format
    preds = data.get('predictions', {})
    if args.out_format == 'json':
        print(json.dumps(data))
    elif args.out_format == 'pretty':
        print(json.dumps(data, indent=2))
        # Print human-friendly list
        try:
            if args.debug and args.out_format != 'list' and data.get('meta'):
                print('\nResponse meta:')
                try:
                    print(json.dumps(data.get('meta'), indent=2))
                except Exception:
                    print(data.get('meta'))
            if preds:
                print('\nNutrient predictions:')
                for code, entry in preds.items():
                    val = entry.get('value')
                    unit = entry.get('unit', '')
                    conf = entry.get('confidence', '')
                    method = entry.get('method', '')
                    print(f"- {code}: {val} {unit} (confidence: {conf}, method: {method})")
        except Exception:
            pass
    elif args.out_format == 'csv':
        # CSV header and rows: code,value,unit,confidence,method
        try:
            import csv, sys
            writer = csv.writer(sys.stdout)
            writer.writerow(['code', 'value', 'unit', 'confidence', 'method'])
            for code, entry in preds.items():
                writer.writerow([code, entry.get('value'), entry.get('unit'), entry.get('confidence'), entry.get('method')])
        except Exception:
            # fallback to JSON print
            print(json.dumps(data))
    elif args.out_format == 'shc':
        # Soil Health Card CSV format (single-row CSV):
        # sample_id,model_version,timestamp,ndvi_mean_90d,ndvi_trend_30d,pH_0_30,soc_0_30,clay,silt,sand,ndvi_std_90d,ndre_mean_90d,bsi_mean_90d,valid_obs_count,cloud_pct,area_ha,elevation,rainfall_30d,N,P,K,OC,pH_val,EC,S,Fe,Zn,Cu,B,Mn
        try:
            import csv, sys, time
            model_version = os.getenv('MODEL_VERSION', 'nutrient_model_v1')
            ts = int(time.time())
            header = [
                'sample_id', 'model_version', 'timestamp',
                'ndvi_mean_90d','ndvi_trend_30d','pH_0_30','soc_0_30','clay','silt','sand','ndvi_std_90d','ndre_mean_90d','bsi_mean_90d','valid_obs_count','cloud_pct','area_ha','elevation','rainfall_30d',
                'N','P','K','OC','pH','EC','S','Fe','Zn','Cu','B','Mn'
            ]
            writer = csv.writer(sys.stdout)
            writer.writerow(header)
            # extract input features if present
            row = [args.sample_id, model_version, ts]
            # input features (use defaults if missing)
            row += [payload.get(k, '') for k in [
                'ndvi_mean_90d','ndvi_trend_30d','pH_0_30','soc_0_30','clay','silt','sand','ndvi_std_90d','ndre_mean_90d','bsi_mean_90d','valid_obs_count','cloud_pct','area_ha','elevation','rainfall_30d'
            ]]
            # nutrient values
            nutrient_order = ['N','P','K','OC','pH','EC','S','Fe','Zn','Cu','B','Mn']
            for n in nutrient_order:
                val = preds.get(n, {}).get('value') if preds else None
                row.append(val)
            writer.writerow(row)
        except Exception:
            print(json.dumps(data))
    elif args.out_format == 'list':
        # Print only nutrient list values without JSON wrapper or meta
        try:
            nutrient_order = ['N','P','K','OC','pH','EC','S','Fe','Zn','Cu','B','Mn']
            for code in nutrient_order:
                entry = preds.get(code, {}) if preds else {}
                val = entry.get('value') if entry else None
                unit = entry.get('unit', '')
                if val is None:
                    continue
                print(f"{code}: {val} {unit}".strip())
        except Exception:
            pass

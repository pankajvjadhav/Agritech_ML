import requests
import json
import time
import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Path to the static dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static_dataset.csv")

def fetch_soil_data(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch soil data from ISRIC SoilGrids API.
    Returns pH, SOC, clay, silt, sand for 0-30cm.
    """
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&property=soc&property=clay&property=silt&property=sand&depth=0-30cm&value=mean"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        data = response.json()
        if 'properties' not in data:
            raise Exception("No properties in response")
        properties = data['properties']
        ph = properties['phh2o']['layers'][0]['depths'][0]['values']['mean'] / 10.0
        soc = properties['soc']['layers'][0]['depths'][0]['values']['mean'] / 100.0
        clay = properties['clay']['layers'][0]['depths'][0]['values']['mean'] / 10.0
        silt = properties['silt']['layers'][0]['depths'][0]['values']['mean'] / 10.0
        sand = properties['sand']['layers'][0]['depths'][0]['values']['mean'] / 10.0
        return {
            "pH_0_30": ph,
            "soc_0_30": soc,
            "clay": clay,
            "silt": silt,
            "sand": sand
        }
    except Exception as e:
        logger.warning("Failed to fetch soil data: %s. Using defaults.", e)
        return {
            "pH_0_30": 6.5,
            "soc_0_30": 0.8,
            "clay": 25,
            "silt": 35,
            "sand": 40
        }

def fetch_elevation(lat: float, lon: float) -> float:
    """
    Fetch elevation from Open-Elevation API.
    """
    try:
        url = "https://api.open-elevation.com/api/v1/lookup"
        payload = {"locations": [{"latitude": lat, "longitude": lon}]}
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        data = response.json()
        return data['results'][0]['elevation']
    except Exception as e:
        logger.warning("Failed to fetch elevation: %s. Using default.", e)
        return 300.0

def fetch_rainfall(lat: float, lon: float, api_key: str = None) -> float:
    """
    Fetch 30-day rainfall from NASA POWER API (free).
    """
    try:
        end_date = time.strftime("%Y%m%d")
        start_date = time.strftime("%Y%m%d", time.localtime(time.time() - 30*24*3600))
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=RE&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        data = response.json()
        properties = data['properties']
        rainfall_values = [v for v in properties['parameter']['PRECTOTCORR'].values() if v != -999]
        rainfall = sum(rainfall_values)
        return rainfall
    except Exception as e:
        logger.warning("Failed to fetch rainfall: %s. Using default.", e)
        return 55.0

def fetch_satellite_indices(lat: float, lon: float, api_key: str = None) -> Dict[str, Any]:
    """
    Fetch satellite indices from Agromonitoring API.
    Requires API key from https://agromonitoring.com/
    """
    if not api_key:
        # Fallback to dummy data
        return {
            "ndvi_mean_90d": 0.45,
            "ndvi_trend_30d": 0.02,
            "ndvi_std_90d": 0.08,
            "ndre_mean_90d": 0.18,
            "bsi_mean_90d": 0.04,
            "valid_obs_count": 6,
            "cloud_pct": 12
        }, {"sat_source": "placeholder", "api_key_used": False}
    
    # Agromonitoring API for NDVI
    # First, get polygon or use point
    # For simplicity, use a small polygon around the point
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [lon-0.01, lat-0.01],
            [lon+0.01, lat-0.01],
            [lon+0.01, lat+0.01],
            [lon-0.01, lat+0.01],
            [lon-0.01, lat-0.01]
        ]]
    }
    
    # Get NDVI stats
    url = "https://api.agromonitoring.com/agro/1.0/ndvi/history"
    params = {
        "appid": api_key,
        "start": int(time.time() - 90*24*3600),  # 90 days ago
        "end": int(time.time())
    }
    try:
        response = requests.post(url, json=polygon, params=params, timeout=10)
    except Exception as e:
        # treat as failure and fall back
        logger.warning("Agromonitoring API error (POST): %s", e)
        response = None
        response = None
    if response is None or response.status_code != 200:
        if response is not None:
            logger.warning("Agromonitoring API error (POST): %s - %s", response.status_code, response.text)
        # Try a fallback GET using lat/lon as query params (account/api may require GET or point lookup)
        try:
            params_get = params.copy()
            params_get.update({"lat": lat, "lon": lon})
            response_get = requests.get(url, params=params_get, timeout=10)
            if response_get.status_code == 200:
                ndvi_data = response_get.json()
                response = response_get
                api_ok = True
            else:
                print(f"Agromonitoring API error (GET fallback): {response_get.status_code} - {response_get.text}")
                raise Exception("Failed to fetch NDVI data")
        except Exception as e:
            logger.warning("Agromonitoring API GET fallback error: %s", e)
            raise Exception("Failed to fetch NDVI data") from e
    ndvi_data = response.json()
    api_ok = True
    
    # Process NDVI data
    ndvi_values = [item['data']['ndvi'] for item in ndvi_data if 'data' in item and 'ndvi' in item['data']]
    if not ndvi_values:
        ndvi_mean = 0.45
        ndvi_std = 0.08
        ndvi_trend = 0.02
    else:
        ndvi_mean = np.mean(ndvi_values)
        ndvi_std = np.std(ndvi_values)
        # Trend: simple difference
        if len(ndvi_values) > 1:
            ndvi_trend = (ndvi_values[-1] - ndvi_values[0]) / len(ndvi_values) * 30  # rough
        else:
            ndvi_trend = 0.02
    
    # For NDRE, BSI, etc., similar, but API may not have all. Use defaults or extend.
    # For now, use defaults for others
    meta = {"sat_source": "agromonitoring", "api_key_used": bool(api_key)}
    return {
        "ndvi_mean_90d": ndvi_mean,
        "ndvi_trend_30d": ndvi_trend,
        "ndvi_std_90d": ndvi_std,
        "ndre_mean_90d": 0.18,  # Placeholder
        "bsi_mean_90d": 0.04,   # Placeholder
        "valid_obs_count": len(ndvi_values),
        "cloud_pct": 12  # Placeholder
    }, meta

def fetch_from_dataset(index: int = None) -> Dict[str, Any]:
    """
    Fetch data from the static dataset by index.
    Returns features dictionary for the specified row index.
    If index is None, returns a random row.
    """
    try:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
        
        df = pd.read_csv(DATASET_PATH)
        
        if index is None:
            # Return random row if no index specified
            index = np.random.randint(0, len(df))
            logger.info(f"No index provided, using random index: {index}")
        
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} is out of range. Dataset has {len(df)} rows (valid indices: 0-{len(df)-1})")
        
        row = df.iloc[index]
        
        # Convert to dictionary with proper types
        features = {
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
        
        meta = {
            "source": "static_dataset",
            "dataset_path": DATASET_PATH,
            "row_index": index,
            "total_rows": len(df)
        }
        
        return features, meta
        
    except Exception as e:
        logger.error(f"Failed to fetch data from dataset at index {index}: {e}")
        raise

def find_closest_dataset_row(lat: float = None, lon: float = None, target_features: Dict[str, Any] = None) -> int:
    """
    Find the closest matching row in the dataset based on location or feature similarity.
    Returns the index of the closest row.
    
    Priority:
    1. If target_features provided, find closest by feature similarity
    2. If lat/lon provided, try to match by elevation/rainfall (proxy for location)
    3. Otherwise return random index
    """
    try:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
        
        df = pd.read_csv(DATASET_PATH)
        
        if len(df) == 0:
            raise ValueError("Dataset is empty")
        
        # If target features provided, find closest match by feature similarity
        if target_features:
            # Features to match on (excluding area_ha which is user-specific)
            match_features = [
                "ndvi_mean_90d", "ndvi_trend_30d", "pH_0_30", "soc_0_30",
                "clay", "silt", "sand", "elevation", "rainfall_30d"
            ]
            
            # Build query vector
            query_values = []
            for feat in match_features:
                query_values.append(target_features.get(feat, 0))
            
            # Calculate distances to all rows
            distances = []
            for idx, row in df.iterrows():
                row_values = [float(row[feat]) for feat in match_features]
                # Euclidean distance
                dist = np.sqrt(sum((q - r) ** 2 for q, r in zip(query_values, row_values)))
                distances.append((dist, idx))
            
            # Return index of closest match
            distances.sort()
            return int(distances[0][1])
        
        # If lat/lon provided, try to match by elevation/rainfall
        # (This is a simple heuristic - in production you'd want location in dataset)
        if lat is not None and lon is not None:
            try:
                # Try to get elevation for this location
                elevation = fetch_elevation(lat, lon)
                rainfall = fetch_rainfall(lat, lon)
                
                # Find closest match by elevation and rainfall
                distances = []
                for idx, row in df.iterrows():
                    elev_diff = abs(float(row["elevation"]) - elevation)
                    rain_diff = abs(float(row["rainfall_30d"]) - rainfall)
                    dist = elev_diff + rain_diff * 0.1  # Weight rainfall less
                    distances.append((dist, idx))
                
                distances.sort()
                return int(distances[0][1])
            except Exception as e:
                logger.warning(f"Could not fetch elevation/rainfall for matching: {e}. Using random.")
        
        # Fallback: return random index
        return np.random.randint(0, len(df))
        
    except Exception as e:
        logger.error(f"Error finding closest dataset row: {e}")
        # Return random index as fallback
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            return np.random.randint(0, len(df))
        raise

def fetch_real_time_data(lat: float, lon: float, area_ha: float = 1.0, satellite_api_key: str = None, use_dataset: bool = True):
    """
    Fetch all real-time data for the given location.
    If use_dataset is True, randomly selects from static dataset instead of calling APIs.
    """
    if use_dataset:
        # Use random row from static dataset
        try:
            if not os.path.exists(DATASET_PATH):
                raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
            
            df = pd.read_csv(DATASET_PATH)
            if len(df) == 0:
                raise ValueError("Dataset is empty")
            
            # Randomly select a row
            random_idx = np.random.randint(0, len(df))
            row = df.iloc[random_idx]
            
            # Use area_ha from parameter, but keep other values from dataset
            features = {
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
                "area_ha": area_ha,  # Use provided area_ha
                "elevation": float(row["elevation"]),
                "rainfall_30d": float(row["rainfall_30d"])
            }
            
            meta = {
                "source": "static_dataset_random",
                "dataset_path": DATASET_PATH,
                "row_index": int(random_idx),
                "total_rows": len(df),
                "location": {"lat": lat, "lon": lon}
            }
            
            logger.info(f"Using random dataset row {random_idx} for location ({lat}, {lon})")
            return features, meta
            
        except Exception as e:
            logger.warning(f"Failed to use dataset, falling back to API: {e}")
            # Fall through to API calls below
    
    # Original API-based approach (fallback)
    soil_data = fetch_soil_data(lat, lon)
    elevation = fetch_elevation(lat, lon)
    rainfall = fetch_rainfall(lat, lon)
    satellite_data, sat_meta = fetch_satellite_indices(lat, lon, satellite_api_key)
    
    features = {
        "ndvi_mean_90d": satellite_data["ndvi_mean_90d"],
        "ndvi_trend_30d": satellite_data["ndvi_trend_30d"],
        "pH_0_30": soil_data["pH_0_30"],
        "soc_0_30": soil_data["soc_0_30"],
        "clay": soil_data["clay"],
        "silt": soil_data["silt"],
        "sand": soil_data["sand"],
        "ndvi_std_90d": satellite_data["ndvi_std_90d"],
        "ndre_mean_90d": satellite_data["ndre_mean_90d"],
        "bsi_mean_90d": satellite_data["bsi_mean_90d"],
        "valid_obs_count": satellite_data["valid_obs_count"],
        "cloud_pct": satellite_data["cloud_pct"],
        "area_ha": area_ha,
        "elevation": elevation,
        "rainfall_30d": rainfall
    }
    meta = {"satellite_source": sat_meta.get('sat_source'), "api_key_used": sat_meta.get('api_key_used')}
    return features, meta
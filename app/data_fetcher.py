import requests
import time
import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, Any
from datetime import datetime, timedelta

from satellite_features.rainfall_features import get_rainfall_features
from satellite_features.sentinel1_features import get_sentinel1_features
from satellite_features.sentinel2_features import get_sentinel2_features
from satellite_features.soil_type_features import get_soil_type
from satellite_features.terrain_features import get_terrain_features

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = int(os.getenv("EXTERNAL_API_TIMEOUT", "45"))
MAX_RETRIES = 3


class ExternalDataFetchError(RuntimeError):
    pass

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "final_training_dataset.csv"
)


def _http_get_json(url: str, source: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                raise ExternalDataFetchError(
                    f"{source} request failed (status={response.status_code}): {response.text[:200]}"
                )
            return response.json()
        except Exception as e:
            last_error = e
            logger.warning("%s attempt %s/%s failed: %s", source, attempt, MAX_RETRIES, e)

    raise ExternalDataFetchError(f"{source} unavailable after retries: {last_error}")


def _weighted_topsoil_mean(layer: Dict[str, Any]) -> float:
    depths = layer.get("depths", [])
    d_factor = float(layer.get("unit_measure", {}).get("d_factor", 1.0) or 1.0)
    weights = {
        "0-5cm": 5.0,
        "5-15cm": 10.0,
        "15-30cm": 15.0,
    }

    total = 0.0
    weight_sum = 0.0

    for depth in depths:
        label = depth.get("label")
        mean_val = depth.get("values", {}).get("mean")
        if label in weights and mean_val is not None:
            total += (float(mean_val) / d_factor) * weights[label]
            weight_sum += weights[label]

    if weight_sum > 0:
        return total / weight_sum

    if depths:
        fallback_mean = depths[0].get("values", {}).get("mean")
        if fallback_mean is not None:
            return float(fallback_mean) / d_factor

    raise ExternalDataFetchError("SoilGrids layer has no usable depth values")


def _extract_soil_property(properties: Dict[str, Any], name: str) -> float:
    for layer in properties.get("layers", []):
        if layer.get("name") == name:
            return _weighted_topsoil_mean(layer)
    raise ExternalDataFetchError(f"Missing soil property '{name}' in SoilGrids response")


def _to_float_or_none(value: Any):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_ec(soil_moisture, clay_content, oc_value, rainfall_mean):
    moisture = _to_float_or_none(soil_moisture)
    clay = _to_float_or_none(clay_content)
    oc = _to_float_or_none(oc_value)
    rain = _to_float_or_none(rainfall_mean)

    if moisture is None:
        return None, None

    moisture = max(0.0, min(moisture, 1.0))
    if moisture < 0.2:
        category = "Low"
    elif moisture < 0.5:
        category = "Medium"
    else:
        category = "High"

    # Numeric EC is emitted only when all primary inputs are available.
    if clay is None or oc is None or rain is None:
        return None, category

    ec_value = (moisture * 2.5) + (clay * 0.1) + (oc * 1.2)
    return round(ec_value, 3), category

# -------------------------------
# SOIL DATA
# -------------------------------
def fetch_soil_data(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["phh2o", "soc", "clay", "silt", "sand"],
    }
    data = _http_get_json(url, source="SoilGrids", params=params)

    properties = data.get("properties", {})
    ph = _extract_soil_property(properties, "phh2o")
    # --- Weighted OC calculation (0-30cm) ---
    soc_layer = None
    for layer in properties.get("layers", []):
        if layer.get("name") == "soc":
            soc_layer = layer
            break
    soc_weighted = None
    if soc_layer:
        soc_weighted = _weighted_topsoil_mean(soc_layer)
    else:
        soc_weighted = 0.0
    oc = soc_weighted / 100.0
    print(f"[SoilGrids] Weighted SOC (0-30cm): {soc_weighted} g/kg, OC: {oc} %")

    clay = _extract_soil_property(properties, "clay")
    silt = _extract_soil_property(properties, "silt")
    sand = _extract_soil_property(properties, "sand")

    if sand >= 55 and clay < 25:
        soil_type = "Sandy"
    elif clay >= 35:
        soil_type = "Clayey"
    else:
        soil_type = "Loamy"

    # Dynamic EC proxy based on live soil texture/OC instead of static constants.
    ec = max(0.05, min(2.5, 0.08 + 0.003 * clay + 0.2 * oc))

    return {
        "ph": ph,
        "oc": oc,
        "ec": ec,
        "soil_type": soil_type,
        "clay": clay,
        "silt": silt,
        "sand": sand,
    }


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def fetch_real_time_data(lat: float, lon: float, area_ha: float = 1.0, satellite_api_key: str = None, use_dataset: bool = False):

    if use_dataset:
        raise ExternalDataFetchError("Dataset fallback is disabled in production realtime mode")

    soil_data = fetch_soil_data(lat, lon)

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=30)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    terrain = get_terrain_features(lat, lon)
    rainfall = get_rainfall_features(lat, lon, start_date_str, end_date_str)
    sentinel2 = get_sentinel2_features(lat, lon, start_date_str, end_date_str)
    sentinel1 = get_sentinel1_features(lat, lon, start_date_str, end_date_str)
    gee_soil_type = get_soil_type(lat, lon)

    month = datetime.utcnow().month
    ndvi_mean = sentinel2["NDVI_mean"]
    ndmi_mean = sentinel2["NDMI_mean"]
    bsi_mean = sentinel2["BSI_mean"]
    nbr2_mean = sentinel2["NBR2_mean"]
    vv_mean = sentinel1["VV_mean"]
    vh_mean = sentinel1["VH_mean"]
    vv_vh_ratio = sentinel1["VV_VH_ratio_mean"]
    lst_mean = max(0.0, 42.0 - 18.0 * ndvi_mean)
    ph = soil_data["ph"]
    ec = soil_data["ec"]
    oc = soil_data["oc"]
    elevation = terrain["elevation"]
    slope = terrain["slope"]
    rainfall_sum = rainfall["rainfall_sum"]
    rainfall_mean = rainfall["rainfall_mean"]
    rainfall_max = rainfall["rainfall_max"]
    rainy_days = rainfall["rainy_days"]

    # Moisture proxy from NDMI plus rainfall intensity for EC categorization.
    ndmi_norm = max(0.0, min((ndmi_mean + 1.0) / 2.0, 1.0))
    rain_norm = max(0.0, min(rainfall_mean / 10.0, 1.0))
    soil_moisture = (ndmi_norm * 0.7) + (rain_norm * 0.3)

    oc_value = _to_float_or_none(soil_data.get("oc"))
    ec_estimated_value, ec_category = _estimate_ec(
        soil_moisture=soil_moisture,
        clay_content=soil_data.get("clay"),
        oc_value=oc_value,
        rainfall_mean=rainfall_mean,
    )

    logger.info("OC VALUE: %s", oc_value)
    logger.info("EC ESTIMATION: %s %s", ec_estimated_value, ec_category)

    soil_type_lookup = {
        0: soil_data["soil_type"],
        1: "Sandy",
        2: "Loamy",
        3: "Clayey",
    }
    soil_type = soil_type_lookup.get(gee_soil_type.get("soil_type", 0), soil_data["soil_type"])

    features = {
        "NDVI_mean": ndvi_mean,
        "NDMI_mean": ndmi_mean,
        "BSI_mean": bsi_mean,
        "NBR2_mean": nbr2_mean,
        "VV_mean": vv_mean,
        "VH_mean": vh_mean,
        "VV_VH_ratio_mean": vv_vh_ratio,
        "elevation": elevation,
        "slope": slope,
        "LST_mean": lst_mean,
        "rainfall_sum": rainfall_sum,
        "rainfall_mean": rainfall_mean,
        "rainfall_max": rainfall_max,
        "rainy_days": rainy_days,
        "soil_type": soil_type,
        "month": month,
        "ph": ph,
        "ec": ec,
        "oc": oc,
        "NDVI_rainfall": ndvi_mean * rainfall_mean,
        "OC_rainfall": oc * rainfall_mean,
        "pH_OC": ph * oc,
        "slope_rainfall": slope * rainfall_mean,
        "temp_moisture": lst_mean * rainfall_mean,
        "NDVI_OC": ndvi_mean * oc,
        "NDMI_rain": ndmi_mean * rainfall_mean
    }

    meta = {
        "satellite_source": "gee",
        "soil_source": "soilgrids",
        "terrain_source": "gee",
        "rainfall_source": "gee",
        "api_key_used": False,
        "properties": {
            "oc": oc_value,
            "ec": {
                "value": ec_estimated_value,
                "category": ec_category,
            },
        },
    }

    return features, meta

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
        
        ndvi_mean = float(row["NDVI_mean"])
        ndmi_mean = float(row["NDMI_mean"])
        bsi_mean = float(row["BSI_mean"])
        nbr2_mean = float(row["NBR2_mean"])
        vv_mean = float(row["VV_mean"])
        vh_mean = float(row["VH_mean"])
        vv_vh_ratio = float(row["VV_VH_ratio_mean"])
        elevation = float(row["elevation"])
        slope = float(row["slope"])
        lst_mean = float(row["LST_mean"])
        rainfall_sum = float(row["rainfall_sum"])
        rainfall_mean = float(row["rainfall_mean"])
        rainfall_max = float(row["rainfall_max"])
        rainy_days = int(row["rainy_days"])
        soil_type = str(row["soil_type"])
        month = int(row["month"])
        ph = float(row["ph"])
        ec = float(row["ec"])
        oc = float(row["oc"])

        features = {
            "NDVI_mean": ndvi_mean,
            "NDMI_mean": ndmi_mean,
            "BSI_mean": bsi_mean,
            "NBR2_mean": nbr2_mean,
            "VV_mean": vv_mean,
            "VH_mean": vh_mean,
            "VV_VH_ratio_mean": vv_vh_ratio,
            "elevation": elevation,
            "slope": slope,
            "LST_mean": lst_mean,
            "rainfall_sum": rainfall_sum,
            "rainfall_mean": rainfall_mean,
            "rainfall_max": rainfall_max,
            "rainy_days": rainy_days,
            "soil_type": soil_type,
            "month": month,
            "ph": ph,
            "ec": ec,
            "oc": oc,
            "NDVI_rainfall": ndvi_mean * rainfall_mean,
            "OC_rainfall": oc * rainfall_mean,
            "pH_OC": ph * oc,
            "slope_rainfall": slope * rainfall_mean,
            "temp_moisture": lst_mean * rainfall_mean,
            "NDVI_OC": ndvi_mean * oc,
            "NDMI_rain": ndmi_mean * rainfall_mean,
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
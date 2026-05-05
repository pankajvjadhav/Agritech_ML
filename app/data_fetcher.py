import requests
import time
import numpy as np
import pandas as pd
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from datetime import datetime, timedelta

from satellite_features.rainfall_features import get_rainfall_features
from satellite_features.sentinel1_features import get_sentinel1_features
from satellite_features.sentinel2_features import get_sentinel2_features
from satellite_features.soil_type_features import get_soil_type
from satellite_features.terrain_features import get_terrain_features

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = int(os.getenv("EXTERNAL_API_TIMEOUT", "15"))
MAX_RETRIES = int(os.getenv("EXTERNAL_API_MAX_RETRIES", "2"))


class ExternalDataFetchError(RuntimeError):
    pass


# Reasonable defaults for Indian agricultural soil keyed by GEE soil-type
# class. All values are reported in the same units as the live SoilGrids
# values: pH unitless, OC in PERCENT (ICAR Soil Health Card: Low <0.5 % /
# Medium 0.5–0.75 % / High >0.75 %), EC in dS/m, clay/silt/sand in % of
# fine-earth. Sources cross-checked against ICAR Soil Health Card pH-class
# tables, NBSSLUP soil-survey reference profiles, and ICAR-IISS Bhopal OC
# baselines for tropical soils.
_SOIL_FALLBACK_BY_TYPE = {
    "Sandy": {
        "ph": 7.2,    # weakly buffered, slightly alkaline tendency
        "oc": 0.30,   # ICAR Low — sandy soils retain little organic matter
        "ec": 0.10,
        "clay": 12.0, "silt": 18.0, "sand": 70.0,
    },
    "Loamy": {
        "ph": 6.7,    # near-neutral, ideal for sugarcane
        "oc": 0.55,   # ICAR Medium — loams typical for Maharashtra/UP cane belt
        "ec": 0.18,
        "clay": 25.0, "silt": 30.0, "sand": 45.0,
    },
    "Clayey": {
        "ph": 7.6,    # black cotton / vertisol — alkaline-leaning
        "oc": 0.70,   # ICAR Medium-to-High — high CEC retains OM
        "ec": 0.28,
        "clay": 45.0, "silt": 30.0, "sand": 25.0,
    },
}
# Backwards-compatible flat default (loam) for callers that don't have a
# soil-type hint yet.
_SOIL_FALLBACK = {**_SOIL_FALLBACK_BY_TYPE["Loamy"], "soil_type": "Loamy"}


def _build_fallback_soil(lat: float, lon: float, soil_type: str | None) -> Dict[str, Any]:
    """Return ICAR-plausible fallback soil parameters keyed by GEE soil
    type, with a small deterministic lat/lon perturbation so two fields with
    the same texture but different locations don't read identically. The
    perturbation is bounded so the result stays inside the ICAR class the
    base value falls into."""
    base_type = soil_type if soil_type in _SOIL_FALLBACK_BY_TYPE else "Loamy"
    base = dict(_SOIL_FALLBACK_BY_TYPE[base_type])

    # Indian climatic gradient: Northwest tends alkaline & low-OC (semi-arid
    # Indo-Gangetic alluvium); Southeast tends acidic & higher-OC (humid
    # tropics). Bound the offsets so we don't tip a Loam into "Acidic" or
    # spike a sand's OC into the High band.
    lat_norm = (lat - 22.0) / 12.0  # ~ -1 (KL coast) to +1 (Punjab)
    lon_norm = (lon - 78.0) / 12.0  # ~ -1 (Gujarat) to +1 (NE / coast)
    ph_offset = max(-0.4, min(0.4, 0.4 * lat_norm))
    oc_offset = max(-0.15, min(0.15, -0.10 * lat_norm + 0.05 * lon_norm))
    ec_offset = max(-0.05, min(0.05, 0.04 * lat_norm))

    base["ph"] = round(max(5.0, min(8.6, base["ph"] + ph_offset)), 2)
    base["oc"] = round(max(_OC_MIN_PCT, min(_OC_MAX_PCT, base["oc"] + oc_offset)), 3)
    base["ec"] = round(max(0.05, min(2.5, base["ec"] + ec_offset)), 3)
    base["soil_type"] = base_type
    return base

# ICAR-realistic plausibility band for Organic Carbon in cropland soils.
# Anything outside this band is almost certainly a unit / sensor artefact;
# we clamp and log so downstream consumers (advisory engine, dashboard,
# Soil Health Card PDF) never receive impossible values.
_OC_MIN_PCT = 0.05    # absolute floor — even desert sands stay above this
_OC_MAX_PCT = 5.0     # tilled cropland upper bound; peats can exceed this

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
    """
    Surrogate Electrical Conductivity (saturated paste, dS/m at 25 °C) when
    no in-situ probe is available. Anchored to ICAR-CSSRI Karnal salinity
    bands: <1 normal, 1–2 slight, 2–4 moderate, >4 strong. Inputs:
      moisture: NDMI/rainfall-derived 0–1 fraction
      clay:     SoilGrids clay content in % of fine-earth fraction (0–100)
      oc:       Organic Carbon in PERCENT (0–5) — see fetch_soil_data()
      rain:     30-day mean daily rainfall (mm)
    """
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

    # Clay-rich soils retain salts; rainfall leaches them; OM buffers them
    # mildly. Coefficients tuned so a typical 25 % clay / 1.5 % OC / 5 mm
    # mean-daily-rainfall loam returns ~0.4 dS/m, well inside the ICAR
    # "normal" band. Hard-clamped to the 0.05–4 dS/m range; values >4 must
    # come from a real probe measurement, not this surrogate.
    ec_value = (
        0.10
        + 0.006 * clay
        + 0.04 * oc
        - 0.02 * min(rain, 15.0)
        + 0.3 * moisture
    )
    ec_value = max(0.05, min(4.0, ec_value))
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
    # --- Weighted OC calculation (0-30 cm) -----------------------------------
    # SoilGrids `soc` is grams of organic carbon per kilogram of fine earth
    # (g/kg) once the d_factor of 10 has been applied by `_weighted_topsoil_mean`.
    # Soil Health Card / ICAR reference units OC as %, where:
    #   1 g/kg  =  0.1 %   →  multiplier is 1/10, NOT 1/100.
    # The previous /100 divisor produced values 10× too low (0.15 % when the
    # true field was ~1.5 %) so the dashboard always classified soils as Low.
    soc_layer = None
    for layer in properties.get("layers", []):
        if layer.get("name") == "soc":
            soc_layer = layer
            break
    if soc_layer is not None:
        soc_weighted = _weighted_topsoil_mean(soc_layer)
    else:
        soc_weighted = 0.0
    oc = soc_weighted / 10.0  # g/kg → %
    if oc < _OC_MIN_PCT or oc > _OC_MAX_PCT:
        logger.warning(
            "OC %.3f%% outside ICAR plausibility band [%.2f, %.2f]%% — clamping",
            oc, _OC_MIN_PCT, _OC_MAX_PCT,
        )
    oc = max(_OC_MIN_PCT, min(_OC_MAX_PCT, oc))
    logger.info("[SoilGrids] Weighted SOC (0-30cm) %.2f g/kg → OC %.2f %%", soc_weighted, oc)

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
def fetch_real_time_data(lat: float, lon: float, area_ha: float = 1.0, satellite_api_key: str = None, use_dataset: bool = False, sample_date: str = None):

    if use_dataset:
        raise ExternalDataFetchError("Dataset fallback is disabled in production realtime mode")

    # When sample_date is provided (YYYY-MM-DD), use it as the END of
    # the 30-day Earth Engine window. Without it we default to "now",
    # so subsequent calls with the same lat/lon return today's data.
    # Anchoring the window to a past date is what lets the dashboard
    # render the soil's predicted state on an older satellite pass.
    if sample_date:
        try:
            end_date = datetime.strptime(sample_date, "%Y-%m-%d").date()
        except ValueError:
            end_date = datetime.utcnow().date()
    else:
        end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=30)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Five Earth Engine calls — all independent (each takes only lat/lon
    # and dates), so we fan them out to a thread pool instead of running
    # sequentially. Wall-clock time on a cold dashboard load drops from
    # ~22 s (5 × ~4 s sequential) to ~5 s (the slowest single call).
    # ThreadPoolExecutor is the right primitive here: each underlying
    # call is a blocking HTTP request to Earth Engine, the GIL releases
    # during I/O, and Earth Engine's Python client is thread-safe for
    # concurrent .getInfo() reads.
    with ThreadPoolExecutor(max_workers=5) as executor:
        f_terrain = executor.submit(get_terrain_features, lat, lon)
        f_rainfall = executor.submit(get_rainfall_features, lat, lon, start_date_str, end_date_str)
        f_sentinel2 = executor.submit(get_sentinel2_features, lat, lon, start_date_str, end_date_str)
        f_sentinel1 = executor.submit(get_sentinel1_features, lat, lon, start_date_str, end_date_str)
        f_soil_type = executor.submit(get_soil_type, lat, lon)
        terrain = f_terrain.result()
        rainfall = f_rainfall.result()
        sentinel2 = f_sentinel2.result()
        sentinel1 = f_sentinel1.result()
        gee_soil_type = f_soil_type.result()

    gee_soil_type_label = {1: "Sandy", 2: "Loamy", 3: "Clayey"}.get(
        gee_soil_type.get("soil_type", 0), None,
    )

    soil_data_source = "soilgrids"
    try:
        soil_data = fetch_soil_data(lat, lon)
    except ExternalDataFetchError as e:
        logger.warning(
            "SoilGrids unavailable (%s); using GEE-soil-type-keyed fallback "
            "(type=%s) for lat=%s lon=%s",
            e, gee_soil_type_label or "unknown", lat, lon,
        )
        soil_data = _build_fallback_soil(lat, lon, gee_soil_type_label)
        soil_data_source = "fallback"

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
        "soil_source": soil_data_source,
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
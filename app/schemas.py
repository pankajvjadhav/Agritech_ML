from pydantic import BaseModel
from typing import Dict, Optional, Any, Literal


# -------------------------------
# EXISTING FEATURE INPUT (UNCHANGED)
# -------------------------------
class FeatureRequest(BaseModel):
    NDVI_mean: float
    NDMI_mean: float
    BSI_mean: float
    NBR2_mean: float
    VV_mean: float
    VH_mean: float
    VV_VH_ratio_mean: float
    elevation: float
    slope: float
    LST_mean: float
    rainfall_sum: float
    rainfall_mean: float
    rainfall_max: float
    rainy_days: int
    soil_type: Literal["Sandy", "Loamy", "Clayey"]
    month: int
    ph: float
    ec: float
    oc: float
    NDVI_rainfall: float
    OC_rainfall: float
    pH_OC: float
    slope_rainfall: float
    temp_moisture: float
    NDVI_OC: float
    NDMI_rain: float


# -------------------------------
# NEW: FARMER INPUT (ADDED)
# -------------------------------
class FarmerInput(BaseModel):
    fertilizer: Optional[Literal["Low", "Medium", "High"]] = "Medium"
    soil_type: Optional[Literal["Sandy", "Loamy", "Clayey"]] = "Loamy"
    irrigation: Optional[Literal["Rainfed", "Moderate", "Heavy"]] = "Moderate"


# -------------------------------
# NEW: HYBRID REQUEST (ADDED)
# -------------------------------
class HybridPredictionRequest(BaseModel):
    features: FeatureRequest
    farmer_input: Optional[FarmerInput] = None


# -------------------------------
# EXISTING (UNCHANGED)
# -------------------------------
class LocationRequest(BaseModel):
    lat: float
    lon: float
    area_ha: Optional[float] = 1.0
    satellite_api_key: Optional[str] = None


class IndexRequest(BaseModel):
    index: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class NutrientResult(BaseModel):
    status: str
    value: float
    unit: str
    confidence: str
    method: str


class PredictionResponse(BaseModel):
    success: bool
    predictions: Dict[str, NutrientResult]
    meta: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
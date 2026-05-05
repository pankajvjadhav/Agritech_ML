from pydantic import BaseModel, Field
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
    month: int = Field(..., ge=1, le=12)
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
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    area_ha: Optional[float] = Field(1.0, gt=0.0, le=100000.0)
    satellite_api_key: Optional[str] = None
    # When supplied (YYYY-MM-DD), the Earth Engine fetch uses the
    # 30-day window ENDING at this date instead of "today minus 30
    # days". Lets the soil dashboard surface the field's predicted
    # state on a past satellite pass — drives the "score over time"
    # demo flow.
    sample_date: Optional[str] = None


class IndexRequest(BaseModel):
    index: Optional[int] = Field(None, ge=0)
    lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    lon: Optional[float] = Field(None, ge=-180.0, le=180.0)


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

from pydantic import BaseModel
from typing import Dict, Optional, Any

class FeatureRequest(BaseModel):
    ndvi_mean_90d: float
    ndvi_trend_30d: float
    pH_0_30: float
    soc_0_30: float
    clay: float
    silt: float
    sand: float
    ndvi_std_90d: float
    ndre_mean_90d: float
    bsi_mean_90d: float
    valid_obs_count: int
    cloud_pct: float
    area_ha: float
    elevation: float
    rainfall_30d: float

class LocationRequest(BaseModel):
    lat: float
    lon: float
    area_ha: Optional[float] = 1.0
    satellite_api_key: Optional[str] = None

class IndexRequest(BaseModel):
    index: Optional[int] = None
    """Index of the row in the dataset (0-based). If None, uses closest match or random."""
    lat: Optional[float] = None
    """Optional latitude to find closest matching row."""
    lon: Optional[float] = None
    """Optional longitude to find closest matching row."""


class NutrientResult(BaseModel):
    value: Optional[float] = None
    unit: str = "mg/kg"
    confidence: float = 0.0
    method: str = "ml"


class PredictionResponse(BaseModel):
    success: bool
    predictions: Dict[str, NutrientResult]
    meta: Optional[Dict[str, Any]] = None

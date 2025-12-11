from pydantic import BaseModel
from typing import Dict, Optional

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


class NutrientResult(BaseModel):
    value: float
    unit: str = "mg/kg"
    confidence: float
    method: str = "ml"


class PredictionResponse(BaseModel):
    success: bool
    predictions: Dict[str, NutrientResult]

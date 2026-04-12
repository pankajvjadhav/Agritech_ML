import os
from fastapi import APIRouter, HTTPException
from app.schemas import (
    FeatureRequest,
    PredictionResponse,
    LocationRequest,
    IndexRequest,
    HybridPredictionRequest
)
from app.predictor import make_classification_prediction
from app.data_fetcher import (
    fetch_real_time_data,
    fetch_from_dataset,
    ExternalDataFetchError,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])


# -------------------------------
# MAIN PREDICT (HYBRID)
# -------------------------------
@router.post("", response_model=PredictionResponse)
def predict_route(payload: HybridPredictionRequest):

    features = payload.features.dict()
    farmer = payload.farmer_input

    REQUIRED_FEATURES = [
        "NDVI_mean",
        "NDMI_mean",
        "BSI_mean",
        "NBR2_mean",
        "VV_mean",
        "VH_mean",
        "VV_VH_ratio_mean",
        "elevation",
        "slope",
        "LST_mean",
        "rainfall_sum",
        "rainfall_mean",
        "rainfall_max",
        "rainy_days",
        "soil_type",
        "month",
        "ph",
        "ec",
        "oc",
        "NDVI_rainfall",
        "OC_rainfall",
        "pH_OC",
        "slope_rainfall",
        "temp_moisture",
        "NDVI_OC",
        "NDMI_rain",
    ]

    missing = [f for f in REQUIRED_FEATURES if features.get(f) is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}"
        )

    # -------------------------------
    # HYBRID PREDICTION
    # -------------------------------
    result = make_classification_prediction(
        features,
        fertilizer=farmer.fertilizer if farmer else None,
        soil_type=farmer.soil_type if farmer else None,
        irrigation=farmer.irrigation if farmer else None
    )

    return PredictionResponse(
        success=True,
        predictions=result,
        meta={"model": "hybrid"},
        properties={
            "ph": features.get("ph"),
            "ec": features.get("ec"),
            "oc": features.get("oc"),
        }
    )


# -------------------------------
# REALTIME
# -------------------------------
@router.post("/realtime", response_model=PredictionResponse)
def predict_realtime(payload: LocationRequest):
    try:
        features, meta = fetch_real_time_data(
            payload.lat,
            payload.lon,
            payload.area_ha,
            payload.satellite_api_key,
            use_dataset=False
        )
    except ExternalDataFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    predictions = make_classification_prediction(features)

    if "ph" in features:
        predictions["pH"] = {
            "status": "measured",
            "value": features["ph"],
            "unit": "unitless",
            "confidence": "high",
            "method": "measured"
        }

    properties_payload = meta.get("properties", {}) if isinstance(meta, dict) else {}
    properties_payload = {
        "ph": features.get("ph"),
        "oc": properties_payload.get("oc", features.get("oc")),
        "ec": properties_payload.get("ec", {"value": None, "category": None}),
    }

    return PredictionResponse(
        success=True,
        predictions=predictions,
        meta=meta,
        properties=properties_payload,
    )


# -------------------------------
# DATASET
# -------------------------------
@router.post("/from-dataset", response_model=PredictionResponse)
def predict_from_dataset(payload: IndexRequest):
    if os.getenv("ALLOW_DATASET_ENDPOINT", "false").lower() != "true":
        raise HTTPException(
            status_code=403,
            detail="Dataset endpoint is disabled in production mode"
        )

    try:
        features, meta = fetch_from_dataset(payload.index)
    except (FileNotFoundError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching from dataset: {str(e)}")

    predictions = make_classification_prediction(features)

    return PredictionResponse(success=True, predictions=predictions, meta=meta)


# -------------------------------
# DATASET INFO
# -------------------------------
@router.get("/dataset-info")
def get_dataset_info():
    if os.getenv("ALLOW_DATASET_ENDPOINT", "false").lower() != "true":
        raise HTTPException(
            status_code=403,
            detail="Dataset info endpoint is disabled in production mode"
        )

    try:
        import pandas as pd
        from app.data_fetcher import DATASET_PATH

        if not os.path.exists(DATASET_PATH):
            raise HTTPException(status_code=404, detail=f"Dataset file not found at {DATASET_PATH}")

        df = pd.read_csv(DATASET_PATH)

        return {
            "dataset_path": DATASET_PATH,
            "total_rows": len(df),
            "available_indices": list(range(len(df))),
            "columns": list(df.columns),
            "sample_row": df.iloc[0].to_dict() if len(df) > 0 else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset info: {str(e)}")
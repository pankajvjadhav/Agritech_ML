import os
from fastapi import APIRouter, HTTPException
from app.schemas import FeatureRequest, PredictionResponse, LocationRequest, IndexRequest
from app.model_loader import model
from app.predictor import make_prediction
from app.data_fetcher import fetch_real_time_data, fetch_from_dataset, find_closest_dataset_row

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("", response_model=PredictionResponse)
def predict(payload: FeatureRequest):
    features = payload.dict()
    result = make_prediction(model, features)
    return PredictionResponse(success=True, predictions=result)

@router.post("/realtime", response_model=PredictionResponse)
def predict_realtime(payload: LocationRequest):
    """
    Predict nutrients using data from static dataset (randomly selected).
    When lat/lon is provided, randomly selects a row from static_dataset.csv.
    """
    try:
        # Use dataset by default (randomly select from static dataset)
        features, meta = fetch_real_time_data(payload.lat, payload.lon, payload.area_ha, None, use_dataset=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching from dataset: {str(e)}")

    result = make_prediction(model, features)
    return PredictionResponse(success=True, predictions=result, meta=meta)

@router.post("/from-dataset", response_model=PredictionResponse)
def predict_from_dataset(payload: IndexRequest):
    """
    Predict nutrients using data from the static dataset by index.
    This endpoint uses the dataset instead of API calls.
    """
    try:
        features, meta = fetch_from_dataset(payload.index)
    except (FileNotFoundError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching from dataset: {str(e)}")

    result = make_prediction(model, features)
    return PredictionResponse(success=True, predictions=result, meta=meta)

@router.get("/dataset-info")
def get_dataset_info():
    """
    Get information about the static dataset (number of rows, available indices).
    """
    try:
        import pandas as pd
        from app.data_fetcher import DATASET_PATH
        import os
        
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

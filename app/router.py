from fastapi import APIRouter
from app.schemas import FeatureRequest, PredictionResponse
from app.model_loader import model
from app.predictor import make_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("", response_model=PredictionResponse)
def predict(payload: FeatureRequest):
    features = payload.dict()
    result = make_prediction(model, features)
    return PredictionResponse(success=True, predictions=result)

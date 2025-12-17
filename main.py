from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, numpy as np, os, time, csv, json, logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="KisaanSaathi ML Service")

from app.router import router
from app.predictor import NUTRIENT_UNITS, compute_confidence_from_model
app.include_router(router)

MODEL_PATH = "models/nutrient_model_v1.pkl"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "nutrient_model_v1")
LOGFILE = "pred_log.csv"

# Try loading model at import
model = None
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Loaded model from %s", MODEL_PATH)
except Exception as e:
    logging.exception("Failed to load model: %s", e)
    model = None

nutrient_keys = ["N","P","K","OC","pH","EC","S","Fe","Zn","Cu","B","Mn"]

def append_log(inputs: dict, preds: dict, model_version: str):
    header = ["timestamp", "model_version", "input_json", "predictions_json"]
    row = [int(time.time()), model_version, json.dumps(inputs), json.dumps(preds)]
    write_header = not os.path.exists(LOGFILE)
    with open(LOGFILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(header)
        w.writerow(row)

class InputModel(BaseModel):
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

# @app.on_event("startup")
# def check_model():
#     if model is None:
#         logging.error("Model not loaded. Predictions will fail.")

@app.get("/")
def root():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(payload: InputModel):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    x = np.array([[payload.ndvi_mean_90d, payload.ndvi_trend_30d, payload.pH_0_30, payload.soc_0_30,
                   payload.clay, payload.silt, payload.sand, payload.ndvi_std_90d,
                   payload.ndre_mean_90d, payload.bsi_mean_90d, payload.valid_obs_count,
                   payload.cloud_pct, payload.area_ha, payload.elevation, payload.rainfall_30d]])
    raw_pred = model.predict(x)
    if hasattr(raw_pred, "tolist"):
        pl = raw_pred.tolist()
    else:
        pl = list(raw_pred)
    if len(pl) == 1 and isinstance(pl[0], list):
        pl = pl[0]
    # pad/truncate
    if len(pl) < len(nutrient_keys):
        pl = pl + [None] * (len(nutrient_keys) - len(pl))
    pl = pl[:len(nutrient_keys)]

    # compute confidences
    confidences = compute_confidence_from_model(model, x, nutrient_keys=nutrient_keys)

    # compute stds when possible
    stds = None
    try:
        if isinstance(model, MultiOutputRegressor) and hasattr(model, "estimators_"):
            std_list = []
            for est in model.estimators_:
                if hasattr(est, "estimators_") and len(getattr(est, "estimators_")) > 0:
                    tree_preds = [float(tree.predict(x)[0]) for tree in est.estimators_]
                    std_list.append(float(np.std(tree_preds)))
                else:
                    std_list.append(None)
            stds = std_list
        else:
            if hasattr(model, "estimators_") and not isinstance(model, MultiOutputRegressor):
                preds = [np.array(est.predict(x)).reshape(-1) for est in model.estimators_]
                stds = np.std(np.stack(preds, axis=0), axis=0).tolist()
    except Exception:
        logging.exception("Failed computing stds")

    preds = {}
    for i, (k, v) in enumerate(zip(nutrient_keys, pl)):
        val = None if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else float(v)
        conf = float(confidences[i]) if confidences and i < len(confidences) else 0.8
        preds[k] = {
            "value": None if val is None else round(val, 6),
            "unit": NUTRIENT_UNITS.get(k, "mg/kg"),
            "confidence": round(conf, 4),
            "method": "ml",
            "std": None if not stds else (None if i >= len(stds) else (round(stds[i],6) if stds[i] is not None else None)),
            "low_confidence": (conf < 0.3)
        }

    # append to CSV log
    try:
        append_log(payload.dict(), preds, MODEL_VERSION)
    except Exception:
        logging.exception("Failed to append prediction to log")

    return {"success": True, "model_version": MODEL_VERSION, "timestamp": int(time.time()), "predictions": preds}

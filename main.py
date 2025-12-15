from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, numpy as np, os, time, csv, json, logging
from sklearn.multioutput import MultiOutputRegressor

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="KisaanSaathi ML Service")

from app.router import router
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

def compute_confidence_from_model(model, x):
    """
    Returns list of confidences for each nutrient (0..1).
    Handles:
      - model is an ensemble regressor (has estimators_ that each predict whole output)
      - model is MultiOutputRegressor (model.estimators_ is list of per-target estimators).
        If a per-target estimator itself has estimators_, compute std across its trees.
      - fallback to fixed confidence (0.8)
    """
    try:
        # Case A: single ensemble regressor that returns vector predictions per estimator
        if hasattr(model, "estimators_") and not isinstance(model, MultiOutputRegressor):
            preds = []
            for est in model.estimators_:
                p = np.array(est.predict(x), dtype=float).reshape(-1)
                preds.append(p)
            preds = np.stack(preds, axis=0)  # (n_estimators, n_targets)
            mean = preds.mean(axis=0)
            std = preds.std(axis=0)
            rel = np.abs(std) / (np.abs(mean) + 1e-6)
            alpha = 4.0
            conf_raw = 1.0 / (1.0 + alpha * rel)
            conf = np.clip(conf_raw, 0.2, 0.99)
            return [float(c) for c in conf]

        # Case B: MultiOutputRegressor => model.estimators_ is list of per-target estimators
        if isinstance(model, MultiOutputRegressor) and hasattr(model, "estimators_"):
            per_target_conf = []
            for est in model.estimators_:
                # If per-target estimator is itself an ensemble (e.g., RandomForest), use its estimators_
                if hasattr(est, "estimators_") and len(getattr(est, "estimators_")) > 0:
                    tree_preds = []
                    for tree in est.estimators_:
                        p = np.array(tree.predict(x), dtype=float).reshape(-1)  # shape (1,)
                        tree_preds.append(p[0])
                    arr = np.array(tree_preds, dtype=float)  # shape (n_trees,)
                    mean = arr.mean()
                    std = arr.std()
                    rel = abs(std) / (abs(mean) + 1e-6)
                    alpha = 4.0
                    conf_raw = 1.0 / (1.0 + alpha * rel)
                    conf_val = float(np.clip(conf_raw, 0.2, 0.99))
                    per_target_conf.append(conf_val)
                else:
                    # no internal ensemble for this target -> fallback
                    per_target_conf.append(0.8)
            if len(per_target_conf) < len(nutrient_keys):
                per_target_conf += [0.8] * (len(nutrient_keys) - len(per_target_conf))
            return [float(c) for c in per_target_conf[:len(nutrient_keys)]]

    except Exception as e:
        logging.exception("Confidence computation failed: %s", e)

    return [0.8] * len(nutrient_keys)

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
    confidences = compute_confidence_from_model(model, x)

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
            "unit": "mg/kg" if k not in ("pH",) else "pH",
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

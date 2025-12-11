# check_model.py
import joblib, numpy as np, pathlib, traceback

p = pathlib.Path("models") / "nutrient_model_v1.pkl"
print("Model path:", p.resolve())
print("Exists:", p.exists())
if p.exists():
    print("Size (MB):", p.stat().st_size / 1024 / 1024)
try:
    m = joblib.load(str(p))
    print("Loaded model type:", type(m))
    sample = np.array([[0.45,0.02,6.7,0.9,28,32,40,0.08,0.18,0.04,6,12,1.2,260,55]])
    pred = m.predict(sample)
    print("predict OK shape:", getattr(pred, "shape", None))
    print("sample prediction (first row):", pred.tolist()[0])
except Exception:
    traceback.print_exc()


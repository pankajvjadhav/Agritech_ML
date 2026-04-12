from app.predictor import make_classification_prediction
from app.model_loader import STAGE2_FEATURES

# ✅ Auto-create all required features
test_features = {f: 0 for f in STAGE2_FEATURES}

# ✅ Override important ones for Nitrogen hybrid
test_features["NDVI_mean"] = 0.6
test_features["NDVI_std"] = 0.1
test_features["rainfall_30d"] = 80

# Optional: improve realism
test_features["ph"] = 6.5
test_features["oc"] = 0.7
test_features["ec"] = 0.5

# Run prediction
result = make_classification_prediction(test_features)

print("\n✅ OUTPUT:\n")
for k, v in result.items():
    print(k, ":", v)
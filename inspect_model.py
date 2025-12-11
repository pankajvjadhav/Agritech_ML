import joblib

model = joblib.load("models/nutrient_model_v1.pkl")

print("Model type:", type(model))
print("Has estimators_?", hasattr(model, "estimators_"))

if hasattr(model, "estimators_"):
    print("Number of estimators:", len(model.estimators_))
else:
    print("No estimators_ found (this is not an ensemble model)")

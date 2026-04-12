from fastapi import FastAPI
from app.router import router

app = FastAPI(title="KisaanSaathi ML Service")

# Register API routes
app.include_router(router)

@app.get("/")
def root():
    return {
        "status": "ok",
        "mode": "nutrient_classification_with_icar"
    }
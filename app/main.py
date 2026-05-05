from dotenv import load_dotenv

# Load .env before importing the router, since router → data_fetcher → satellite_features
# eagerly initializes Google Earth Engine using EE_* env vars at module import time.
load_dotenv()

from fastapi import FastAPI

from app.router import router

app = FastAPI(title="KisaanSaathi ML Service")

# Register API routes
app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "mode": "nutrient_classification_with_icar",
    }

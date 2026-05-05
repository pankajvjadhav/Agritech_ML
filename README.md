# Agritech_ML

This repository contains a FastAPI-based ML service for predicting soil nutrient values.

## Prerequisites
- Python 3.8 or later
- Git (optional)
- (Optional) Docker

## Setup (Windows / Linux)
1. Create a virtual environment and activate it

Windows (PowerShell):

```
python -m venv venv
venv\Scripts\Activate.ps1   # or `venv\Scripts\activate` for cmd
```

Linux / macOS:

```
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

3. Model initialization (automatic)

The trained model file is not stored in Git.

If the model file is missing, it will be automatically trained
when the ML service starts and saved locally at:

models/nutrient_model_v2.pkl

(Manual training via `python train_model.py` is optional.)

A trained model will be saved to `models/nutrient_model_v2.pkl`.

4. Run the app

```
venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
RunTest(optional)
pytest tests/
```

## Notes
- `train_model.py` uses `pandas` for data generation and model training, so `pandas` is included in `requirements.txt`.
- `pytest` is included so tests can be run via the `pytest` command.

- The `app/data_fetcher.py` functions make network calls to public APIs; they handle failures and return defaults when requests fail.
 - Satellite API keys (e.g., Agromonitoring) are optional for the realtime endpoint. You can provide them via the environment variable `AGROMONITOR_API_KEY` 
 or pass as `--api-key` to `test_predict.py`.


6. Test script format options

`test_predict.py` supports multiple output formats using the `--format` flag:

```
# Default: human-friendly pretty print (JSON + readable list)
python test_predict.py --format pretty

# JSON only
python test_predict.py --format json

# CSV output (columns: code,value,unit,confidence,method)
python test_predict.py --format csv

# HTTP testing with realtime endpoint and API key
python test_predict.py --http --realtime --lat 12.34 --lon 56.78 --api-key YOUR_KEY --format csv

# Local realtime: fetch data locally and use local model
python test_predict.py --realtime --lat 12.34 --lon 56.78 --api-key YOUR_KEY --format csv
```
docker run -p 8000:8000 agritech-ml
```

## Environment
- Optional environment variable: `MODEL_VERSION` — used in responses and logs.

## Contact
For further help, add an issue or reach out to the maintainer.

## ML Model Setup

The trained model file (`.pkl`) is not stored in Git.

When the ML service starts and the model file is missing,
it will automatically train the model and save it locally at:

models/nutrient_model_v2.pkl

No manual steps are required.

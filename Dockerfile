# ml_service/Dockerfile
FROM python:3.10.14-slim AS builder

WORKDIR /app

# Install build-time deps only in the builder layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /app/requirements.txt


FROM python:3.10.14-slim AS runtime

# Create an unprivileged user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy installed Python packages from the builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser app /app/app
COPY --chown=appuser:appuser config /app/config
COPY --chown=appuser:appuser satellite_features /app/satellite_features

# Models directory (mount actual models at runtime)
RUN mkdir -p /app/models && chown -R appuser:appuser /app/models

USER appuser

EXPOSE 8000

ENV MODEL_PATH=/app/models/nutrient_model_v2.pkl \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

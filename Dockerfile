# ml_service/Dockerfile
FROM python:3.10-slim

# set workdir
WORKDIR /app

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY app /app/app
COPY main.py /app/main.py

# create models dir and expect model to be mounted there
RUN mkdir -p /app/models

# expose port
EXPOSE 8000

# default env (can be overridden)
ENV MODEL_PATH=/app/models/nutrient_model_v1.pkl
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

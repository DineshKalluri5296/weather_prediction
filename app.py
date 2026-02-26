import boto3
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import os
import time
import mlflow

# -----------------------------
# Prometheus Metrics
# -----------------------------
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

prediction_requests = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction"
)

model_accuracy = Gauge(
    "model_accuracy",
    "Current model accuracy"
)

# -----------------------------
# MLflow Config
# -----------------------------
mlflow.set_tracking_uri("http://52.54.86.23:5000")  # <-- your MLflow server
mlflow.set_experiment("Seattle_weather_prediction256")

# -----------------------------
# S3 Config
# -----------------------------
BUCKET_NAME = "seattle-ml-app"
MODEL_KEY = "models/latest/model.pkl"
LOCAL_MODEL_PATH = "model.pkl"

app = FastAPI(title="Seattle Weather Prediction API")

model = None


# -----------------------------
# Download Model
# -----------------------------
def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        s3 = boto3.client("s3")
        s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)


# -----------------------------
# Startup Event
# -----------------------------
@app.on_event("startup")
def load_model():
    global model
    download_model()
    model = joblib.load(LOCAL_MODEL_PATH)

    # If you saved accuracy during training, set it here
    # Replace 0.87 with actual value if available
    model_accuracy.set(0.832)


# -----------------------------
# Input Schema
# -----------------------------
class WeatherInput(BaseModel):
    precipitation: float
    temp_max: float
    temp_min: float
    wind: float


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: WeatherInput):

    try:
        global model

        if model is None:
            return {"error": "Model not loaded"}

        start_time = time.time()
        prediction_requests.inc()

        input_df = pd.DataFrame([{
            "precipitation": data.precipitation,
            "temp_max": data.temp_max,
            "temp_min": data.temp_min,
            "wind": data.wind
        }])

        prediction = model.predict(input_df)[0]

        latency = time.time() - start_time
        prediction_latency.observe(latency)

        return {
            "prediction": str(prediction),
        }

    except Exception as e:
        return {"error": str(e)}
# -----------------------------
# Prometheus Metrics Endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# import boto3
# import pandas as pd
# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# import uvicorn
# import joblib
# import os
# import mlflow

# # -----------------------------
# # Configuration
# # -----------------------------

# BUCKET_NAME = "seattle-ml-app"
# MODEL_KEY = "model.pkl"
# LOCAL_MODEL_PATH = "model.pkl"

# MLFLOW_TRACKING_URI = "http://98.80.75.155:5000/"
# EXPERIMENT_NAME = "Seattle_weather_prediction13"

# app = FastAPI(title="Seattle Weather Prediction API")

# model = None

# # -----------------------------
# # Download Model
# # -----------------------------

# def download_model():

#     if os.path.exists(LOCAL_MODEL_PATH):
#         print("Model already exists locally.")
#         return

#     s3 = boto3.client("s3")

#     print("Downloading model from S3...")

#     s3.download_file(
#         BUCKET_NAME,
#         MODEL_KEY,
#         LOCAL_MODEL_PATH
#     )

#     print("Model downloaded successfully.")

# # -----------------------------
# # Load Model at Startup
# # -----------------------------

# @app.on_event("startup")
# def load_model():

#     global model

#     try:
#         download_model()
#         model = joblib.load(LOCAL_MODEL_PATH)
#         print("Model loaded successfully")

#     except Exception as e:
#         print("Model Loading Error:", str(e))

# # -----------------------------
# # Request Schema
# # -----------------------------

# class WeatherInput(BaseModel):
#     precipitation: float
#     temp_max: float
#     temp_min: float
#     wind: float

# # -----------------------------
# # Async MLflow Inference Logger
# # -----------------------------
# def log_inference_metrics(data_dict, prediction_value):

#     try:
#         mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#         experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

#         if experiment is None:
#             mlflow.create_experiment(EXPERIMENT_NAME)

#         mlflow.set_experiment(EXPERIMENT_NAME)
#         # run_name = "prediction_" + datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
#         with mlflow.start_run(run_name="fastapi_inference"):

#             # Log input features
#             mlflow.log_params(data_dict)

#             # Log prediction as tag (NOT metric)
#             mlflow.set_tag("prediction", str(prediction_value))

#     except Exception as e:
#         print("MLflow Logging Error:", str(e))

# # -----------------------------
# # Prediction Endpoint (Non-blocking inference)
# # -----------------------------
# @app.post("/predict")
# def predict(data: WeatherInput, background_tasks: BackgroundTasks):

#     try:
#         global model

#         if model is None:
#             return {"error": "Model not loaded"}

#         input_df = pd.DataFrame([data.dict()])
#         prediction = model.predict(input_df)

#         pred_value = prediction[0]

#         # Auto create MLflow run for every prediction request
#         background_tasks.add_task(
#             log_inference_metrics,
#             data.dict(),
#             pred_value
#         )

#         return {
#             "prediction": pred_value
#         }

#     except Exception as e:
#         return {"error": str(e)}
# # -----------------------------
# # Server Start
# # -----------------------------

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
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
mlflow.set_tracking_uri("http://3.88.182.216:5000")  # <-- your MLflow server
mlflow.set_experiment("Seattle_weather_prediction13")

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

    # -----------------------------
    # MLflow Logging (per request)
    # -----------------------------
    with mlflow.start_run():
        mlflow.log_param("precipitation", data.precipitation)
        mlflow.log_param("temp_max", data.temp_max)
        mlflow.log_param("temp_min", data.temp_min)
        mlflow.log_param("wind", data.wind)
        mlflow.log_metric("prediction_latency", latency)
        mlflow.log_metric("prediction_output", float(prediction))

    return {
        "prediction": str(prediction),
        "latency_seconds": latency
    }


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

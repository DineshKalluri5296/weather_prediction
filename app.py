import boto3
import pandas as pd
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import joblib
import os
import mlflow

# -----------------------------
# Configuration
# -----------------------------

BUCKET_NAME = "seattle-ml-app"
MODEL_KEY = "models/latest/model.pkl"
LOCAL_MODEL_PATH = "model.pkl"

MLFLOW_TRACKING_URI = "http://98.80.75.155:5000/"
EXPERIMENT_NAME = "Seattle_weather123"

app = FastAPI(title="Seattle Weather Prediction API")

model = None

# -----------------------------
# Download Model
# -----------------------------

def download_model():

    if os.path.exists(LOCAL_MODEL_PATH):
        print("Model already exists locally.")
        return

    s3 = boto3.client("s3")

    print("Downloading model from S3...")

    s3.download_file(
        BUCKET_NAME,
        MODEL_KEY,
        LOCAL_MODEL_PATH
    )

    print("Model downloaded successfully.")

# -----------------------------
# Load Model at Startup
# -----------------------------

@app.on_event("startup")
def load_model():

    global model

    try:
        download_model()
        model = joblib.load(LOCAL_MODEL_PATH)
        print("Model loaded successfully")

    except Exception as e:
        print("Model Loading Error:", str(e))

# -----------------------------
# Request Schema
# -----------------------------

class WeatherInput(BaseModel):
    precipitation: float
    temp_max: float
    temp_min: float
    wind: float

# -----------------------------
# Async MLflow Inference Logger
# -----------------------------
def log_inference_metrics(data_dict, prediction_value):

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            mlflow.create_experiment(EXPERIMENT_NAME)

        mlflow.set_experiment(EXPERIMENT_NAME)
        # run_name = "prediction_" + datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        with mlflow.start_run(run_name="fastapi_inference"):

            # Log input features
            mlflow.log_params(data_dict)

            # Log prediction as tag (NOT metric)
            mlflow.set_tag("prediction", str(prediction_value))

    except Exception as e:
        print("MLflow Logging Error:", str(e))

# -----------------------------
# Prediction Endpoint (Non-blocking inference)
# -----------------------------
@app.post("/predict")
def predict(data: WeatherInput, background_tasks: BackgroundTasks):

    try:
        global model

        if model is None:
            return {"error": "Model not loaded"}

        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)

        pred_value = prediction[0]

        # Auto create MLflow run for every prediction request
        background_tasks.add_task(
            log_inference_metrics,
            data.dict(),
            pred_value
        )

        return {
            "prediction": pred_value
        }

    except Exception as e:
        return {"error": str(e)}
# -----------------------------
# Server Start
# -----------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import boto3
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import os
import mlflow

BUCKET_NAME = "seattle-ml-app"
MODEL_KEY = "models/latest/model.pkl"
LOCAL_MODEL_PATH = "model.pkl"

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
# Startup Event
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
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict(data: WeatherInput):

    try:
        global model

        if model is None:
            return {"error": "Model not loaded"}

        mlflow.set_tracking_uri("http://98.80.75.155:5000/")

        experiment_name = "Seattle_weather_prediction12"

        # Create experiment if not exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)

        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)

        pred_value = float(prediction[0])

        # Logging inference metrics safely
        with mlflow.start_run(run_name="fastapi_inference", nested=False):

            mlflow.log_params(data.dict())
            mlflow.log_metric("prediction", pred_value)

        return {"prediction": pred_value}

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Server Start
# -----------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

import boto3
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import os

# -----------------------------
# S3 Model Configuration
# -----------------------------

BUCKET_NAME = "seattle-ml-app"
MODEL_KEY = "models/latest/model.pkl"
LOCAL_MODEL_PATH = "model.pkl"

def download_model():

    if os.path.exists(LOCAL_MODEL_PATH):
        print("Model already exists locally.")
        return

    try:
        s3 = boto3.client("s3")

        print("Downloading model from S3...")

        s3.download_file(
            BUCKET_NAME,
            MODEL_KEY,
            LOCAL_MODEL_PATH
        )

        print("Model downloaded successfully.")

    except Exception as e:
        print("S3 Model Download Error:", str(e))
        raise Exception("Failed to download model from S3")

# -----------------------------
# Load Model
# -----------------------------

download_model()
model = joblib.load(LOCAL_MODEL_PATH)

# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(title="Seattle Weather Prediction API")

# Request Schema
class WeatherInput(BaseModel):
    precipitation: float
    temp_max: float
    temp_min: float
    wind: float

# Prediction Endpoint
@app.post("/predict")
def predict(data: WeatherInput):

    try:
        input_df = pd.DataFrame([data.dict()])

        prediction = model.predict(input_df)

        return {
            "prediction": str(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Server Start
# -----------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

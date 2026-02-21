import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
import joblib
# -----------------------------
# 1️⃣ Set MLflow Tracking Server
# -----------------------------

model = joblib.load("model.pkl")
mlflow.set_tracking_uri("http://98.80.75.155:5000/")

# Optional: create separate experiment for inference logs
mlflow.set_experiment("Seattle_weather_prediction23")

# # Load latest model version
# model_name = "SeattleWeatherModel"
# model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

# -----------------------------
# 2️⃣ Create FastAPI App
# -----------------------------
app = FastAPI(title="Seattle Weather Prediction API")

# -----------------------------
# 3️⃣ Define Request Schema
# -----------------------------
class WeatherInput(BaseModel):
    precipitation: float
    temp_max: float
    temp_min: float
    wind: float

# -----------------------------
# 4️⃣ Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: WeatherInput):

    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)

        # Convert to Python native type
        pred_value = prediction[0]

        with mlflow.start_run(run_name="inference_run"):
            mlflow.log_params(data.dict())

            # Only log metric if numeric
            if isinstance(pred_value, (int, float)):
                mlflow.log_metric("prediction", float(pred_value))
            else:
                mlflow.set_tag("prediction", str(pred_value))

        return {
            "prediction": str(pred_value)
        }

    except Exception as e:
        return {"error": str(e)}
# -----------------------------
# 5️⃣ Run Server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

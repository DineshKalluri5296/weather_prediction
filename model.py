import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

mlflow.set_tracking_uri("http://100.54.111.135:5000")   
mlflow.set_experiment("Seattle_weather12")

df = pd.read_csv("seattle-weather.csv")
df = df.dropna()

X = df.drop(["date", "weather"], axis=1)
y = df["weather"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


with mlflow.start_run() as run:

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    # model=LogisticRegression()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("Accuracy:", accuracy)

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)
    # Register model (creates new version automatically)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="SeattleWeatherModel5"
    )
    joblib.dump(model, "model.pkl")
    print("Model saved locally as model.pkl")
# -----------------------------
# 4️⃣ Add Description to Latest Version
# -----------------------------
client = MlflowClient()

latest_version = client.get_latest_versions("SeattleWeatherModel5")[0].version
client.update_model_version(
    name="SeattleWeatherModel5",
    version=latest_version,
    description="Randomforestclassifier model trained on Seattle weather dataset"
)

print(f"Model Version {latest_version} updated with description successfully!")

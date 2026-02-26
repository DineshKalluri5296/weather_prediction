import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score 
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

mlflow.set_tracking_uri("http://52.54.86.23:5000")   
mlflow.set_experiment("Seattle_weather_prediction216")

df = pd.read_csv("seattle-weather.csv")
df = df.dropna()

X = df.drop(["date", "weather"], axis=1)
y = df["weather"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


with mlflow.start_run() as run:

    # model = RandomForestClassifier(
    #     n_estimators=100,
    #     max_depth=10,
    #     random_state=42
    # )
    model=LogisticRegression()
    # model=DecisionTreeClassifier()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="weighted")
    recall = recall_score(y_test, pred, average="weighted")
    f1 = f1_score(y_test, pred, average="weighted")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    report = classification_report(y_test, pred)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")
    
    # Register model (creates new version automatically)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="SeattleWeatherModel15"
    )
    joblib.dump(model, "model.pkl")
    print("Model saved locally as model.pkl")
# -----------------------------
# 4️⃣ Add Description to Latest Version
# -----------------------------
client = MlflowClient()

latest_version = client.get_latest_versions("SeattleWeatherModel15")[0].version
client.update_model_version(
    name="SeattleWeatherModel15",
    version=latest_version,
    description="RandomForestClassifier model trained on Seattle weather dataset"
)

print(f"Model Version {latest_version} updated with description successfully!")

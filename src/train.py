import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

# Set MLflow tracking URI to a local directory
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))


def train():
    data_path = "data/processed/train_labeled.csv"
    if not os.path.exists(data_path):
        print("Processed data not found. Run data_processing.py first.")
        return

    df = pd.read_csv(data_path)

    # Split Features and Target
    X = df.drop(["risk_label"], axis=1)
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("Credit_Risk_Model")

    # --- Model 1: Random Forest ---
    with mlflow.start_run(run_name="Random_Forest"):
        n_estimators = 50
        max_depth = 10

        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Random Forest Accuracy: {acc:.4f}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(rf, "model")

        # Save model locally for API
        os.makedirs("src/api", exist_ok=True)
        joblib.dump(rf, "src/api/model.pkl")

    # --- Model 2: Logistic Regression ---
    with mlflow.start_run(run_name="Logistic_Regression"):
        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Logistic Regression Accuracy: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)


if __name__ == "__main__":
    train()

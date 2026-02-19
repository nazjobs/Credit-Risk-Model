import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.config import config

mlflow.set_tracking_uri(config.mlflow_tracking_uri)

def load_processed_data() -> tuple:
    df = pd.read_csv(config.processed_data_path)
    X = df.drop(, axis=1)
    y = df
    return X, y

def train_models() -> None:
    if not os.path.exists(config.processed_data_path):
        print("Processed data not found.")
        return

    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.random_state
    )

    mlflow.set_experiment("Credit_Risk_Model")

    with mlflow.start_run(run_name="Random_Forest"):
        rf = RandomForestClassifier(
            n_estimators=config.rf_n_estimators, 
            max_depth=config.rf_max_depth, 
            random_state=config.random_state
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test) # FIXED: ROC AUC needs 1D array
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"RF - Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")

        mlflow.log_params({"n_estimators": config.rf_n_estimators, "max_depth": config.rf_max_depth})
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc})
        mlflow.sklearn.log_model(rf, "model")

        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        joblib.dump(rf, config.model_save_path)

    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(max_iter=config.lr_max_iter)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"LR - Accuracy: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)

if __name__ == "__main__":
    train_models()

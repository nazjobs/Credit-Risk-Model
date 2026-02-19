import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from src.config import config

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.to_datetime(df)
    snapshot_date = df.max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg({
            "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
            "TransactionId": "count",
            "Amount": "sum",
        })
        .reset_index()
    )
    rfm.columns =
    return rfm

def create_proxy_target(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[])

    kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.random_state, n_init=10)
    rfm = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby("Cluster")].mean()
    bad_cluster = cluster_summary.idxmax()

    rfm = rfm.apply(lambda x: 1 if x == bad_cluster else 0)
    df = df.merge(rfm[], on="CustomerId", how="left")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dt.hour
    df = df.dt.day
    df = df.dt.month

    le = LabelEncoder()
    for col in config.cat_cols:
        if col in df.columns:
            df = le.fit_transform(df.astype(str))

    numeric_cols = df.select_dtypes(include=).columns
    return df

def main() -> None:
    os.makedirs(os.path.dirname(config.processed_data_path), exist_ok=True)
    if not os.path.exists(config.raw_data_path):
        print(f"Error: {config.raw_data_path} not found.")
        return

    df = load_data(config.raw_data_path)
    rfm = create_rfm_features(df)
    df = create_proxy_target(df, rfm)
    df = feature_engineering(df)

    df = df.drop(columns=config.cols_to_drop, errors="ignore")
    df = df.fillna(0)
    df.to_csv(config.processed_data_path, index=False)
    print("Data processing complete.")

if __name__ == "__main__":
    main()

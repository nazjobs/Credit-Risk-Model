import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from xverse.transformer import WOE
import joblib
import os


def load_data(filepath):
    return pd.read_csv(filepath)


def create_rfm_features(df):
    """
    Task 4: Calculate Recency, Frequency, Monetary (RFM)
    """
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Global snapshot date (max date + 1 day)
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            {
                "TransactionStartTime": lambda x: (
                    snapshot_date - x.max()
                ).days,  # Recency
                "TransactionId": "count",  # Frequency
                "Amount": "sum",  # Monetary
            }
        )
        .reset_index()
    )

    rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]
    return rfm


def create_proxy_target(df, rfm):
    """
    Task 4: Cluster users to define 'Good' vs 'Bad' (High Risk)
    """
    # 1. Scale Data for Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # 2. KMeans Clustering (3 groups: Good, Average, Bad)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # 3. Define 'High Risk'.
    # Logic: The cluster with the Highest Recency (most inactive) and Lowest Frequency is 'High Risk'.
    cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency"]].mean()

    # We define risk=1 (Bad) if Recency is high or Frequency is very low.
    # Simple heuristic: The cluster with highest Recency is the "Bad" cluster.
    bad_cluster = cluster_summary["Recency"].idxmax()

    # 1 = High Risk (Bad), 0 = Low Risk (Good)
    rfm["risk_label"] = rfm["Cluster"].apply(lambda x: 1 if x == bad_cluster else 0)

    # Merge label back to original transaction data
    df = df.merge(rfm[["CustomerId", "risk_label"]], on="CustomerId", how="left")

    print(f"Risk Label Distribution:\n{rfm['risk_label'].value_counts()}")
    return df


def feature_engineering(df):
    """
    Task 3: Extract time features and encode categorical variables
    """
    # Time features
    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month

    # Categorical Encoding
    # We explicitly list columns we know are categorical
    cat_cols = [
        "ProviderId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
        "ProductId",
    ]

    le = LabelEncoder()

    for col in cat_cols:
        if col in df.columns:
            # Convert to string first to handle potential mixed types
            df[col] = le.fit_transform(df[col].astype(str))

    # SAFETY NET: Select only numeric columns for the final output
    # This prevents any leftover strings from crashing the model
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols]


def main():
    # Ensure directories exist
    os.makedirs("data/processed", exist_ok=True)

    input_path = "data/raw/data.csv"
    output_path = "data/processed/train_labeled.csv"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please move your csv file there.")
        return

    print("Loading data...")
    df = load_data(input_path)

    print("Creating RFM Features...")
    rfm = create_rfm_features(df)

    print("Creating Proxy Target...")
    df = create_proxy_target(df, rfm)

    print("Engineering Features...")
    df = feature_engineering(df)

    # Drop columns not needed for training (IDs and dates)
    cols_to_drop = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CurrencyCode",
        "CountryCode",
        "TransactionStartTime",
        "CustomerId",
    ]

    # Only drop columns that actually exist
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Fill any remaining NaNs (simple imputation)
    df = df.fillna(0)

    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()

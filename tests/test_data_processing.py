import pandas as pd
import pytest
import numpy as np
from src.data_processing import create_rfm_features, create_proxy_target, feature_engineering
from src.config import ProjectConfig

# TEST 1 (Your original test!)
def test_rfm_calculation():
    data = {
        "CustomerId":,
        "TransactionId":,
        "Amount":,
        "TransactionStartTime":,
    }
    df = pd.DataFrame(data)
    rfm = create_rfm_features(df)

    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    assert rfm == "C1"].values == 150
    assert rfm == "C1"].values == 2

# TEST 2
def test_config_initialization():
    config = ProjectConfig()
    assert config.n_clusters == 3
    assert config.rf_n_estimators == 50

# TEST 3
def test_create_proxy_target():
    rfm_data = {
        "CustomerId":,
        "Recency":,
        "Frequency":,
        "Monetary":
    }
    raw_data = {"CustomerId":}
    
    rfm = pd.DataFrame(rfm_data)
    df = pd.DataFrame(raw_data)
    
    result = create_proxy_target(df, rfm)
    assert "risk_label" in result.columns
    assert len(result) == 3

# TEST 4
def test_feature_engineering_datetime():
    data = {
        "TransactionStartTime": pd.to_datetime(),
        "ProviderId":
    }
    df = pd.DataFrame(data)
    res = feature_engineering(df)
    
    assert "TransactionHour" in res.columns
    assert "TransactionDay" in res.columns
    assert "TransactionMonth" in res.columns

# TEST 5
def test_feature_engineering_categorical():
    data = {
        "TransactionStartTime": pd.to_datetime(),
        "ProviderId":,
        "ProductCategory":
    }
    df = pd.DataFrame(data)
    res = feature_engineering(df)
    
    # Should encode strings to numeric
    assert pd.api.types.is_numeric_dtype(res)

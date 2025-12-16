import pandas as pd
import pytest
from src.data_processing import create_rfm_features


def test_rfm_calculation():
    # Mock Data
    data = {
        "CustomerId": ["C1", "C1"],
        "TransactionId": ["T1", "T2"],
        "Amount": [100, 50],
        "TransactionStartTime": ["2023-01-01", "2023-01-02"],
    }
    df = pd.DataFrame(data)
    # Ensure datetime conversion happens inside function or before
    # The function expects string or datetime, let's pass strings as per CSV read

    rfm = create_rfm_features(df)

    # Assertions
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns

    # C1 total amount should be 150
    assert rfm[rfm["CustomerId"] == "C1"]["Monetary"].values[0] == 150

    # Frequency should be 2
    assert rfm[rfm["CustomerId"] == "C1"]["Frequency"].values[0] == 2

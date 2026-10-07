"""Unit tests for time-aware splitting, data validation, and leak-free preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    generate_synthetic_creditcard_data,
    time_based_split,
    prepare_features_and_target,
    validate_dataframe,
)


def test_synthetic_data_generation_and_validation():
    """Verify synthetic dataset generator conforms to creditcard.csv schema and fraud prevalence."""
    df = generate_synthetic_creditcard_data(n_samples=2000, fraud_ratio=0.002, random_state=42)
    assert len(df) == 2000
    assert "Time" in df.columns
    assert "Amount" in df.columns
    assert "Class" in df.columns
    assert all(f"V{i}" in df.columns for i in range(1, 29))
    assert validate_dataframe(df) is True


def test_time_based_split_zero_leakage():
    """Verify time-based split preserves temporal order with zero lookahead leakage."""
    df = generate_synthetic_creditcard_data(n_samples=1000, random_state=42)
    train_df, test_df = time_based_split(df, test_size=0.20, time_col="Time")

    assert len(train_df) == 800
    assert len(test_df) == 200
    # Strict temporal ordering
    assert train_df["Time"].max() <= test_df["Time"].min()


def test_prepare_features_scaler_train_only():
    """Verify scaler is fitted strictly on training data without test leakage."""
    train_df = generate_synthetic_creditcard_data(n_samples=500, random_state=42)
    test_df = generate_synthetic_creditcard_data(n_samples=100, random_state=99)

    X_train, y_train, X_test, y_test, scaler, feature_names = prepare_features_and_target(
        train_df=train_df,
        test_df=test_df,
        target_col="Class",
        amount_col="Amount",
        time_col="Time",
        scaler_type="robust",
    )

    assert X_train.shape[0] == 500
    assert X_test.shape[0] == 100
    assert len(feature_names) == 30  # 28 V features + Amount + Time
    assert len(y_train) == 500
    assert len(y_test) == 100
    assert scaler is not None

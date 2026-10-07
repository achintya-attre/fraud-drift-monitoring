"""Unit tests for Population Stability Index (PSI), KS test, and score drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift import calculate_psi, calculate_ks, calculate_feature_drift, calculate_score_drift


def test_psi_identical_distributions():
    """Identical distributions should have PSI very close to 0 (< 0.01)."""
    rng = np.random.RandomState(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=5000)
    curr = rng.normal(loc=0.0, scale=1.0, size=5000)

    psi_val, breakdown = calculate_psi(ref, curr, num_bins=10)
    assert psi_val >= 0.0
    assert psi_val < 0.03
    assert not breakdown.empty


def test_psi_moderate_drift():
    """Moderate distribution shift should yield PSI in expected range (~0.10 - 0.25)."""
    rng = np.random.RandomState(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=5000)
    curr = rng.normal(loc=0.35, scale=1.05, size=5000)

    psi_val, _ = calculate_psi(ref, curr, num_bins=10)
    assert 0.08 <= psi_val <= 0.35


def test_psi_severe_drift():
    """Severe distribution shift should yield PSI >= 0.25."""
    rng = np.random.RandomState(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=5000)
    curr = rng.normal(loc=1.5, scale=1.5, size=5000)

    psi_val, _ = calculate_psi(ref, curr, num_bins=10)
    assert psi_val >= 0.25


def test_psi_epsilon_smoothing_handles_empty_bins():
    """Epsilon smoothing prevents division by zero or inf when target has empty bins."""
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 100)
    # curr has values only in higher range
    curr = np.array([8.0, 9.0, 10.0] * 300)

    psi_val, breakdown = calculate_psi(ref, curr, num_bins=5, epsilon=1e-4)
    assert np.isfinite(psi_val)
    assert psi_val > 0.5


def test_ks_test_detection():
    """Kolmogorov-Smirnov test correctly distinguishes identical vs shifted distributions."""
    rng = np.random.RandomState(42)
    sample_a = rng.normal(0, 1, 1000)
    sample_b = rng.normal(0, 1, 1000)
    sample_c = rng.normal(0.8, 1, 1000)

    _, p_same = calculate_ks(sample_a, sample_b)
    assert p_same > 0.01  # Cannot reject H0

    stat_diff, p_diff = calculate_ks(sample_a, sample_c)
    assert stat_diff > 0.15
    assert p_diff < 1e-5  # Strongly rejects H0


def test_calculate_feature_drift_table():
    """Feature drift calculation produces properly formatted DataFrame with drift flags."""
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame({
        "V1": rng.normal(0, 1, 1000),
        "V2": rng.normal(0, 1, 1000),
    })
    curr_df = pd.DataFrame({
        "V1": rng.normal(0, 1, 1000),       # Stable
        "V2": rng.normal(1.2, 1.2, 1000),  # Drifting
    })

    drift_df = calculate_feature_drift(ref_df, curr_df, feature_cols=["V1", "V2"])
    assert list(drift_df.columns) == ["feature", "psi", "ks_stat", "p_value", "drift_flag", "drift_level"]
    assert drift_df.loc[drift_df["feature"] == "V2", "drift_flag"].values[0] == True
    assert drift_df.loc[drift_df["feature"] == "V1", "drift_flag"].values[0] == False


def test_calculate_score_drift():
    """Score drift detection correctly calculates PSI and mean shifts on prediction scores."""
    rng = np.random.RandomState(42)
    ref_scores = rng.beta(0.5, 20.0, size=2000)
    curr_scores = rng.beta(2.0, 5.0, size=2000)  # Substantial shift

    res = calculate_score_drift(ref_scores, curr_scores)
    assert "score_psi" in res
    assert "ks_stat" in res
    assert res["score_drift_detected"] == True
    assert res["score_psi"] > 0.20

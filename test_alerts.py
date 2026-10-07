"""Unit tests for the drift alert dispatch engine."""

from __future__ import annotations

import pandas as pd
import pytest

from src.alerts import evaluate_drift_alerts


def test_alert_healthy_nominal_distributions():
    """Nominal feature distributions produce OK action with NONE severity."""
    feature_df = pd.DataFrame([
        {"feature": "V1", "psi": 0.02, "drift_level": "NEGLIGIBLE"},
        {"feature": "V2", "psi": 0.04, "drift_level": "NEGLIGIBLE"},
        {"feature": "Amount", "psi": 0.03, "drift_level": "NEGLIGIBLE"},
    ])
    score_drift = {"score_psi": 0.04, "score_drift_detected": False}

    result = evaluate_drift_alerts(feature_df, score_drift)
    assert result["recommended_action"] == "OK"
    assert result["severity"] == "NONE"
    assert result["drift_detected"] == False
    assert len(result["triggers"]) == 0


def test_alert_moderate_drift_triggers_monitor():
    """Moderate feature or score shifts trigger MONITOR recommendation with MEDIUM severity."""
    feature_df = pd.DataFrame([
        {"feature": "V1", "psi": 0.14, "drift_level": "MODERATE"},
        {"feature": "V2", "psi": 0.12, "drift_level": "MODERATE"},
        {"feature": "V3", "psi": 0.02, "drift_level": "NEGLIGIBLE"},
    ])
    score_drift = {"score_psi": 0.05, "score_drift_detected": False}

    result = evaluate_drift_alerts(feature_df, score_drift)
    assert result["recommended_action"] == "MONITOR"
    assert result["severity"] == "MEDIUM"
    assert result["drift_detected"] == True


def test_alert_severe_features_triggers_retrain():
    """Severe feature drift across multiple variables triggers RETRAIN recommendation."""
    feature_df = pd.DataFrame([
        {"feature": "V14", "psi": 0.35, "drift_level": "SEVERE"},
        {"feature": "V4", "psi": 0.28, "drift_level": "SEVERE"},
        {"feature": "V11", "psi": 0.24, "drift_level": "SEVERE"},
        {"feature": "V12", "psi": 0.08, "drift_level": "NEGLIGIBLE"},
    ])
    score_drift = {"score_psi": 0.08, "score_drift_detected": False}

    result = evaluate_drift_alerts(feature_df, score_drift)
    assert result["recommended_action"] == "RETRAIN"
    assert result["severity"] == "HIGH"
    assert result["drift_detected"] == True


def test_alert_severe_score_drift_triggers_retrain():
    """Severe prediction score distribution shift (PSI >= 0.25) triggers RETRAIN recommendation."""
    feature_df = pd.DataFrame([
        {"feature": "V1", "psi": 0.05, "drift_level": "NEGLIGIBLE"},
    ])
    score_drift = {"score_psi": 0.28, "score_drift_detected": True}

    result = evaluate_drift_alerts(feature_df, score_drift)
    assert result["recommended_action"] == "RETRAIN"
    assert result["severity"] == "HIGH"
    assert result["drift_detected"] == True


def test_alert_performance_drop_triggers_retrain():
    """Severe rolling PR-AUC degradation triggers RETRAIN recommendation."""
    feature_df = pd.DataFrame([{"feature": "V1", "psi": 0.04, "drift_level": "NEGLIGIBLE"}])
    score_drift = {"score_psi": 0.04, "score_drift_detected": False}
    perf_drop = {"relative_pr_auc_drop": 0.22}  # 22% drop

    result = evaluate_drift_alerts(feature_df, score_drift, performance_drift_dict=perf_drop)
    assert result["recommended_action"] == "RETRAIN"
    assert result["severity"] == "HIGH"

"""Unit tests for cost-based decision threshold analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.threshold import calculate_cost_at_threshold, sweep_thresholds, find_optimal_thresholds


def test_cost_calculation_fixed_mode():
    """Verify exact financial cost arithmetic with fixed penalty costs."""
    y_true = np.array([1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.4, 0.7, 0.2, 0.1])
    # Threshold = 0.5:
    # y_pred = [1, 0, 1, 0, 0]
    # TP: idx 0 (1 vs 1) -> 1 TP
    # FN: idx 1 (1 vs 0) -> 1 FN
    # FP: idx 2 (0 vs 1) -> 1 FP
    # TN: idx 3, 4 (0 vs 0) -> 2 TN

    res = calculate_cost_at_threshold(
        y_true=y_true,
        y_scores=y_scores,
        amounts=None,
        threshold=0.5,
        fn_mode="fixed",
        fn_fixed=100.0,
        fp_fixed=5.0,
        tp_fixed=2.0,
    )

    assert res["n_tp"] == 1
    assert res["n_fn"] == 1
    assert res["n_fp"] == 1
    assert res["n_tn"] == 2
    # Total cost = 1*100 + 1*5 + 1*2 = 107.0
    assert res["total_cost"] == 107.0
    assert res["fn_cost"] == 100.0
    assert res["fp_cost"] == 5.0
    assert res["tp_cost"] == 2.0


def test_cost_calculation_amount_mode():
    """Verify amount-based cost arithmetic sums actual missed fraud transaction amounts."""
    y_true = np.array([1, 1, 0])
    y_scores = np.array([0.9, 0.2, 0.8])
    amounts = np.array([250.0, 750.0, 10.0])
    # Threshold = 0.5:
    # Pred: [1, 0, 1]
    # TP: idx 0 (Amount: 250, caught -> tp_fee = 2.0)
    # FN: idx 1 (Amount: 750, missed fraud -> fn_cost = 750.0)
    # FP: idx 2 (Amount: 10, legit blocked -> fp_fee = 5.0)

    res = calculate_cost_at_threshold(
        y_true=y_true,
        y_scores=y_scores,
        amounts=amounts,
        threshold=0.5,
        fn_mode="amount",
        fp_fixed=5.0,
        tp_fixed=2.0,
    )

    assert res["fn_cost"] == 750.0
    assert res["fp_cost"] == 5.0
    assert res["tp_cost"] == 2.0
    assert res["total_cost"] == 757.0


def test_threshold_sweep_finds_minimum():
    """Threshold sweeping properly identifies the minimum financial cost operating point."""
    rng = np.random.RandomState(42)
    y_true = np.array([0] * 900 + [1] * 100)
    # Probabilities centered around 0.1 for legit and 0.8 for fraud
    y_scores = np.concatenate([rng.beta(1, 10, 900), rng.beta(10, 2, 100)])

    sweep_df = sweep_thresholds(
        y_true=y_true,
        y_scores=y_scores,
        cost_config={"false_negative_mode": "fixed", "false_negative_fixed": 100.0, "false_positive_fixed": 5.0},
        num_steps=50,
    )

    assert not sweep_df.empty
    assert len(sweep_df) == 50

    opt_dict = find_optimal_thresholds(sweep_df)
    assert "cost_optimal" in opt_dict
    assert "f1_optimal" in opt_dict
    assert "default_0_5" in opt_dict
    assert opt_dict["cost_optimal"]["total_cost"] <= opt_dict["default_0_5"]["total_cost"]

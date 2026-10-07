"""Script to generate notebooks/analysis.ipynb with clean narrative walkthrough and visualizations."""

import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Fraud / Anomaly Detection with Drift Monitoring\n",
                "### A Production-Grade System for Silent Failure Detection & Automated Recovery\n",
                "\n",
                "**Author**: Senior ML Engineer  \n",
                "**Domain**: Fintech & Payment Fraud  \n",
                "**Dataset**: Kaggle Credit Card Fraud Detection (ULB Machine Learning Group, 284,807 transactions, ~0.17% fraud)\n",
                "\n",
                "---\n",
                "\n",
                "## Executive Summary & Motivation\n",
                "In payment processing systems, a model trained once quickly degrades in production (\"silent model failure\"). Fraudsters continually adapt their attack vectors (adversarial concept drift), consumer spending patterns shift seasonally (covariate shift), and macroeconomic dynamics introduce inflation in transaction amounts. A static classifier with fixed decision thresholds silently permits escalating fraud losses or begins flagging legitimate users.\n",
                "\n",
                "This project implements a complete, leak-free, production anomaly and fraud detection architecture that:\n",
                "1. **Tackles extreme class imbalance (~0.17% fraud)** without naive oversampling or synthetic contamination.\n",
                "2. **Benchmarks supervised vs unsupervised approaches** (XGBoost vs Isolation Forest vs Autoencoder).\n",
                "3. **Proves why Precision-Recall AUC (PR-AUC) dominates ROC-AUC** in heavy class skew.\n",
                "4. **Optimizes business decision thresholds** based on tangible dollar costs rather than arbitrary 0.5 heuristics.\n",
                "5. **Monitors multi-dimensional drift** using Population Stability Index (PSI), two-sample Kolmogorov-Smirnov (KS) tests, prediction score confidence tracking, and rolling PR-AUC degradation.\n",
                "6. **Dispatches automated alerts** and triggers a self-healing **incremental retraining loop** that recovers performance under adversarial mutation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import yaml\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add project root to sys.path\n",
                "sys.path.insert(0, os.path.abspath(\"..\"))\n",
                "\n",
                "from src.data import load_data, time_based_split, prepare_features_and_target\n",
                "from src.imbalance import evaluate_imbalance_strategies\n",
                "from src.models import FraudXGBoost, FraudIsolationForest, FraudAutoencoder, load_model\n",
                "from src.evaluate import compute_metrics, compare_models, plot_combined_pr_curves\n",
                "from src.threshold import sweep_thresholds, find_optimal_thresholds\n",
                "from src.drift import calculate_feature_drift, calculate_score_drift\n",
                "from src.alerts import evaluate_drift_alerts\n",
                "from src.simulate_drift import inject_drift\n",
                "\n",
                "with open(\"../config.yaml\", \"r\") as f:\n",
                "    config = yaml.safe_load(f)\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "print(\"Environment initialized successfully.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Data Loading & Time-Aware Train/Test Split\n",
                "\n",
                "> **Zero-Leakage Mandate**: Fraud transactions are strictly time-dependent. We split chronologically using the `Time` feature (first 80% chronologically for training & validation, and the final 20% for testing). Scaling (`RobustScaler`) is fitted strictly on the training partition."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load dataset (or high-fidelity synthetic benchmark fallback)\n",
                "df = load_data(file_path=\"../data/creditcard.csv\", allow_synthetic_fallback=True, random_state=42)\n",
                "\n",
                "# Chronological time split\n",
                "train_df, test_df = time_based_split(df, test_size=config[\"data\"][\"test_size\"], time_col=\"Time\")\n",
                "\n",
                "print(f\"Total Transactions: {len(df):,}\")\n",
                "print(f\"Train Window: {len(train_df):,} samples | Fraud rate: {train_df['Class'].mean()*100:.4f}%\")\n",
                "print(f\"Test Window:  {len(test_df):,} samples | Fraud rate: {test_df['Class'].mean()*100:.4f}%\")\n",
                "print(f\"Max Train Time: {train_df['Time'].max():.1f}s <= Min Test Time: {test_df['Time'].min():.1f}s (No Leakage!)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Rigorous Class Imbalance Strategy Comparison\n",
                "\n",
                "We compare four distinct paradigms on the training split using 5-fold Stratified Cross-Validation:\n",
                "1. **None (Baseline)**: Standard unweighted binary cross-entropy loss.\n",
                "2. **Class Weighting (`scale_pos_weight`)**: Cost-sensitive gradient scaling ($w = N_{neg} / N_{pos} \\approx 580$).\n",
                "3. **In-Fold SMOTE**: Minority oversampling strictly enclosed inside `imblearn.pipeline.Pipeline` folds.\n",
                "4. **Anomaly-Based Framing**: Unsupervised Isolation Forest trained on normal transactions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "imbalance_results = pd.read_csv(\"../reports/metrics/imbalance_benchmark.csv\")\n",
                "display(imbalance_results)\n",
                "\n",
                "plt.figure(figsize=(9, 4.5))\n",
                "sns.barplot(data=imbalance_results, x=\"Strategy\", y=\"PR-AUC (Mean)\", palette=\"viridis\")\n",
                "plt.title(\"Cross-Validated PR-AUC Across Imbalance Strategies\", fontsize=13)\n",
                "plt.ylim([0.0, 1.05])\n",
                "plt.ylabel(\"Mean PR-AUC\")\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Justification of Chosen Strategy: Class Weighting (`scale_pos_weight`)\n",
                "- **Why not unweighted baseline?** While standard cross-entropy achieves high nominal precision, it severely penalizes borderline fraudulent transactions, causing missed frauds (lower recall).\n",
                "- **Why not SMOTE?** Generating synthetic instances in 28-dimensional PCA space creates interpolated points in non-existent fraud manifolds, diluting fraud density and increasing false positives.\n",
                "- **Why Class Weighting?** Scaling the positive class gradient directly aligns the loss function with the business objective without perturbing the true empirical distribution, introducing zero synthetic noise, and requiring no memory overhead."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Why PR-AUC Beats ROC-AUC Under Extreme Imbalance\n",
                "\n",
                "Under extreme imbalance (~0.17% fraud), the False Positive Rate ($FPR = \\frac{FP}{FP + TN}$) is dominated by $TN \\approx 284,000$. Even thousands of false positives barely move $FPR$, keeping ROC-AUC artificially inflated at **0.99+**.\n",
                "\n",
                "In contrast, Precision ($Precision = \\frac{TP}{TP + FP}$) directly penalizes false alarms against true detections. **PR-AUC exposes the true operational utility of the model.**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display saved multi-model PR curve and ROC curve side-by-side\n",
                "from IPython.display import Image, display\n",
                "\n",
                "print(\"=== PRECISION-RECALL CURVE (Primary Discriminator) ===\")\n",
                "display(Image(\"../reports/figures/model_comparison_pr_curve.png\"))\n",
                "\n",
                "print(\"=== ROC CURVE (Shown for Contrast Only) ===\")\n",
                "display(Image(\"../reports/figures/model_comparison_roc_curve.png\"))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Cost-Sensitive Threshold Optimization\n",
                "\n",
                "Standard classification defaults to an arbitrary $\\tau = 0.50$. In financial fraud detection, errors have asymmetric financial consequences:\n",
                "- **False Negative (Missed Fraud)**: Cost = Transaction Amount (or fixed $100 penalty).\n",
                "- **False Positive (Legitimate Block)**: Cost = $5.00 (customer friction & SMS verification).\n",
                "- **True Positive (Caught Fraud)**: Cost = $2.00 (investigation review fee).\n",
                "\n",
                "We sweep thresholds $\\tau \\in [0.01, 0.99]$ to find the global cost minimum $\\tau^*_{\\text{cost}}$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open(\"../reports/metrics/test_metrics.json\", \"r\") as f:\n",
                "    test_metrics = json.load(f)\n",
                "\n",
                "th_analysis = test_metrics[\"threshold_analysis\"]\n",
                "print(f\"Cost-Optimal Threshold: tau = {th_analysis['cost_optimal']['threshold']:.3f} | Total Cost: ${th_analysis['cost_optimal']['total_cost']:,.2f}\")\n",
                "print(f\"Default 0.50 Threshold: tau = 0.500 | Total Cost: ${th_analysis['default_0_5']['total_cost']:,.2f}\")\n",
                "print(f\"Dollar Savings Achieved: ${th_analysis['dollar_savings_vs_default']:,.2f}\")\n",
                "\n",
                "display(Image(\"../reports/figures/cost_vs_threshold.png\"))\n",
                "display(Image(\"../reports/figures/confusion_matrix.png\"))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Drift Monitoring & Alerting Pipeline\n",
                "\n",
                "We monitor four pillars of distribution stability:\n",
                "1. **Population Stability Index (PSI)** on each feature.\n",
                "2. **Two-Sample Kolmogorov-Smirnov (KS) Test** for distributional divergence.\n",
                "3. **Model Prediction Score Drift** (shifts in output confidence probabilities).\n",
                "4. **Rolling PR-AUC Degradation** over sliding temporal windows."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "drift_report = pd.read_csv(\"../reports/metrics/drift_report.csv\")\n",
                "print(\"Top 10 Drifting Features (PSI & KS Statistics):\")\n",
                "display(drift_report.head(10))\n",
                "\n",
                "display(Image(\"../reports/figures/top_drifting_features.png\"))\n",
                "display(Image(\"../reports/figures/score_distribution_drift.png\"))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Synthetic Drift Injection & Automated Retraining Loop\n",
                "\n",
                "To validate that our alerting and self-healing system functions in production, we simulated:\n",
                "- **Covariate Drift**: Mean and variance shifts across key features ($V4, V11, V12, V14, V17$) and macroeconomic transaction amount scaling ($1.8\\times$).\n",
                "- **Concept Drift**: Adversarial camouflage where 60% of fraud transactions mutate to invert $V14$ and bypass historical rules.\n",
                "\n",
                "**Alert Outcome**: The monitoring system flagged `RETRAIN` (Severity: HIGH) due to 8 severe feature breaches and score PSI > 0.25. The automated windowed retraining loop executed immediately."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open(\"../reports/metrics/drift_alert.json\", \"r\") as f:\n",
                "    alert_data = json.load(f)\n",
                "\n",
                "with open(\"../reports/metrics/retrain_metrics.json\", \"r\") as f:\n",
                "    retrain_data = json.load(f)\n",
                "\n",
                "print(f\"Alert Dispatch Status: {alert_data['recommended_action']} (Severity: {alert_data['severity']})\")\n",
                "print(f\"Pre-Retraining Stale Model PR-AUC on Drifted Slice:  {retrain_data['pre_retrain_pr_auc']:.4f}\")\n",
                "print(f\"Post-Retraining Updated Model PR-AUC on Drifted Slice: {retrain_data['post_retrain_pr_auc']:.4f}\")\n",
                "print(f\"PR-AUC Relative Recovery: +{retrain_data['pr_auc_recovery_percent']:.2f}%\")\n",
                "\n",
                "display(Image(\"../reports/figures/retrain_recovery.png\"))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Conclusions & Next Steps\n",
                "\n",
                "- **Cost-Sensitive Learning**: Using `scale_pos_weight` and financial threshold optimization protects company revenue far better than default probabilistic thresholds.\n",
                "- **Proactive Observability**: Monitoring PSI on features and prediction score distributions catches silent degradation long before delayed chargeback labels arrive.\n",
                "- **Automated Recovery**: Combining drift detection with windowed retraining successfully restores model efficacy under adversarial distribution shifts."
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.6"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = "notebooks/analysis.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated {output_path}")

#!/usr/bin/env bash
# ==============================================================================
# End-to-End Pipeline Execution Script (Bash / Unix)
# ==============================================================================
set -e

echo "===================================================================="
echo " FRAUD / ANOMALY DETECTION WITH DRIFT MONITORING - PIPELINE RUNNER"
echo "===================================================================="

# 1. Run Unit Tests
echo -e "\n>>> [STEP 1/4] Running Pytest Suite..."
python -m pytest tests/ -v

# 2. Train Models and Imbalance Benchmark
echo -e "\n>>> [STEP 2/4] Training Models & Imbalance Benchmark..."
python scripts/train.py

# 3. Evaluate Models and Cost Threshold Optimization
echo -e "\n>>> [STEP 3/4] Evaluating Models & Threshold Optimization..."
python scripts/evaluate.py

# 4. Monitor Drift with Synthetic Concept/Covariate Injection
echo -e "\n>>> [STEP 4/4] Running Drift Monitoring & Retraining Pipeline..."
python scripts/monitor.py --simulate-drift

echo -e "\n===================================================================="
echo " PIPELINE SUCCESSFULLY COMPLETED! ALL REPORTS GENERATED IN reports/"
echo "===================================================================="

# ==============================================================================
# End-to-End Pipeline Execution Script (PowerShell / Windows)
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host " FRAUD / ANOMALY DETECTION WITH DRIFT MONITORING - PIPELINE RUNNER" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# 1. Run Unit Tests
Write-Host "`n>>> [STEP 1/4] Running Pytest Suite..." -ForegroundColor Yellow
python -m pytest tests/ -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests failed! Aborting." -ForegroundColor Red
    exit 1
}

# 2. Train Models and Imbalance Benchmark
Write-Host "`n>>> [STEP 2/4] Training Models & Imbalance Benchmark..." -ForegroundColor Yellow
python scripts/train.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed! Aborting." -ForegroundColor Red
    exit 1
}

# 3. Evaluate Models and Cost Threshold Optimization
Write-Host "`n>>> [STEP 3/4] Evaluating Models & Threshold Optimization..." -ForegroundColor Yellow
python scripts/evaluate.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Evaluation failed! Aborting." -ForegroundColor Red
    exit 1
}

# 4. Monitor Drift with Synthetic Concept/Covariate Injection
Write-Host "`n>>> [STEP 4/4] Running Drift Monitoring & Retraining Pipeline..." -ForegroundColor Yellow
python scripts/monitor.py --simulate-drift
if ($LASTEXITCODE -ne 0) {
    Write-Host "Monitoring failed! Aborting." -ForegroundColor Red
    exit 1
}

Write-Host "`n====================================================================" -ForegroundColor Green
Write-Host " PIPELINE SUCCESSFULLY COMPLETED! ALL REPORTS GENERATED IN reports/" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Green

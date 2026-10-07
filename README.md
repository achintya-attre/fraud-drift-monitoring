# Fraud & Anomaly Detection with Drift Monitoring

## 1. Project Overview
This project implements an end-to-end Machine Learning pipeline to identify credit card fraud transactions. It addresses extreme class imbalance (0.2% fraud targets) and simulates real-time data drift monitoring to detect model degradation over time.

## 2. Pipeline Implementation Details
- **Supervised Model:** Implemented a Random Forest Classifier utilizing balanced class weights to directly accommodate sparse fraud examples without overfitting normal behavior.
- **Unsupervised Model:** Implemented an Isolation Forest trained strictly on clean, normal transaction profiles to isolate anomalies as outliers.
- **Evaluation Metric:** Evaluated performance utilizing **Precision-Recall AUC (PR-AUC)** to accurately measure minority class tracking accuracy, avoiding misleading standard accuracy scoring.

## 3. Data Drift Monitoring Analysis
Implemented an operational statistical monitoring system utilizing the **Kolmogorov-Smirnov (KS) Test** across incoming transaction distribution parameters. When live production data shifted from the training baseline, the pipeline returned a p-value < 0.05, raising a functional **🚨 DRIFT ALERT!** to automatically cue a model retraining workflow cycle.
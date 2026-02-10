## Overview
This repository contains an end-to-end machine learning workflow for predicting opioid overdose risk using large-scale prescription and healthcare data.

# Predicting Long-Term Opioid Use from PDMP Data

This repository contains an R-based machine learning pipeline for predicting **new long-term opioid use** using features derived from Prescription Drug Monitoring Program (PDMP) data.  
Due to privacy, ethical, and institutional restrictions, **individual-level data are not publicly shared**. The repository provides the full modeling and evaluation workflow and a data schema (if available).

## Repository Structure

- `1.Load Data.R`  
  Loads raw/processed input files, performs feature engineering (e.g., drug category one-hot encoding), and creates the train/test datasets.

- `2.Model Run.R`  
  Trains models and generates predicted probabilities. Models include:
  - Random Forest
  - XGBoost
  - LASSO (glmnet)
  - Logistic Regression (MLR)
  - Logistic Regression with interaction terms (MLR_comp)
  - Neural Network (nnet) with PCA
  - Elastic Net (glmnet)

- `3.Results.R`  
  Evaluates model performance and produces summary tables using:
  - ROC AUC + confidence intervals
  - Threshold strategies: Base (0.5), Youden-optimal, and F1-optimal
  - Confusion matrix counts (TP/FP/TN/FN) and derived metrics (Sensitivity, Specificity, PPV, NPV, Youden’s J, etc.)

## Requirements

### R Packages
The pipeline uses common ML and evaluation libraries including:
`caret`, `pROC`, `randomForest`, `xgboost`, `glmnet`, `nnet`, `dplyr`, `tibble`, `purrr`, `ggplot2`, `Matrix`

Install missing packages:
```r
pkgs <- c("caret","pROC","randomForest","xgboost","glmnet","nnet","dplyr","tibble","purrr","ggplot2","Matrix")
install.packages(setdiff(pkgs, rownames(installed.packages())))

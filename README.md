# Predicting Long-Term Opioid Use from PDMP Data

This repository contains an R-based machine learning pipeline for predicting **new long-term opioid use** using features derived from Prescription Drug Monitoring Program (PDMP) data.  
Due to privacy, ethical, and institutional restrictions, **individual-level data are not publicly shared**. The repository provides the full modeling and evaluation workflow and a data schema (if available).

## Repository Structure

Predicting-Long-Term-Opioid-Use-from-PDMP-Data/
├── R/
│ ├── 01_load_data.R
│ ├── 02_model_run.R
│ └── 03_results.R
├── data/
│ ├── schema_github.csv
│ └── README.md
├── run_all.R
├── README.md
├── LICENSE
└── .gitignore

- `R/01_load_data.R`  
  Loads data, performs feature engineering, and prepares training/testing sets.

- `R/02_model_run.R`  
  Trains machine learning models and generates predicted probabilities.

- `R/03_results.R`  
  Evaluates model performance, applies multiple thresholding strategies, and exports summary tables.

- `run_all.R`  
  Runs the full pipeline in the correct order.

- `data/schema_github.csv`  
  Data schema (variable names and types). No individual-level data are included.

---

## Models Included

The pipeline evaluates the following models:

- Random Forest  
- XGBoost  
- LASSO (glmnet)  
- Logistic Regression (MLR)  
- Logistic Regression with interaction terms (MLR_comp)  
- Neural Network (nnet) with PCA  
- Elastic Net (glmnet)

---

## Evaluation Strategy

Each model is evaluated under three thresholding strategies:

- **Base**: Fixed cutoff at 0.5  
- **Youden**: Cutoff that maximizes *(Sensitivity + Specificity − 1)*  
- **F1**: Cutoff that maximizes the F1 score across a grid of thresholds  

Reported metrics typically include:

- Accuracy  
- Sensitivity, Specificity  
- PPV, NPV  
- Youden’s J  
- Confusion matrix counts (TN, FP, FN, TP)  
- AUC (C-statistic) with confidence intervals  

---

## Quickstart

After cloning the repository, run the full pipeline with:

```r
source("run_all.R")
```

---

## Data Availability

Due to privacy, ethical, and institutional restrictions, individual-level PDMP data cannot be publicly shared.

This repository provides:

 - Code for data processing, modeling, and evaluation

 - A complete data schema describing all variables used

Access to the underlying data may be granted upon reasonable request and appropriate institutional approvals.

---

## Data Schema

The full data schema is available here:
 - `data/schema_github.csv`
The schema includes variable names and data types for all features used in model training and evaluation.

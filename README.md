# Predicting Long-Term Opioid Use from PDMP Data
This project trains and evaluates multiple machine learning models to identify patients at risk of developing new long-term opioid use using real-world prescription data.

The repository provides an R-based, reproducible modeling pipeline built on features derived from Prescription Drug Monitoring Program (PDMP) data, including comprehensive performance summaries and threshold optimization strategies.

Due to privacy, ethical, and institutional restrictions, individual-level data are not publicly shared; however, the full modeling and evaluation workflow, along with the data schema, is provided to support transparency and reproducibility.

Motivation
------------------------------------------------------
Early identification of patients at risk of developing long-term opioid use is critical for prevention and intervention efforts. PDMPs provide rich longitudinal prescribing information that can support risk stratification, but require robust modeling, calibration, and evaluation frameworks to ensure reliable and interpretable predictions.

Modeling Framework
------------------------------------------------------
All models are trained using a shared feature set derived from PDMP data to enable direct comparison across modeling approaches. The pipeline is designed to be modifiable and extensible, allowing alternative models, feature sets, or evaluation strategies to be incorporated within a consistent analytical framework.

Feature Engineering
------------------------------------------------------
Features are derived from prescription histories and patient characteristics, including medication types, dispensing patterns, temporal utilization measures, and summary statistics capturing the intensity and variability of opioid exposure. Feature construction is standardized across models to support fair comparison.

Repository Structure
------------------------------------------------------
```
Predicting-Long-Term-Opioid-Use-from-PDMP-Data/
├── R/
│ ├── 01_load_data.R
│ ├── 02_model_run.R
│ ├── 03_results.R
│ └── 04_visualization.R
├── data/
│ └── schema_github.csv
├── .gitignore
├── LICENSE
├── README.md
├── renv.lock
└── run_all.R
 
```
- `R/01_load_data.R`  
  Loads data, performs feature engineering, and prepares training/testing sets.

- `R/02_model_run.R`  
  Trains machine learning models and generates predicted probabilities.

- `R/03_results.R`  
  Evaluates model performance, applies multiple thresholding strategies, and exports summary tables.

- `R/04_visualization.R`  
  Generates diagnostic and evaluation visualizations (e.g., calibration plots and distribution summaries) for model assessment.

- `run_all.R`  
  Runs the full pipeline in the correct order.

- `data/schema_github.csv`  
  Data schema (variable names and types). No individual-level data are included.

Models Included
------------------------------------------------------
The pipeline evaluates the following models:

- Random Forest  
- XGBoost  
- LASSO (glmnet)  
- Logistic Regression (MLR)  
- Logistic Regression with interaction terms (MLR_comp)  
- Neural Network (nnet) with PCA  
- Elastic Net (glmnet)

Evaluation Strategy
------------------------------------------------------
Each model is evaluated under three thresholding strategies:

- **Base**: Fixed cutoff at 0.5  
- **Youden**: Cutoff that maximizes *(Sensitivity + Specificity − 1)*  
- **F1**: Cutoff that maximizes the F1 score across a grid of thresholds  

Reported metrics typically include:
------------------------------------------------------
- Accuracy  
- Sensitivity, Specificity  
- PPV, NPV  
- Youden’s J  
- Confusion matrix counts (TN, FP, FN, TP)  
- AUC (C-statistic) with confidence intervals  

Calibration
------------------------------------------------------
Calibration plots assess agreement between predicted probabilities and observed outcomes. Loess-smoothed curves with confidence intervals are overlaid on histograms of predicted probabilities to contextualize calibration quality and prediction density.

Quickstart
------------------------------------------------------
After cloning the repository, run the full pipeline with:

```r
source("run_all.R")
```

**Note:** Model results and visualizations are intentionally omitted from this repository due to data sensitivity and privacy considerations.

Data Availability
------------------------------------------------------
Due to privacy, ethical, and institutional restrictions, individual-level PDMP data cannot be publicly shared.

This repository provides:

 - Code for data processing, modeling, and evaluation

 - A complete data schema describing all variables used

Access to the underlying data may be granted upon reasonable request and appropriate institutional approvals.

Data Schema
------------------------------------------------------
The full data schema is available here:
 - `data/schema_github.csv`
The schema includes variable names and data types for all features used in model training and evaluation.

---

Reproducibility
------------------------------------------------------
This project uses renv to manage R package dependencies.

To restore the project environment after cloning:

```r
install.packages("renv")
renv::restore()
```

Random seeds are set where applicable; minor numerical differences may still occur across platforms.

Outputs
------------------------------------------------------
`R/03_results.R` generates:

- Model performance summary tables (CSV)

- AUC with confidence intervals

- Confusion matrix–based performance metrics

- Threshold-specific evaluation results

Output filenames and paths are defined within the script.

Privacy and Ethics
------------------------------------------------------
This repository does not contain individual-level patient or PDMP data.

All analyses were conducted in accordance with applicable privacy, ethical, and institutional guidelines.
The MIT License applies to the code only and does not grant access to any underlying data.

Citation
------------------------------------------------------
If you use this repository, please cite:

`Eugene Shin. Predicting Long-Term Opioid Use from PDMP Data. GitHub repository.`

License
------------------------------------------------------
This project is licensed under the MIT License.
See the `LICENSE` file for details.

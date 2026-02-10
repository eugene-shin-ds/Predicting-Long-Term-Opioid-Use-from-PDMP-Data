# ============================================================
# Unified model performance summary (LATEST logic)
# - One final object: results_final
# - Youden_J computed directly from each confusion matrix
# - AUC/CI computed once per ROC, then reused
# - Print and CSV export use the same object
# ============================================================

library(dplyr)
library(purrr)
library(tibble)
library(stringr)

# ----------------------------
# 1) Helper: metrics from a confusionMatrix object
# ----------------------------
extract_metrics <- function(cm) {
  # Extract confusion matrix counts
  TN <- cm$table[1, 1]; FN <- cm$table[1, 2]
  FP <- cm$table[2, 1]; TP <- cm$table[2, 2]
  
  # Compute core metrics
  sensitivity <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  specificity <- ifelse((TN + FP) == 0, 0, TN / (TN + FP))
  
  # Latest/consistent Youden's J (from completed confusion matrix)
  youden_j <- sensitivity + specificity - 1
  
  # Other metrics
  ppv <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  npv <- ifelse((TN + FN) == 0, 0, TN / (TN + FN))
  nnp <- ifelse(ppv == 0, Inf, 1 / ppv)
  acc <- (TP + TN) / sum(cm$table)
  
  tibble(
    Accuracy    = acc,
    Sensitivity = sensitivity,
    Specificity = specificity,
    Youden_J    = youden_j,
    PPV         = ppv,
    NPV         = npv,
    NNP         = nnp
  )
}

# ----------------------------
# 2) Confusion matrices (must already exist)
# ----------------------------
cm_list <- list(
  RF_Base       = conf_base_rf,        RF_Youden       = conf_youden_rf,        RF_F1       = conf_f1_rf,
  XGB_Base      = conf_base_xgb,       XGB_Youden      = conf_youden_xgb,       XGB_F1      = conf_f1_xgb,
  LASSO_Base    = conf_base_lasso,     LASSO_Youden    = conf_youden_lasso,     LASSO_F1    = conf_f1_lasso,
  MLR_Base      = conf_base_mlr,       MLR_Youden      = conf_youden_mlr,       MLR_F1      = conf_f1_mlr,
  MLR_comp_Base = conf_base_mlr_comp,  MLR_comp_Youden = conf_youden_mlr_comp,  MLR_comp_F1 = conf_f1_mlr_comp,
  DNN_Base      = conf_base_dnn,       DNN_Youden      = conf_youden_dnn,       DNN_F1      = conf_f1_dnn,
  Elastic_Base  = conf_base_elastic,   Elastic_Youden  = conf_youden_elastic,   Elastic_F1  = conf_f1_elastic
)

# Keep stable ordering based on names
model_cutoffs <- names(cm_list)

# ----------------------------
# 3) Metrics + counts from confusion matrices
# ----------------------------
metrics_tbl <- imap_dfr(cm_list, \(cm, nm) {
  extract_metrics(cm) %>% mutate(Model_Cutoff = nm, .before = 1)
})

counts_tbl <- imap_dfr(cm_list, \(cm, nm) {
  tibble(
    Model_Cutoff = nm,
    TN = cm$table[1, 1],
    FN = cm$table[1, 2],
    FP = cm$table[2, 1],
    TP = cm$table[2, 2]
  )
})

# ----------------------------
# 4) AUC + CI (compute once per ROC and reuse)
#     - assumes extract_auc_ci(roc_obj) returns list(auc=..., ci=...)
# ----------------------------
auc_map <- list(
  RF       = extract_auc_ci(roc_rf),
  XGB      = extract_auc_ci(roc_xgb),
  LASSO    = extract_auc_ci(roc_lasso),
  MLR      = extract_auc_ci(roc_mlr),
  MLR_comp = extract_auc_ci(roc_mlr_comp),
  DNN      = extract_auc_ci(roc_dnn),
  Elastic  = extract_auc_ci(roc_elastic)
)

auc_ci_tbl <- tibble(
  Model_Cutoff = model_cutoffs,
  Model = str_replace(Model_Cutoff, "_(Base|Youden|F1)$", ""),
  C_stat = map_dbl(Model, \(m) auc_map[[m]]$auc),
  CI     = map_chr(Model, \(m) auc_map[[m]]$ci)
) %>%
  select(-Model)

# ----------------------------
# 5) Cutoff table (latest variables you already computed)
# ----------------------------
cutoff_tbl <- tibble(
  Model_Cutoff = model_cutoffs,
  Cutoff = round(c(
    0.5, cutoff_youden_rf,       cutoff_f1_rf,
    0.5, cutoff_youden_xgb,      cutoff_f1_xgb,
    0.5, cutoff_youden_lasso,    cutoff_f1_lasso,
    0.5, cutoff_youden_mlr,      cutoff_f1_mlr,
    0.5, cutoff_youden_mlr_comp, cutoff_f1_mlr_comp,
    0.5, cutoff_youden_dnn,      cutoff_f1_dnn,
    0.5, cutoff_youden_elastic,  cutoff_f1_elastic
  ), 3)
)

# ----------------------------
# 6) Final merge (ONE final object only)
# ----------------------------
results_final <- metrics_tbl %>%
  left_join(auc_ci_tbl,  by = "Model_Cutoff") %>%
  left_join(counts_tbl,  by = "Model_Cutoff") %>%
  left_join(cutoff_tbl,  by = "Model_Cutoff") %>%
  select(Model_Cutoff, Cutoff, Youden_J, everything())

print(results_final)

# ----------------------------
# 7) Export (same object that you printed)
# ----------------------------
write.csv(
  results_final,
  file = "model_performance_summary_corrected_251215_2.csv",
  row.names = FALSE
)

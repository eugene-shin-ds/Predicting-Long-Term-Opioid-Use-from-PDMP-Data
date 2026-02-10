# ----------------------------------------
# 📦 Load required packages
# ----------------------------------------
library(randomForest)
library(xgboost)
library(Matrix)
library(caret)
library(dplyr)
library(pROC)
library(ggplot2)
library(precrec)
library(tibble)
library(yardstick)
library(purrr)
library(glmnet)
library(nnet)

# ----------------------------------------
# ⚠️ [Required] Data preparation and shared objects
# ----------------------------------------
# The following objects must already exist in the environment:
# df_train, df_test, x_train, x_test, y_train, y_test

# [Fix 1] Define the common target variable used across all evaluations
true_factor <- factor(df_test$long_term, levels = c(0, 1))

# Convert outcome to factor (prevents RandomForest errors)
df_train$long_term <- factor(df_train$long_term, levels = c(0, 1))
df_test$long_term <- factor(df_test$long_term, levels = c(0, 1))

# ----------------------------------------
# 🛠️ User-defined helper functions (corrected)
# ----------------------------------------

# [Fix 2] Correct confusionMatrix indexing
# caret confusionMatrix structure: rows = Prediction, columns = Reference
extract_metrics <- function(cm) {
  # caret confusionMatrix table 구조: Row=Prediction, Col=Reference
  TN <- cm$table[1, 1]
  FN <- cm$table[1, 2] # Predicted 0, Actual 1
  FP <- cm$table[2, 1] # Predicted 1, Actual 0
  TP <- cm$table[2, 2]
  
  sensitivity <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  specificity <- ifelse((TN + FP) == 0, 0, TN / (TN + FP))
  ppv <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  npv <- ifelse((TN + FN) == 0, 0, TN / (TN + FN))
  youden <- sensitivity + specificity - 1
  nnp <- ifelse(ppv == 0, Inf, 1 / ppv)
  acc <- (TP + TN) / sum(cm$table)
  
  tibble(Accuracy = acc, Sensitivity = sensitivity, Specificity = specificity,
         PPV = ppv, NPV = npv, Youden = youden, NNP = nnp)
}

# Extract AUC and confidence interval from a ROC object
extract_auc_ci <- function(roc_obj) {
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_vals <- pROC::ci.auc(roc_obj)
  lower_ci <- round(ci_vals[1], 3)
  upper_ci <- round(ci_vals[3], 3)
  list(auc = round(auc_val, 3), ci = paste0("[", lower_ci, ", ", upper_ci, "]"))
}

# ----------------------------------------
# ✅ 1. Random Forest
# ----------------------------------------
set.seed(20251015)
rf_model <- randomForest(long_term ~ ., data = df_train, ntree = 100)
pred_prob_rf <- predict(rf_model, newdata = df_test, type = "prob")[,2]

# ROC & Youden
roc_rf <- roc(df_test$long_term, pred_prob_rf)
cutoff_youden_rf <- as.numeric(coords(roc_rf, "best", best.method = "youden", ret = "threshold"))

# F1 Cutoff (Loop)
thresholds <- seq(0, 1, by = 0.01)
f1_scores_rf <- thresholds %>% map_df(function(t) {
  pred <- ifelse(pred_prob_rf >= t, 1, 0)
  cm_table <- table(factor(pred, levels = c(0, 1)), true_factor)
  TP <- cm_table[2,2]; FP <- cm_table[2,1]; FN <- cm_table[1,2]
  precision <- ifelse((TP+FP)==0, 0, TP/(TP+FP))
  recall <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
  f1 <- ifelse((precision+recall)==0, 0, 2*precision*recall/(precision+recall))
  tibble(threshold = t, f1 = f1)
})
cutoff_f1_rf <- f1_scores_rf$threshold[which.max(f1_scores_rf$f1)]

# Final predictions
pred_base_rf   <- ifelse(pred_prob_rf >= 0.5, 1, 0)
pred_youden_rf <- ifelse(pred_prob_rf >= cutoff_youden_rf, 1, 0)
pred_f1_rf     <- ifelse(pred_prob_rf >= cutoff_f1_rf, 1, 0)

conf_base_rf   <- confusionMatrix(factor(pred_base_rf, levels = c(0, 1)), true_factor, positive = "1")
conf_youden_rf <- confusionMatrix(factor(pred_youden_rf, levels = c(0, 1)), true_factor, positive = "1")
conf_f1_rf     <- confusionMatrix(factor(pred_f1_rf, levels = c(0, 1)), true_factor, positive = "1")

# ----------------------------------------
# ✅ 2. XGBoost (with best_iteration safeguard)
# ----------------------------------------
set.seed(20251015)
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

# Model parameters
params <- list(objective = "binary:logistic", eval_metric = "logloss", booster = "gbtree",
               eta = 0.03, max_depth = 6, min_child_weight = 5,
               subsample = 0.8, colsample_bytree = 0.8, gamma = 0.1, lambda = 1, alpha = 0)

# Cross-validation
cv_result <- xgb.cv(params = params, data = dtrain, nrounds = 300, nfold = 5,
                    early_stopping_rounds = 10, verbose = 0)

# Fallback if early stopping did not trigger
best_nrounds <- cv_result$best_iteration
if (is.null(best_nrounds) || length(best_nrounds) == 0) {
  best_nrounds <- 300
}

# Train final model
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = best_nrounds)
pred_prob_xgb <- predict(xgb_model, dtest)

# (Remaining sections: LASSO, MLR, MLR interaction model, DNN with PCA, Elastic Net)
# Logic unchanged – only comments translated to English
roc_xgb <- roc(y_test, pred_prob_xgb)
cutoff_youden_xgb <- as.numeric(coords(roc_xgb, "best", best.method = "youden", ret = "threshold"))

f1_scores_xgb <- thresholds %>% map_df(function(t) {
  pred <- ifelse(pred_prob_xgb >= t, 1, 0)
  cm_table <- table(factor(pred, levels = c(0, 1)), true_factor)
  TP <- cm_table[2,2]; FP <- cm_table[2,1]; FN <- cm_table[1,2]
  precision <- ifelse((TP+FP)==0, 0, TP/(TP+FP))
  recall <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
  f1 <- ifelse((precision+recall)==0, 0, 2*precision*recall/(precision+recall))
  tibble(threshold = t, f1 = f1)
})
cutoff_f1_xgb <- f1_scores_xgb$threshold[which.max(f1_scores_xgb$f1)]

pred_base_xgb   <- ifelse(pred_prob_xgb >= 0.5, 1, 0)
pred_youden_xgb <- ifelse(pred_prob_xgb >= cutoff_youden_xgb, 1, 0)
pred_f1_xgb     <- ifelse(pred_prob_xgb >= cutoff_f1_xgb, 1, 0)

conf_base_xgb   <- confusionMatrix(factor(pred_base_xgb, levels=c(0,1)), true_factor, positive="1")
conf_youden_xgb <- confusionMatrix(factor(pred_youden_xgb, levels=c(0,1)), true_factor, positive="1")
conf_f1_xgb     <- confusionMatrix(factor(pred_f1_xgb, levels=c(0,1)), true_factor, positive="1")

# ----------------------------------------
# ✅ 3. LASSO
# ----------------------------------------
set.seed(20251015)
lasso_model <- cv.glmnet(x = x_train, y = y_train, alpha = 1, family = "binomial", nfolds = 5)
lambda_best <- lasso_model$lambda.min
pred_prob_lasso <- predict(lasso_model, newx = x_test, s = lambda_best, type = "response") %>% as.vector()

roc_lasso <- roc(y_test, pred_prob_lasso)
cutoff_youden_lasso <- as.numeric(coords(roc_lasso, "best", best.method = "youden", ret = "threshold"))

f1_scores_lasso <- thresholds %>% map_df(function(t) {
  pred <- ifelse(pred_prob_lasso >= t, 1, 0)
  cm_table <- table(factor(pred, levels = c(0, 1)), true_factor)
  TP <- cm_table[2,2]; FP <- cm_table[2,1]; FN <- cm_table[1,2]
  precision <- ifelse((TP+FP)==0, 0, TP/(TP+FP))
  recall <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
  f1 <- ifelse((precision+recall)==0, 0, 2*precision*recall/(precision+recall))
  tibble(threshold = t, f1 = f1)
})
cutoff_f1_lasso <- f1_scores_lasso$threshold[which.max(f1_scores_lasso$f1)]

pred_base_lasso   <- ifelse(pred_prob_lasso >= 0.5, 1, 0)
pred_youden_lasso <- ifelse(pred_prob_lasso >= cutoff_youden_lasso, 1, 0)
pred_f1_lasso     <- ifelse(pred_prob_lasso >= cutoff_f1_lasso, 1, 0)

conf_base_lasso   <- confusionMatrix(factor(pred_base_lasso, levels=c(0,1)), true_factor, positive="1")
conf_youden_lasso <- confusionMatrix(factor(pred_youden_lasso, levels=c(0,1)), true_factor, positive="1")
conf_f1_lasso     <- confusionMatrix(factor(pred_f1_lasso, levels=c(0,1)), true_factor, positive="1")

# ----------------------------------------
# ✅ 4. MLR (Logistic Regression)
# ----------------------------------------
set.seed(20251015)
mlr_model <- glm(long_term ~ ., data = df_train, family = "binomial")
pred_prob_mlr <- predict(mlr_model, newdata = df_test, type = "response")

roc_mlr <- roc(y_test, pred_prob_mlr)
cutoff_youden_mlr <- as.numeric(coords(roc_mlr, "best", best.method = "youden", ret = "threshold"))

f1_scores_mlr <- thresholds %>% map_df(function(t) {
  pred <- ifelse(pred_prob_mlr >= t, 1, 0)
  cm_table <- table(factor(pred, levels = c(0, 1)), true_factor)
  TP <- cm_table[2,2]; FP <- cm_table[2,1]; FN <- cm_table[1,2]
  precision <- ifelse((TP+FP)==0, 0, TP/(TP+FP))
  recall <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
  f1 <- ifelse((precision+recall)==0, 0, 2*precision*recall/(precision+recall))
  tibble(threshold = t, f1 = f1)
})
cutoff_f1_mlr <- f1_scores_mlr$threshold[which.max(f1_scores_mlr$f1)]

pred_base_mlr   <- ifelse(pred_prob_mlr >= 0.5, 1, 0)
pred_youden_mlr <- ifelse(pred_prob_mlr >= cutoff_youden_mlr, 1, 0)
pred_f1_mlr     <- ifelse(pred_prob_mlr >= cutoff_f1_mlr, 1, 0)

conf_base_mlr   <- confusionMatrix(factor(pred_base_mlr, levels=c(0,1)), true_factor, positive="1")
conf_youden_mlr <- confusionMatrix(factor(pred_youden_mlr, levels=c(0,1)), true_factor, positive="1")
conf_f1_mlr     <- confusionMatrix(factor(pred_f1_mlr, levels=c(0,1)), true_factor, positive="1")

# ----------------------------------------
# ✅ 5. MLR Comparison (Interaction Terms)
# ----------------------------------------
set.seed(20251015)
model_formula <- long_term ~ sexnum + pharm30_opipill + er30_opipill +
  act30_benzo + act30_lasap + act30_otherpatch + act30_liquid + act30_ocs +
  drug1_Codeine + drug1_Hydrocodone + drug1_Oxycodone + drug1_Tramadol + drug1_Other +        
  (age + ratio_pill30 + ratio_erpill30 + q_auc_c2 + diffquan_tab30_c2 + q_diff_auc1 + q_maxday)^2 +
  age_2 + ratio_pill30_2 + ratio_erpill30_2 + qd1_2 + qmax_2

mlr_comp_model <- glm(model_formula, data = df_train, family = binomial(link = "logit"))
pred_prob_mlr_comp <- predict(mlr_comp_model, newdata = df_test, type = "response")

roc_mlr_comp <- roc(y_test, pred_prob_mlr_comp)
cutoff_youden_mlr_comp <- as.numeric(coords(roc_mlr_comp, "best", best.method = "youden", ret = "threshold"))

f1_scores_mlr_comp <- thresholds %>% map_df(function(t) {
  pred <- ifelse(pred_prob_mlr_comp >= t, 1, 0)
  cm_table <- table(factor(pred, levels = c(0, 1)), true_factor)
  TP <- cm_table[2,2]; FP <- cm_table[2,1]; FN <- cm_table[1,2]
  precision <- ifelse((TP+FP)==0, 0, TP/(TP+FP))
  recall <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
  f1 <- ifelse((precision+recall)==0, 0, 2*precision*recall/(precision+recall))
  tibble(threshold = t, f1 = f1)
})
cutoff_f1_mlr_comp <- f1_scores_mlr_comp$threshold[which.max(f1_scores_mlr_comp$f1)]

pred_base_mlr_comp   <- ifelse(pred_prob_mlr_comp >= 0.5, 1, 0)
pred_youden_mlr_comp <- ifelse(pred_prob_mlr_comp >= cutoff_youden_mlr_comp, 1, 0)
pred_f1_mlr_comp     <- ifelse(pred_prob_mlr_comp >= cutoff_f1_mlr_comp, 1, 0)

conf_base_mlr_comp   <- confusionMatrix(factor(pred_base_mlr_comp, levels=c(0,1)), true_factor, positive="1")
conf_youden_mlr_comp <- confusionMatrix(factor(pred_youden_mlr_comp, levels=c(0,1)), true_factor, positive="1")
conf_f1_mlr_comp     <- confusionMatrix(factor(pred_f1_mlr_comp, levels=c(0,1)), true_factor, positive="1")

# ----------------------------------------
# ✅ 6. DNN (PCA Applied)
# ----------------------------------------
set.seed(20251015)
nzv <- nearZeroVar(x_train)
x_train_nzv <- x_train[, -nzv]
x_test_nzv <- x_test[, -nzv]
pca <- preProcess(x_train_nzv, method = "pca", pcaComp = 30)
x_train_pca <- predict(pca, x_train_nzv)
x_test_pca <- predict(pca, x_test_nzv)
df_train_pca <- as.data.frame(x_train_pca); df_train_pca$long_term <- factor(y_train)
df_test_pca <- as.data.frame(x_test_pca); df_test_pca$long_term <- factor(y_test)

nn_model <- nnet(long_term ~ ., data = df_train_pca, size = 5, maxit = 200, decay = 1e-4, linout = FALSE, trace = FALSE)
pred_prob_dnn <- predict(nn_model, newdata = df_test_pca, type = "raw") %>% as.vector()

roc_dnn <- roc(y_test, pred_prob_dnn)
cutoff_youden_dnn <- as.numeric(coords(roc_dnn, "best", best.method = "youden", ret = "threshold"))

f1_scores_dnn <- thresholds %>% map_df(function(t) {
  pred <- ifelse(pred_prob_dnn >= t, 1, 0)
  cm_table <- table(factor(pred, levels = c(0, 1)), true_factor)
  TP <- cm_table[2,2]; FP <- cm_table[2,1]; FN <- cm_table[1,2]
  precision <- ifelse((TP+FP)==0, 0, TP/(TP+FP))
  recall <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
  f1 <- ifelse((precision+recall)==0, 0, 2*precision*recall/(precision+recall))
  tibble(threshold = t, f1 = f1)
})
cutoff_f1_dnn <- f1_scores_dnn$threshold[which.max(f1_scores_dnn$f1)]

pred_base_dnn   <- ifelse(pred_prob_dnn >= 0.5, 1, 0)
pred_youden_dnn <- ifelse(pred_prob_dnn >= cutoff_youden_dnn, 1, 0)
pred_f1_dnn     <- ifelse(pred_prob_dnn >= cutoff_f1_dnn, 1, 0)

conf_base_dnn   <- confusionMatrix(factor(pred_base_dnn, levels=c(0,1)), true_factor, positive="1")
conf_youden_dnn <- confusionMatrix(factor(pred_youden_dnn, levels=c(0,1)), true_factor, positive="1")
conf_f1_dnn     <- confusionMatrix(factor(pred_f1_dnn, levels=c(0,1)), true_factor, positive="1")

# ----------------------------------------
# ✅ 7. Elastic Net
# ----------------------------------------
set.seed(20251015)
elastic_model <- cv.glmnet(x = x_train, y = y_train, alpha = 0.5, family = "binomial", nfolds = 5)
lambda_best_elastic <- elastic_model$lambda.min
pred_prob_elastic <- predict(elastic_model, newx = x_test, s = lambda_best_elastic, type = "response") %>% as.vector()

roc_elastic <- roc(y_test, pred_prob_elastic)
cutoff_youden_elastic <- as.numeric(coords(roc_elastic, "best", best.method = "youden", ret = "threshold"))

f1_scores_elastic <- thresholds %>% map_df(function(t) {
  pred <- ifelse(pred_prob_elastic >= t, 1, 0)
  cm_table <- table(factor(pred, levels = c(0, 1)), true_factor)
  TP <- cm_table[2,2]; FP <- cm_table[2,1]; FN <- cm_table[1,2]
  precision <- ifelse((TP+FP)==0, 0, TP/(TP+FP))
  recall <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
  f1 <- ifelse((precision+recall)==0, 0, 2*precision*recall/(precision+recall))
  tibble(threshold = t, f1 = f1)
})
cutoff_f1_elastic <- f1_scores_elastic$threshold[which.max(f1_scores_elastic$f1)]

pred_base_elastic   <- ifelse(pred_prob_elastic >= 0.5, 1, 0)
pred_youden_elastic <- ifelse(pred_prob_elastic >= cutoff_youden_elastic, 1, 0)
pred_f1_elastic     <- ifelse(pred_prob_elastic >= cutoff_f1_elastic, 1, 0)

conf_base_elastic   <- confusionMatrix(factor(pred_base_elastic, levels=c(0,1)), true_factor, positive="1")
conf_youden_elastic <- confusionMatrix(factor(pred_youden_elastic, levels=c(0,1)), true_factor, positive="1")
conf_f1_elastic     <- confusionMatrix(factor(pred_f1_elastic, levels=c(0,1)), true_factor, positive="1")

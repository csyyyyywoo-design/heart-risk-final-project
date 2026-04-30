# Heart Risk Final Project
# Logistic Regression + Random Forest + XGBoost
# Hannah Chen

# -----------------------------
# 1. Load packages
# -----------------------------
library(tidyverse)
library(dplyr)
library(MASS)
library(caret)
library(pROC)
library(randomForest)
library(xgboost)

# -----------------------------
# 2. Load and clean data
# -----------------------------
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")
heart_nomis <- na.omit(heart)

# -----------------------------
# 3. Convert categorical variables consistently
# -----------------------------
heart_nomis$Alcohol.Consumption <- factor(
  heart_nomis$Alcohol.Consumption,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Diet <- factor(
  heart_nomis$Diet,
  levels = c(0, 1, 2),
  labels = c("0", "1", "2")
)

heart_nomis$Previous.Heart.Problems <- factor(
  heart_nomis$Previous.Heart.Problems,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Medication.Use <- factor(
  heart_nomis$Medication.Use,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Diabetes <- factor(heart_nomis$Diabetes, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Family.History <- factor(heart_nomis$Family.History, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Smoking <- factor(heart_nomis$Smoking, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Obesity <- factor(heart_nomis$Obesity, levels = c(0, 1), labels = c("0", "1"))

heart_nomis$Heart.Attack.Risk..Binary. <- factor(
  heart_nomis$Heart.Attack.Risk..Binary.,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Gender <- factor(heart_nomis$Gender)

# -----------------------------
# 4. Train/test split: 80/20
# -----------------------------
set.seed(123)

train_index <- createDataPartition(
  heart_nomis$Heart.Attack.Risk..Binary.,
  p = 0.8,
  list = FALSE
)

train_data <- heart_nomis[train_index, ]
test_data  <- heart_nomis[-train_index, ]

# -----------------------------
# 5. Full formula
# -----------------------------
full_formula <- Heart.Attack.Risk..Binary. ~
  Age + Cholesterol + Heart.rate +
  Diabetes + Family.History + Smoking + Obesity +
  Alcohol.Consumption + Exercise.Hours.Per.Week + Diet +
  Previous.Heart.Problems + Medication.Use + Stress.Level +
  Sedentary.Hours.Per.Day + Income + BMI + Triglycerides +
  Physical.Activity.Days.Per.Week + Sleep.Hours.Per.Day +
  Blood.sugar + CK.MB + Troponin + Gender +
  Systolic.blood.pressure + Diastolic.blood.pressure

# ============================================================
# 6. Logistic Regression Model
# ============================================================

full_log_model <- glm(
  full_formula,
  data = train_data,
  family = binomial
)

log_model <- stepAIC(
  full_log_model,
  direction = "both",
  trace = FALSE
)

summary(log_model)

test_data$log_prob <- predict(
  log_model,
  newdata = test_data,
  type = "response"
)

test_data$log_class <- factor(
  ifelse(test_data$log_prob > 0.5, "1", "0"),
  levels = c("0", "1")
)

log_cm <- confusionMatrix(
  test_data$log_class,
  test_data$Heart.Attack.Risk..Binary.,
  positive = "1"
)

log_roc <- roc(
  response = test_data$Heart.Attack.Risk..Binary.,
  predictor = test_data$log_prob,
  levels = c("0", "1")
)

log_auc <- auc(log_roc)

# ============================================================
# 7. Random Forest Model
# ============================================================

set.seed(123)

rf_model <- randomForest(
  full_formula,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

print(rf_model)

rf_pred <- predict(
  rf_model,
  newdata = test_data
)

rf_prob <- predict(
  rf_model,
  newdata = test_data,
  type = "prob"
)[, "1"]

rf_cm <- confusionMatrix(
  rf_pred,
  test_data$Heart.Attack.Risk..Binary.,
  positive = "1"
)

rf_roc <- roc(
  response = test_data$Heart.Attack.Risk..Binary.,
  predictor = rf_prob,
  levels = c("0", "1")
)

rf_auc <- auc(rf_roc)

# ============================================================
# 8. XGBoost Model
# ============================================================

train_label <- as.numeric(as.character(train_data$Heart.Attack.Risk..Binary.))
test_label  <- as.numeric(as.character(test_data$Heart.Attack.Risk..Binary.))

x_train <- model.matrix(full_formula, data = train_data)[, -1]
x_test  <- model.matrix(full_formula, data = test_data)[, -1]

dtrain <- xgb.DMatrix(data = x_train, label = train_label)
dtest  <- xgb.DMatrix(data = x_test, label = test_label)

set.seed(123)

xgb_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 3,
    eta = 0.1
  ),
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

xgb_prob <- predict(
  xgb_model,
  newdata = dtest
)

xgb_pred <- factor(
  ifelse(xgb_prob > 0.5, "1", "0"),
  levels = c("0", "1")
)

xgb_cm <- confusionMatrix(
  xgb_pred,
  test_data$Heart.Attack.Risk..Binary.,
  positive = "1"
)

xgb_roc <- roc(
  response = test_data$Heart.Attack.Risk..Binary.,
  predictor = xgb_prob,
  levels = c("0", "1")
)

xgb_auc <- auc(xgb_roc)

# ============================================================
# 9. Compare model performance
# ============================================================

model_results <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  AUC = c(
    round(as.numeric(log_auc), 4),
    round(as.numeric(rf_auc), 4),
    round(as.numeric(xgb_auc), 4)
  ),
  Accuracy = c(
    round(unname(log_cm$overall["Accuracy"]), 4),
    round(unname(rf_cm$overall["Accuracy"]), 4),
    round(unname(xgb_cm$overall["Accuracy"]), 4)
  ),
  Sensitivity = c(
    round(unname(log_cm$byClass["Sensitivity"]), 4),
    round(unname(rf_cm$byClass["Sensitivity"]), 4),
    round(unname(xgb_cm$byClass["Sensitivity"]), 4)
  ),
  Specificity = c(
    round(unname(log_cm$byClass["Specificity"]), 4),
    round(unname(rf_cm$byClass["Specificity"]), 4),
    round(unname(xgb_cm$byClass["Specificity"]), 4)
  )
)

print(model_results)

# ============================================================
# 10. Combined ROC curve
# ============================================================

plot(
  log_roc,
  col = "blue",
  lwd = 2,
  main = "ROC Curves: Logistic Regression vs Random Forest vs XGBoost"
)

plot(
  rf_roc,
  col = "green",
  lwd = 2,
  add = TRUE
)

plot(
  xgb_roc,
  col = "purple",
  lwd = 2,
  add = TRUE
)

abline(a = 0, b = 1, lty = 2, col = "gray")

legend(
  "bottomright",
  legend = c(
    paste("Logistic AUC =", round(as.numeric(log_auc), 3)),
    paste("Random Forest AUC =", round(as.numeric(rf_auc), 3)),
    paste("XGBoost AUC =", round(as.numeric(xgb_auc), 3))
  ),
  col = c("blue", "green", "purple"),
  lwd = 2
)

# ============================================================
# 11. Feature importance
# ============================================================

# Random Forest importance
rf_importance <- as.data.frame(importance(rf_model))
rf_importance$Feature <- rownames(rf_importance)

rf_importance <- rf_importance %>%
  arrange(desc(MeanDecreaseGini))

print(head(rf_importance, 10))

# XGBoost importance
xgb_importance <- xgb.importance(
  feature_names = colnames(x_train),
  model = xgb_model
)

print(head(xgb_importance, 10))
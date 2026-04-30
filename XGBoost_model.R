# Heart Risk Final Project
# XGBoost Model
# Hannah Chen

# -----------------------------
# Load packages
# -----------------------------
library(tidyverse)
library(dplyr)
library(caret)
library(pROC)
library(xgboost)

# -----------------------------
# Load and clean data
# -----------------------------
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")
heart_nomis <- na.omit(heart)

# -----------------------------
# Load data and clean data
# -----------------------------
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")

# Remove missing values
heart_nomis <- na.omit(heart)

# Check cleaned data
colSums(is.na(heart_nomis))
dim(heart_nomis)

# -----------------------------
# Convert variables to factors
# -----------------------------
heart_nomis$Alcohol.Consumption <- factor(
  heart_nomis$Alcohol.Consumption,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Diet <- as.factor(heart_nomis$Diet)

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

heart_nomis$Diabetes <- factor(
  heart_nomis$Diabetes,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Family.History <- factor(
  heart_nomis$Family.History,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Smoking <- factor(
  heart_nomis$Smoking,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Obesity <- factor(
  heart_nomis$Obesity,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Heart.Attack.Risk..Binary. <- factor(
  heart_nomis$Heart.Attack.Risk..Binary.,
  levels = c(0, 1),
  labels = c("0", "1")
)

heart_nomis$Gender <- factor(heart_nomis$Gender)

# -----------------------------
# Train/test split
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
# Full formula
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


# -----------------------------
# 6. Prepare data for XGBoost
# -----------------------------

# XGBoost needs numeric 0/1 labels
train_label <- as.numeric(as.character(train_data$Heart.Attack.Risk..Binary.))
test_label  <- as.numeric(as.character(test_data$Heart.Attack.Risk..Binary.))

# Convert predictors into numeric matrix
x_train <- model.matrix(full_formula, data = train_data)[, -1]
x_test  <- model.matrix(full_formula, data = test_data)[, -1]

# Convert to XGBoost DMatrix format
dtrain <- xgb.DMatrix(data = x_train, label = train_label)
dtest  <- xgb.DMatrix(data = x_test, label = test_label)

# -----------------------------
# Fit XGBoost model
# -----------------------------
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

print(xgb_model)

# -----------------------------
# Predictions
# -----------------------------
xgb_prob <- predict(xgb_model, newdata = dtest)

xgb_pred <- factor(
  ifelse(xgb_prob > 0.5, "1", "0"),
  levels = c("0", "1")
)

# -----------------------------
# Model evaluation
# -----------------------------
xgb_cm <- confusionMatrix(
  xgb_pred,
  test_data$Heart.Attack.Risk..Binary.,
  positive = "1"
)

print(xgb_cm)

xgb_roc <- roc(
  response = test_data$Heart.Attack.Risk..Binary.,
  predictor = xgb_prob,
  levels = c("0", "1")
)

xgb_auc <- auc(xgb_roc)

print(xgb_auc)

# -----------------------------
# ROC curve
# -----------------------------
plot(
  xgb_roc,
  main = "ROC Curve for XGBoost Model",
  col = "purple",
  lwd = 2
)

abline(a = 0, b = 1, lty = 2, col = "gray")

# -----------------------------
# 11. Feature importance
# -----------------------------
xgb_importance <- xgb.importance(
  feature_names = colnames(x_train),
  model = xgb_model
)

print(xgb_importance)

xgb.plot.importance(
  xgb_importance,
  top_n = 10,
  main = "Top 10 XGBoost Feature Importance"
)
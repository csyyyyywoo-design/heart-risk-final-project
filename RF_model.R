# Heart Risk Final Project
# Random Forest Regression Model
# Hannah Chen

# -----------------------------
# Load packages
# -----------------------------
library(tidyverse)
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(randomForest)

# -----------------------------
# Load data
# -----------------------------
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")

# -----------------------------
# Remove missing values
# -----------------------------
heart_nomis <- na.omit(heart)

# Check cleaned data
colSums(is.na(heart_nomis))
dim(heart_nomis)

# -----------------------------
# Convert variables to factors
# -----------------------------
heart_nomis$Diabetes <- as.factor(heart_nomis$Diabetes)
heart_nomis$Family.History <- as.factor(heart_nomis$Family.History)
heart_nomis$Smoking <- as.factor(heart_nomis$Smoking)
heart_nomis$Obesity <- as.factor(heart_nomis$Obesity)
heart_nomis$Alcohol.Consumption <- as.factor(heart_nomis$Alcohol.Consumption)
heart_nomis$Diet <- as.factor(heart_nomis$Diet)
heart_nomis$Previous.Heart.Problems <- as.factor(heart_nomis$Previous.Heart.Problems)
heart_nomis$Medication.Use <- as.factor(heart_nomis$Medication.Use)
heart_nomis$Gender <- as.factor(heart_nomis$Gender)

heart_nomis$Heart.Attack.Risk..Binary. <- factor(
  heart_nomis$Heart.Attack.Risk..Binary.,
  levels = c(0, 1),
  labels = c("0", "1")
)

# Final structure check
str(heart_nomis)

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

# Check proportions
dim(train_data)
dim(test_data)
prop.table(table(train_data$Heart.Attack.Risk..Binary.))
prop.table(table(test_data$Heart.Attack.Risk..Binary.))

# -----------------------------
# Random forest model
# -----------------------------
set.seed(123)

rf_model <- randomForest(
  Heart.Attack.Risk..Binary. ~ Age + Cholesterol + Heart.rate +
    Diabetes + Family.History + Smoking + Obesity +
    Alcohol.Consumption + Exercise.Hours.Per.Week + Diet +
    Previous.Heart.Problems + Medication.Use + Stress.Level +
    Sedentary.Hours.Per.Day + Income + BMI + Triglycerides +
    Physical.Activity.Days.Per.Week + Sleep.Hours.Per.Day +
    Blood.sugar + CK.MB + Troponin + Gender +
    Systolic.blood.pressure + Diastolic.blood.pressure,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

print(rf_model)

# -----------------------------
# Predictions
# -----------------------------
rf_pred <- predict(rf_model, newdata = test_data)

rf_prob_1 <- predict(
  rf_model,
  newdata = test_data,
  type = "prob"
)[, "1"]

# -----------------------------
# Model evaluation
# -----------------------------
rf_cm <- confusionMatrix(
  rf_pred,
  test_data$Heart.Attack.Risk..Binary.
)
print(rf_cm)

rf_roc <- roc(test_data$Heart.Attack.Risk..Binary., rf_prob_1)
rf_auc <- auc(rf_roc)

print(rf_auc)

# -----------------------------
# ROC curve
# -----------------------------
plot(
  rf_roc,
  main = "ROC Curve for Random Forest",
  col = "blue",
  lwd = 2
)
abline(a = 0, b = 1, lty = 2, col = "gray")

# -----------------------------
# Variable importance
# -----------------------------
varImpPlot(rf_model, main = "Random Forest Variable Importance")
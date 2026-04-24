# Heart Risk Final Project
# Logistic Regression Model
# Hannah Chen

# Load packages
library(tidyverse)
library(ggplot2)
library(dplyr)
library(MASS)
library(caret)
library(pROC)

# Load data
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")

# Remove incomplete rows
heart_nomis <- na.omit(heart)

# Convert variables to factors
heart_nomis$Diabetes <- as.factor(heart_nomis$Diabetes)
heart_nomis$Family.History <- as.factor(heart_nomis$Family.History)
heart_nomis$Smoking <- as.factor(heart_nomis$Smoking)
heart_nomis$Obesity <- as.factor(heart_nomis$Obesity)
heart_nomis$Alcohol.Consumption <- as.factor(heart_nomis$Alcohol.Consumption)
heart_nomis$Diet <- as.factor(heart_nomis$Diet)
heart_nomis$Previous.Heart.Problems <- as.factor(heart_nomis$Previous.Heart.Problems)
heart_nomis$Medication.Use <- as.factor(heart_nomis$Medication.Use)
heart_nomis$Heart.Attack.Risk..Binary. <- as.factor(heart_nomis$Heart.Attack.Risk..Binary.)
heart_nomis$Gender <- as.factor(heart_nomis$Gender)

heart_nomis$Heart.Attack.Risk..Binary. <- factor(
  heart_nomis$Heart.Attack.Risk..Binary.,
  levels = c(0, 1),
  labels = c("0", "1")
)

# Split data into training and testing
set.seed(123)

train_index <- createDataPartition(
  heart_nomis$Heart.Attack.Risk..Binary.,
  p = 0.8,
  list = FALSE
)

train_data <- heart_nomis[train_index, ]
test_data  <- heart_nomis[-train_index, ]

# Fit logistic regression model
full_model <- glm(
  Heart.Attack.Risk..Binary. ~ Age + Cholesterol + Heart.rate +
    Diabetes + Family.History + Smoking + Obesity +
    Alcohol.Consumption + Exercise.Hours.Per.Week + Diet +
    Previous.Heart.Problems + Medication.Use + Stress.Level +
    Sedentary.Hours.Per.Day + Income + BMI + Triglycerides +
    Physical.Activity.Days.Per.Week + Sleep.Hours.Per.Day +
    Blood.sugar + CK.MB + Troponin + Gender +
    Systolic.blood.pressure + Diastolic.blood.pressure,
  data = train_data,
  family = binomial
)

summary(full_model)

# Odds ratios
odds_ratios <- exp(coef(full_model))
print(odds_ratios)

# Stepwise AIC model
step_model <- stepAIC(full_model, direction = "both", trace = FALSE)
summary(step_model)
AIC(full_model, step_model)

# Predicted probabilities on test set
test_data$pred_prob <- predict(step_model, newdata = test_data, type = "response")

# Predicted classes
test_data$pred_class <- factor(
  ifelse(test_data$pred_prob > 0.5, "1", "0"),
  levels = c("0", "1")
)

# Confusion matrix
logit_cm <- confusionMatrix(
  test_data$pred_class,
  test_data$Heart.Attack.Risk..Binary.
)
print(logit_cm)

# ROC and AUC
logit_roc <- roc(test_data$Heart.Attack.Risk..Binary., test_data$pred_prob)
logit_auc <- auc(logit_roc)

print(logit_auc)

# Small 3-variable logistic model
log1 <- glm(
  Heart.Attack.Risk..Binary. ~ Obesity + Cholesterol + Diabetes,
  data = train_data,
  family = binomial
)

log1_prob <- predict(log1, newdata = test_data, type = "response")
log1_auc <- auc(roc(test_data$Heart.Attack.Risk..Binary., log1_prob))

print(log1_auc)

# Logistic regression plotting
# ROC curve
plot(logit_roc, main = "ROC Curve for Step Logistic Regression Model")
abline(a = 0, b = 1, lty = 2)

# Predicted probability histogram
ggplot(test_data, aes(x = pred_prob, fill = Heart.Attack.Risk..Binary.)) +
  geom_histogram(binwidth = 0.02, alpha = 0.6, position = "identity") +
  geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = 1) +
  labs(
    title = "Predicted Probabilities by Actual Risk Group",
    x = "Predicted Probability of High Risk",
    y = "Count",
    fill = "Actual Risk"
  ) +
  scale_fill_manual(
    values = c("0" = "steelblue", "1" = "tomato"),
    labels = c("0" = "Low Risk", "1" = "High Risk")
  ) +
  theme_minimal()

# Coefficient plot
coef_df <- data.frame(
  term = names(coef(step_model)),
  estimate = coef(step_model)
)

ggplot(coef_df[-1, ], aes(x = reorder(term, estimate), y = estimate)) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  coord_flip() +
  labs(
    title = "Logistic Regression Coefficients (Step Model)",
    x = "Predictor",
    y = "Coefficient Estimate"
  ) +
  theme_minimal()
# Heart Risk Final Project
# Hannah Chen

# Load packages
library(tidyverse)
library(ggplot2)
library(dplyr)
library(MASS)
library(caret)
library(pROC)
library(randomForest)
library(xgboost)

# Load data
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")

# Look at the data
head(heart) # first few rows
dim(heart) # number of rows and columns
names(heart) # column names
str(heart) # variable types
summary(heart) # quick summary

# Check the outcome variable
table(heart$Heart.Attack.Risk..Binary.)
prop.table(table(heart$Heart.Attack.Risk..Binary.))

# Check for missing values
colSums(is.na(heart)) # missing values in each column
sum(!complete.cases(heart)) # number of incomplete rows

# Save incomplete rows
heart_missing <- heart[!complete.cases(heart), ] # pulls out those incomplete rows
dim(heart_missing)

# Check missing-data pattern
colSums(is.na(heart_missing)) # which columns are missing in the incomplete rows

# Optional check:
# if this gives 1, then all incomplete rows have the same missing pattern
nrow(unique(is.na(heart_missing)))

# Remove incomplete rows
heart_nomis <- na.omit(heart) # Removes rows whith at least one missing value

# Check cleaned data
colSums(is.na(heart_nomis)) # should all be 0
dim(heart_nomis) # number of rows and columns after removal

# Check variable names
names(heart_nomis)

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
# Final structure check
str(heart_nomis)

# EDA
# Bar plot for
ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary.)) +
  geom_bar(fill = "steelblue") +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3)+
  labs(title = "Distribution of Heart Attack Risk",
       x = "Heart Attack Risk",
       y = "Count") +
  scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
  theme_minimal()

table(heart_nomis$Heart.Attack.Risk..Binary.)
prop.table(table(heart_nomis$Heart.Attack.Risk..Binary.))

# 
ggplot(heart_nomis, aes(x = Gender, fill = Heart.Attack.Risk..Binary.)) +
  geom_bar(position = "fill") +
  labs(title = "Heart Attack Risk by Gender",
       x = "Gender",
       y = "Proportion",
       fill = "Risk Group") +
  scale_fill_discrete(labels = c("Low Risk", "High Risk")) +
  theme_minimal()

table(heart_nomis$Gender, heart_nomis$Heart.Attack.Risk..Binary.)
prop.table(table(heart_nomis$Gender, heart_nomis$Heart.Attack.Risk..Binary.), margin = 1)

# Smoking vs Heart Attack Risk
ggplot(heart_nomis, aes(x = Smoking, fill = Heart.Attack.Risk..Binary.)) +
  geom_bar(position = "fill") +
  labs(title = "Heart Attack Risk by Smoking Status",
       x = "Smoking",
       y = "Proportion",
       fill = "Risk Group") +
  scale_x_discrete(labels = c("0" = "Non-Smoker", "1" = "Smoker")) +
  scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato"),
                    labels = c("0" = "Low Risk", "1" = "High Risk")) +
  theme_minimal()

prop.table(table(heart_nomis$Smoking, heart_nomis$Heart.Attack.Risk..Binary.), margin = 1)

#
ggplot(heart_nomis, aes(x = Diabetes, fill = Heart.Attack.Risk..Binary.)) +
  geom_bar(position = "fill") +
  labs(title = "Heart Attack Risk by Diabetes Status",
       x = "Diabetes",
       y = "Proportion",
       fill = "Risk Group") +
  scale_x_discrete(labels = c("0" = "No Diabetes", "1" = "Diabetes")) +
  scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato"),
                    labels = c("0" = "Low Risk", "1" = "High Risk")) +
  theme_minimal()

prop.table(table(heart_nomis$Diabetes, heart_nomis$Heart.Attack.Risk..Binary.), margin = 1)

#
ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary., y = Cholesterol, fill = Heart.Attack.Risk..Binary.)) +
  geom_boxplot() +
  labs(title = "Cholesterol by Heart Attack Risk",
       x = "Heart Attack Risk",
       y = "Cholesterol") +
  scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
  scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato"),
                    labels = c("0" = "Low Risk", "1" = "High Risk")) +
  theme_minimal()


# Train/test split
set.seed(123) # get the same train/test split again

# Create the row numbers for the training set
train_index <- createDataPartition(heart_nomis$Heart.Attack.Risk..Binary.,
                                   p = 0.7,
                                   list = FALSE)

train_data <- heart_nomis[train_index, ] # gets the selected rows
test_data <- heart_nomis[-train_index, ] # gets everything else

# Check dimensions and proportions
dim(train_data)
dim(test_data)
prop.table(table(train_data$Heart.Attack.Risk..Binary.))
prop.table(table(test_data$Heart.Attack.Risk..Binary.))

# Logistic regression model
full_model <- glm(Heart.Attack.Risk..Binary. ~ Age + Cholesterol + Heart.rate +
                    Diabetes + Family.History + Smoking + Obesity +
                    Alcohol.Consumption + Exercise.Hours.Per.Week + Diet +
                    Previous.Heart.Problems + Medication.Use + Stress.Level +
                    Sedentary.Hours.Per.Day + Income + BMI + Triglycerides +
                    Physical.Activity.Days.Per.Week + Sleep.Hours.Per.Day +
                    Blood.sugar + CK.MB + Troponin + Gender +
                    Systolic.blood.pressure + Diastolic.blood.pressure,
                  data = train_data,
                  family = binomial)

summary(full_model)

# Odds ratios
exp(coef(full_model))

# Run step AIC
step_model <- stepAIC(full_model, direction = "both", trace = FALSE)
summary(step_model)
AIC(full_model, step_model)

# Get predicted probabilities on test data
test_data$pred_prob <- predict(step_model, newdata = test_data, type = "response")
head(test_data$pred_prob)

# Turn probabilities into predicted classes
test_data$pred_class <- factor(
  ifelse(test_data$pred_prob > 0.5, "1", "0"),
  levels = c("0", "1")
)

# Confusion matrix
confusionMatrix(test_data$pred_class, test_data$Heart.Attack.Risk..Binary.)

logit_roc <- roc(test_data$Heart.Attack.Risk..Binary., test_data$pred_prob)
auc(logit_roc)

plot(logit_roc, main = "ROC Curve for Step Logistic Regression Model")
abline(a = 0, b = 1, lty = 2)

# Predicted probability histogram
ggplot(test_data, aes(x = pred_prob, fill = Heart.Attack.Risk..Binary.)) +
  geom_histogram(binwidth = 0.02, alpha = 0.6, position = "identity") +
  geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = 1) +
  labs(title = "Predicted Probabilities by Actual Risk Group",
       x = "Predicted Probability of High Risk",
       y = "Count",
       fill = "Actual Risk") +
  scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato"),
                    labels = c("0" = "Low Risk", "1" = "High Risk")) +
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
  labs(title = "Logistic Regression Coefficients (Step Model)",
       x = "Predictor",
       y = "Coefficient Estimate") +
  theme_minimal()


# Random forest model
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

rf_pred <- predict(rf_model, newdata = test_data)

rf_prob_1 <- predict(rf_model, newdata = test_data, type = "prob")[, "1"]

confusionMatrix(rf_pred, test_data$Heart.Attack.Risk..Binary.)

rf_roc <- roc(test_data$Heart.Attack.Risk..Binary., rf_prob_1)

plot(rf_roc,
     main = "ROC Curve for Random Forest",
     col = "blue",
     lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

auc(logit_roc)
auc(rf_roc)

# Compare ROC curves
logit_roc <- roc(test_data$Heart.Attack.Risk..Binary., test_data$pred_prob)
rf_roc <- roc(test_data$Heart.Attack.Risk..Binary., rf_prob_1)

plot(logit_roc,
     col = "red",
     lwd = 2,
     main = "ROC Curves: Logistic Regression vs Random Forest")

plot(rf_roc,
     col = "blue",
     lwd = 2,
     add = TRUE)

abline(a = 0, b = 1, lty = 2, col = "gray")

legend("bottomright",
       legend = c(
         paste("Logistic Regression AUC =", round(auc(logit_roc), 3)),
         paste("Random Forest AUC =", round(auc(rf_roc), 3))
       ),
       col = c("red", "blue"),
       lwd = 2)
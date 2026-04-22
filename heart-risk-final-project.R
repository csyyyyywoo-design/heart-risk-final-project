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
library(shiny)
library(shinydashboard)

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
heart_nomis$Heart.Attack.Risk..Text. <- as.factor(heart_nomis$Heart.Attack.Risk..Text.)

# Final structure check
str(heart_nomis)

# EDA
# Bar plot for
ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary.)) +
  geom_bar(fill = "steelblue") +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3)
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

#
library(caret)
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
library(MASS)
step_model <- stepAIC(full_model, direction = "both", trace = FALSE)
summary(step_model)
AIC(full_model, step_model)

# Get predicted probabilities on test data
test_data$pred_prob <- predict(step_model, newdata = test_data, type = "response")
head(test_data$pred_prob)

# Turn probabilities into predicted classes
test_data$pred_class <- ifelse(test_data$pred_prob > 0.5, "1", "0")
test_data$pred_class <- factor(test_data$pred_class, levels = c("0", "1"))

# Confusion matrix
confusionMatrix(test_data$pred_class, test_data$Heart.Attack.Risk..Binary.)

library(pROC)
roc_obj <- roc(test_data$Heart.Attack.Risk..Binary., test_data$pred_prob)
auc(roc_obj)

small_model <- glm(Heart.Attack.Risk..Binary. ~ Cholesterol + Systolic.blood.pressure,
                   data = train_data,
                   family = binomial)

summary(small_model)

test_data$small_prob <- predict(small_model, newdata = test_data, type = "response")
head(test_data$small_prob)

test_data$small_class <- ifelse(test_data$small_prob > 0.5, "1", "0")
test_data$small_class <- factor(test_data$small_class, levels = c("0", "1"))

confusionMatrix(test_data$small_class, test_data$Heart.Attack.Risk..Binary.)

small_roc <- roc(test_data$Heart.Attack.Risk..Binary., test_data$small_prob)
auc(small_roc)

# ROC curve
plot(roc_obj, main = "ROC Curve for Step Logistic Regression Model")
abline(a = 0, b = 1, lty = 2)

# histogram of predicted probabilities
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

# coefficient / odds ratio plot
exp(coef(step_model))
coef_df <- data.frame(
  term = names(coef(step_model)),
  estimate = coef(step_model)
)

ggplot(coef_df[-1, ], aes(x = reorder(term, estimate), y = estimate)) +
  geom_point() +
  coord_flip() +
  labs(title = "Logistic Regression Coefficients (Step Model)",
       x = "Predictor",
       y = "Coefficient Estimate") +
  theme_minimal()

# Try random forest model
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
head(rf_pred)

rf_prob <- predict(rf_model, newdata = test_data, type = "prob")
head(rf_prob)
rf_prob_1 <- rf_prob[, "1"]
head(rf_prob_1)

confusionMatrix(rf_pred, test_data$Heart.Attack.Risk..Binary.)

rf_prob_1 <- predict(rf_model, newdata = test_data, type = "prob")[, "1"]
rf_roc <- roc(test_data$Heart.Attack.Risk..Binary., rf_prob_1)
plot(rf_roc,
     main = "ROC Curve for Random Forest",
     col = "blue",
     lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
auc(rf_roc)

# Compare two auc curve
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




library(shiny)
library(shinydashboard)
library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(MASS)

# ── DATA PREP (runs once when app loads) ─────────────────────────────────────
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")
heart_nomis <- na.omit(heart)

# Convert to factors
factor_vars <- c("Diabetes","Family.History","Smoking","Obesity",
                 "Alcohol.Consumption","Diet","Previous.Heart.Problems",
                 "Medication.Use","Heart.Attack.Risk..Binary.","Gender",
                 "Heart.Attack.Risk..Text.")
heart_nomis[factor_vars] <- lapply(heart_nomis[factor_vars], as.factor)

# Train/test split
set.seed(123)
train_index <- createDataPartition(heart_nomis$Heart.Attack.Risk..Binary., p = 0.7, list = FALSE)
train_data  <- heart_nomis[train_index, ]
test_data   <- heart_nomis[-train_index, ]

# Models
full_model <- glm(Heart.Attack.Risk..Binary. ~ Age + Cholesterol + Heart.rate +
                    Diabetes + Family.History + Smoking + Obesity +
                    Alcohol.Consumption + Exercise.Hours.Per.Week + Diet +
                    Previous.Heart.Problems + Medication.Use + Stress.Level +
                    Sedentary.Hours.Per.Day + Income + BMI + Triglycerides +
                    Physical.Activity.Days.Per.Week + Sleep.Hours.Per.Day +
                    Blood.sugar + CK.MB + Troponin + Gender +
                    Systolic.blood.pressure + Diastolic.blood.pressure,
                  data = train_data, family = binomial)

step_model <- stepAIC(full_model, direction = "both", trace = FALSE)

test_data$pred_prob  <- predict(step_model, newdata = test_data, type = "response")
test_data$pred_class <- factor(ifelse(test_data$pred_prob > 0.5, "1", "0"), levels = c("0","1"))

rf_model <- randomForest(
  Heart.Attack.Risk..Binary. ~ Age + Cholesterol + Heart.rate +
    Diabetes + Family.History + Smoking + Obesity +
    Alcohol.Consumption + Exercise.Hours.Per.Week + Diet +
    Previous.Heart.Problems + Medication.Use + Stress.Level +
    Sedentary.Hours.Per.Day + Income + BMI + Triglycerides +
    Physical.Activity.Days.Per.Week + Sleep.Hours.Per.Day +
    Blood.sugar + CK.MB + Troponin + Gender +
    Systolic.blood.pressure + Diastolic.blood.pressure,
  data = train_data, ntree = 500, importance = TRUE
)

rf_prob_1 <- predict(rf_model, newdata = test_data, type = "prob")[, "1"]
rf_pred   <- predict(rf_model, newdata = test_data)

logit_roc <- roc(test_data$Heart.Attack.Risk..Binary., test_data$pred_prob)
rf_roc    <- roc(test_data$Heart.Attack.Risk..Binary., rf_prob_1)

cm_logit <- confusionMatrix(test_data$pred_class,  test_data$Heart.Attack.Risk..Binary.)
cm_rf    <- confusionMatrix(rf_pred, test_data$Heart.Attack.Risk..Binary.)


# ── UI ───────────────────────────────────────────────────────────────────────
ui <- dashboardPage(
  skin = "blue",

  dashboardHeader(title = "Heart Attack Risk"),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview",          tabName = "overview",   icon = icon("heart")),
      menuItem("EDA",               tabName = "eda",        icon = icon("chart-bar")),
      menuItem("Logistic Regression", tabName = "logit",    icon = icon("project-diagram")),
      menuItem("Random Forest",     tabName = "rf",         icon = icon("tree")),
      menuItem("Model Comparison",  tabName = "compare",    icon = icon("balance-scale"))
    )
  ),

  dashboardBody(
    tabItems(

      # ── Tab 1: Overview ───────────────────────────────────────────────────
      tabItem(tabName = "overview",
        fluidRow(
          # Three summary boxes at the top
          valueBoxOutput("n_patients"),
          valueBoxOutput("pct_highrisk"),
          valueBoxOutput("n_vars")
        ),
        fluidRow(
          box(title = "About This Dashboard", width = 12, status = "primary",
              solidHeader = TRUE,
              p("This dashboard explores heart attack risk prediction using logistic
                regression and random forest models."),
              p("Use the sidebar to navigate between exploratory data analysis,
                model results, and a head-to-head comparison.")
          )
        )
      ),

      # ── Tab 2: EDA ────────────────────────────────────────────────────────
      tabItem(tabName = "eda",
        fluidRow(
          box(title = "Explore a Categorical Variable", width = 12,
              status = "warning", solidHeader = TRUE,
              # Dropdown input — user picks a variable
              selectInput("eda_var",
                          "Select a variable to compare with Heart Attack Risk:",
                          choices = c("Gender","Smoking","Diabetes",
                                      "Obesity","Family.History",
                                      "Previous.Heart.Problems","Medication.Use")),
              plotOutput("eda_bar")   # plot goes here
          )
        ),
        fluidRow(
          box(title = "Cholesterol by Risk Group", width = 6,
              status = "warning", solidHeader = TRUE,
              plotOutput("chol_box")),
          box(title = "Outcome Distribution", width = 6,
              status = "warning", solidHeader = TRUE,
              plotOutput("outcome_bar"))
        )
      ),

      # ── Tab 3: Logistic Regression ────────────────────────────────────────
      tabItem(tabName = "logit",
        fluidRow(
          valueBoxOutput("logit_auc"),
          valueBoxOutput("logit_acc"),
          valueBoxOutput("logit_sens")
        ),
        fluidRow(
          box(title = "Predicted Probability Distribution", width = 6,
              status = "info", solidHeader = TRUE,
              plotOutput("pred_hist")),
          box(title = "Coefficient Plot (Step Model)", width = 6,
              status = "info", solidHeader = TRUE,
              plotOutput("coef_plot"))
        )
      ),

      # ── Tab 4: Random Forest ──────────────────────────────────────────────
      tabItem(tabName = "rf",
        fluidRow(
          valueBoxOutput("rf_auc"),
          valueBoxOutput("rf_acc"),
          valueBoxOutput("rf_sens")
        ),
        fluidRow(
          box(title = "Variable Importance", width = 12,
              status = "success", solidHeader = TRUE,
              plotOutput("rf_importance"))
        )
      ),

      # ── Tab 5: Model Comparison ───────────────────────────────────────────
      tabItem(tabName = "compare",
        fluidRow(
          box(title = "ROC Curve Comparison", width = 8,
              status = "danger", solidHeader = TRUE,
              plotOutput("roc_compare")),
          box(title = "Metrics Summary", width = 4,
              status = "danger", solidHeader = TRUE,
              tableOutput("metrics_table"))
        )
      )
    )
  )
)


# ── SERVER ───────────────────────────────────────────────────────────────────
server <- function(input, output) {

  # Overview value boxes
  output$n_patients  <- renderValueBox({
    valueBox(nrow(heart_nomis), "Total Patients", icon = icon("users"), color = "blue")
  })
  output$pct_highrisk <- renderValueBox({
    pct <- round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "1") * 100, 1)
    valueBox(paste0(pct, "%"), "High Risk", icon = icon("exclamation-triangle"), color = "red")
  })
  output$n_vars <- renderValueBox({
    valueBox(25, "Predictors", icon = icon("list"), color = "green")
  })

  # EDA — reactive bar chart (changes when user picks a variable)
  output$eda_bar <- renderPlot({
    ggplot(heart_nomis, aes(x = .data[[input$eda_var]],
                             fill = Heart.Attack.Risk..Binary.)) +
      geom_bar(position = "fill") +
      labs(title = paste("Heart Attack Risk by", input$eda_var),
           x = input$eda_var, y = "Proportion", fill = "Risk") +
      scale_fill_manual(values = c("0" = "steelblue","1" = "tomato"),
                        labels = c("Low Risk","High Risk")) +
      theme_minimal(base_size = 14)
  })

  output$chol_box <- renderPlot({
    ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary., y = Cholesterol,
                             fill = Heart.Attack.Risk..Binary.)) +
      geom_boxplot() +
      scale_x_discrete(labels = c("0" = "Low Risk","1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "steelblue","1" = "tomato")) +
      labs(x = "Risk Group", y = "Cholesterol") +
      theme_minimal(base_size = 14) + theme(legend.position = "none")
  })

  output$outcome_bar <- renderPlot({
    ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary.,
                             fill = Heart.Attack.Risk..Binary.)) +
      geom_bar() +
      geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3) +
      scale_x_discrete(labels = c("0" = "Low Risk","1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "steelblue","1" = "tomato")) +
      labs(x = "Risk Group", y = "Count") +
      theme_minimal(base_size = 14) + theme(legend.position = "none")
  })

  # Logistic regression outputs
  output$logit_auc  <- renderValueBox({
    valueBox(round(auc(logit_roc), 3), "AUC", icon = icon("chart-line"), color = "blue")
  })
  output$logit_acc  <- renderValueBox({
    valueBox(round(cm_logit$overall["Accuracy"], 3),
             "Accuracy", icon = icon("check"), color = "green")
  })
  output$logit_sens <- renderValueBox({
    valueBox(round(cm_logit$byClass["Sensitivity"], 3),
             "Sensitivity", icon = icon("stethoscope"), color = "yellow")
  })

  output$pred_hist <- renderPlot({
    ggplot(test_data, aes(x = pred_prob, fill = Heart.Attack.Risk..Binary.)) +
      geom_histogram(binwidth = 0.02, alpha = 0.6, position = "identity") +
      geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = 1) +
      scale_fill_manual(values = c("0" = "steelblue","1" = "tomato"),
                        labels = c("Low Risk","High Risk")) +
      labs(x = "Predicted Probability", y = "Count", fill = "Actual Risk") +
      theme_minimal(base_size = 14)
  })

  output$coef_plot <- renderPlot({
    coef_df <- data.frame(term = names(coef(step_model)),
                          estimate = coef(step_model))
    ggplot(coef_df[-1, ], aes(x = reorder(term, estimate), y = estimate)) +
      geom_point(color = "steelblue", size = 3) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
      coord_flip() +
      labs(x = "Predictor", y = "Coefficient") +
      theme_minimal(base_size = 12)
  })

  # Random forest outputs
  output$rf_auc  <- renderValueBox({
    valueBox(round(auc(rf_roc), 3), "AUC", icon = icon("chart-line"), color = "blue")
  })
  output$rf_acc  <- renderValueBox({
    valueBox(round(cm_rf$overall["Accuracy"], 3),
             "Accuracy", icon = icon("check"), color = "green")
  })
  output$rf_sens <- renderValueBox({
    valueBox(round(cm_rf$byClass["Sensitivity"], 3),
             "Sensitivity", icon = icon("stethoscope"), color = "yellow")
  })

  output$rf_importance <- renderPlot({
    imp_df <- as.data.frame(importance(rf_model))
    imp_df$Variable <- rownames(imp_df)
    ggplot(imp_df, aes(x = reorder(Variable, MeanDecreaseGini),
                        y = MeanDecreaseGini)) +
      geom_col(fill = "forestgreen") +
      coord_flip() +
      labs(x = "Variable", y = "Mean Decrease in Gini") +
      theme_minimal(base_size = 12)
  })

  # Model comparison
  output$roc_compare <- renderPlot({
    plot(logit_roc, col = "red", lwd = 2,
         main = "ROC Curves: Logistic Regression vs Random Forest")
    plot(rf_roc, col = "blue", lwd = 2, add = TRUE)
    abline(a = 0, b = 1, lty = 2, col = "gray")
    legend("bottomright",
           legend = c(paste("Logistic AUC =", round(auc(logit_roc), 3)),
                      paste("Random Forest AUC =", round(auc(rf_roc), 3))),
           col = c("red","blue"), lwd = 2)
  })

  output$metrics_table <- renderTable({
    data.frame(
      Metric    = c("AUC","Accuracy","Sensitivity","Specificity"),
      Logistic  = c(round(auc(logit_roc), 3),
                    round(cm_logit$overall["Accuracy"], 3),
                    round(cm_logit$byClass["Sensitivity"], 3),
                    round(cm_logit$byClass["Specificity"], 3)),
      RF        = c(round(auc(rf_roc), 3),
                    round(cm_rf$overall["Accuracy"], 3),
                    round(cm_rf$byClass["Sensitivity"], 3),
                    round(cm_rf$byClass["Specificity"], 3))
    )
  })
}

shinyApp(ui, server)
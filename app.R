# ============================================================
# Final Project Dashboard
# Heart Risk Statistical Analysis
# Hannah Chen and Kyndal Schlup
# ============================================================

library(shiny)
library(bslib)
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(randomForest)
library(xgboost)
library(scales)

# ----------------------------
# Load and clean data
# ----------------------------
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")
heart_nomis <- na.omit(heart)

# Convert categorical variables consistently
heart_nomis$Alcohol.Consumption <- factor(heart_nomis$Alcohol.Consumption, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Diet <- as.factor(heart_nomis$Diet)
heart_nomis$Previous.Heart.Problems <- factor(heart_nomis$Previous.Heart.Problems, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Medication.Use <- factor(heart_nomis$Medication.Use, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Diabetes <- factor(heart_nomis$Diabetes, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Family.History <- factor(heart_nomis$Family.History, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Smoking <- factor(heart_nomis$Smoking, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Obesity <- factor(heart_nomis$Obesity, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Heart.Attack.Risk..Binary. <- factor(heart_nomis$Heart.Attack.Risk..Binary., levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Gender <- factor(heart_nomis$Gender)

# ----------------------------
# Train / test split
# ----------------------------
set.seed(123)
train_index <- createDataPartition(
  heart_nomis$Heart.Attack.Risk..Binary.,
  p = 0.8,
  list = FALSE
)

train_data <- heart_nomis[train_index, ]
test_data  <- heart_nomis[-train_index, ]

# ----------------------------
# Model formulas
# ----------------------------
full_formula <- Heart.Attack.Risk..Binary. ~
  Age + Cholesterol + Heart.rate +
  Diabetes + Family.History + Smoking + Obesity +
  Alcohol.Consumption + Exercise.Hours.Per.Week + Diet +
  Previous.Heart.Problems + Medication.Use + Stress.Level +
  Sedentary.Hours.Per.Day + Income + BMI + Triglycerides +
  Physical.Activity.Days.Per.Week + Sleep.Hours.Per.Day +
  Blood.sugar + CK.MB + Troponin + Gender +
  Systolic.blood.pressure + Diastolic.blood.pressure

# These are the variables we chose to focus on for the calculator.
model_formula <- Heart.Attack.Risk..Binary. ~
  Age + Gender + Diabetes + Smoking + Obesity + Family.History +
  Cholesterol + Heart.rate + BMI + Triglycerides +
  Blood.sugar + CK.MB + Troponin

# ----------------------------
# Data-driven Low/Med/High mapping
# ----------------------------
# The dataset is scaled, so Low/Med/High should be mapped to actual data values,
# not fixed values like 0.2 / 0.5 / 0.8 for every variable.
calc_vars <- c("Age", "Cholesterol", "Heart.rate", "BMI", "Triglycerides", "Blood.sugar", "CK.MB", "Troponin")

quantile_map <- lapply(calc_vars, function(v) {
  vals <- train_data[[v]]
  q <- quantile(vals, probs = c(0.25, 0.50, 0.75), na.rm = TRUE)
  if (q[1] == q[3]) {
    q <- quantile(vals, probs = c(0.10, 0.50, 0.90), na.rm = TRUE)
  }
  if (q[2] == q[3]) {
    q <- c(min(vals, na.rm = TRUE), median(vals, na.rm = TRUE), max(vals, na.rm = TRUE))
  }
  setNames(as.numeric(q), c("Low", "Med", "High"))
})
names(quantile_map) <- calc_vars

map_to_quantile <- function(level, variable) {
  as.numeric(quantile_map[[variable]][[level]])
}

age_quantiles <- quantile(train_data$Age, probs = c(0.10, 0.25, 0.50, 0.75, 0.90), na.rm = TRUE)
age_group_map <- c(
  "Under 40" = as.numeric(age_quantiles[1]),
  "40-50"   = as.numeric(age_quantiles[2]),
  "50-60"   = as.numeric(age_quantiles[3]),
  "60-70"   = as.numeric(age_quantiles[4]),
  "70+"     = as.numeric(age_quantiles[5])
)

# ----------------------------
# Fit models
# ----------------------------
# Logistic regression final model from the peer version
log_model_final <- glm(
  Heart.Attack.Risk..Binary. ~ Obesity + Cholesterol + Diabetes,
  data = train_data,
  family = binomial
)

# Logistic calculator model using selected calculator variables
log_model_calculator <- glm(model_formula, data = train_data, family = binomial)

# Random forest with all predictors for model performance
# This keeps the stronger RF AUC around 0.59.
set.seed(123)
rf_model <- randomForest(
  full_formula,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# Random forest with focus predictors for calculator
set.seed(123)
rf_model_calculator <- randomForest(
  model_formula,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# XGBoost needs numeric labels
train_label <- as.numeric(as.character(train_data$Heart.Attack.Risk..Binary.))
test_label  <- as.numeric(as.character(test_data$Heart.Attack.Risk..Binary.))

# XGBoost model for performance section
# This uses the focused predictor set to keep XGBoost consistent with the calculator.
x_train_xgb <- model.matrix(model_formula, data = train_data)[, -1, drop = FALSE]
x_test_xgb  <- model.matrix(model_formula, data = test_data)[, -1, drop = FALSE]

dtrain_xgb <- xgb.DMatrix(data = as.matrix(x_train_xgb), label = train_label)
dtest_xgb  <- xgb.DMatrix(data = as.matrix(x_test_xgb), label = test_label)

set.seed(123)
xgb_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 3,
    eta = 0.1
  ),
  data = dtrain_xgb,
  nrounds = 100,
  verbose = 0
)

# XGBoost calculator model
x_train_calc <- model.matrix(model_formula, data = train_data)[, -1, drop = FALSE]
x_test_calc  <- model.matrix(model_formula, data = test_data)[, -1, drop = FALSE]

dtrain_calc <- xgb.DMatrix(data = as.matrix(x_train_calc), label = train_label)
dtest_calc  <- xgb.DMatrix(data = as.matrix(x_test_calc), label = test_label)

set.seed(123)
xgb_model_calculator <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 3,
    eta = 0.1
  ),
  data = dtrain_calc,
  nrounds = 100,
  verbose = 0
)

# ----------------------------
# Evaluate models
# ----------------------------
# Logistic regression
test_data$log_prob <- predict(log_model_final, newdata = test_data, type = "response")
test_data$log_class <- factor(ifelse(test_data$log_prob > 0.5, "1", "0"), levels = c("0", "1"))
cm_log <- confusionMatrix(test_data$log_class, test_data$Heart.Attack.Risk..Binary., positive = "1")
roc_log <- roc(response = test_data$Heart.Attack.Risk..Binary., predictor = test_data$log_prob, levels = c("0", "1"), quiet = TRUE)

# Random forest
rf_pred <- predict(rf_model, newdata = test_data)
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "1"]
cm_rf <- confusionMatrix(rf_pred, test_data$Heart.Attack.Risk..Binary., positive = "1")
roc_rf <- roc(response = test_data$Heart.Attack.Risk..Binary., predictor = rf_prob, levels = c("0", "1"), quiet = TRUE)

# XGBoost
xgb_prob <- predict(xgb_model, newdata = dtest_xgb)
xgb_pred <- factor(ifelse(xgb_prob > 0.5, "1", "0"), levels = c("0", "1"))
cm_xgb <- confusionMatrix(xgb_pred, test_data$Heart.Attack.Risk..Binary., positive = "1")
roc_xgb <- roc(response = test_data$Heart.Attack.Risk..Binary., predictor = xgb_prob, levels = c("0", "1"), quiet = TRUE)

# ----------------------------
# Helper functions
# ----------------------------
yes_no_to_factor <- function(x) {
  switch(x, "Yes" = "1", "No" = "0", "0")
}

score_weight <- function(x) {
  switch(x, "Low" = 0.2, "Med" = 0.5, "High" = 0.8, 0.5)
}

binary_score <- function(x) {
  switch(x, "No" = 0, "Yes" = 0.8, "0" = 0, "1" = 0.8, 0)
}

rule_score <- function(age, cholesterol, heart_rate, bmi, triglycerides, blood_sugar,
                       ckmb, troponin, smoking, diabetes, family_history, obesity) {
  s <- 0

  if (age %in% c("60-70", "70+")) {
    s <- s + 1
  }

  s <- s + score_weight(cholesterol)
  s <- s + score_weight(heart_rate)
  s <- s + score_weight(bmi)
  s <- s + score_weight(triglycerides)
  s <- s + score_weight(blood_sugar)

  s <- s + 2 * score_weight(ckmb)
  s <- s + 3 * score_weight(troponin)

  s <- s + binary_score(smoking)
  s <- s + binary_score(diabetes)
  s <- s + binary_score(family_history)
  s <- s + binary_score(obesity)

  return(s)
}

summary_stats <- function(data, var) {
  data %>%
    group_by(Heart.Attack.Risk..Binary.) %>%
    summarise(
      Mean = round(mean(.data[[var]], na.rm = TRUE), 3),
      Median = round(median(.data[[var]], na.rm = TRUE), 3),
      SD = round(sd(.data[[var]], na.rm = TRUE), 3),
      Min = round(min(.data[[var]], na.rm = TRUE), 3),
      Max = round(max(.data[[var]], na.rm = TRUE), 3),
      .groups = "drop"
    ) %>%
    mutate(
      RiskGroup = ifelse(Heart.Attack.Risk..Binary. == "0", "Low Risk", "High Risk")
    ) %>%
    dplyr::select(RiskGroup, Mean, Median, SD, Min, Max)
}

find_similar_patient <- function(patient_df, dataset) {
  num_cols <- c("Age", "Cholesterol", "Heart.rate", "BMI", "Triglycerides", "Blood.sugar", "CK.MB", "Troponin")
  cat_cols <- c("Gender", "Diabetes", "Smoking", "Obesity", "Family.History")

  dists_num <- sapply(num_cols, function(col) {
    rng <- max(dataset[[col]], na.rm = TRUE) - min(dataset[[col]], na.rm = TRUE)
    if (rng == 0) return(rep(0, nrow(dataset)))
    abs(as.numeric(patient_df[[col]]) - dataset[[col]]) / rng
  })

  dists_cat <- sapply(cat_cols, function(col) {
    ifelse(as.character(patient_df[[col]]) == as.character(dataset[[col]]), 0, 1)
  })

  total_dist <- rowMeans(cbind(dists_num, dists_cat))
  best_idx <- which.min(total_dist)
  similarity <- round((1 - total_dist[best_idx]) * 100, 1)

  list(
    patient = dataset[best_idx, ],
    similarity = similarity,
    index = best_idx
  )
}

metric_df <- data.frame(
  Metric = c("AUC", "Accuracy", "Sensitivity", "Specificity"),
  Logistic = c(
    round(as.numeric(auc(roc_log)), 3),
    round(unname(cm_log$overall["Accuracy"]), 3),
    round(unname(cm_log$byClass["Sensitivity"]), 3),
    round(unname(cm_log$byClass["Specificity"]), 3)
  ),
  RandomForest = c(
    round(as.numeric(auc(roc_rf)), 3),
    round(unname(cm_rf$overall["Accuracy"]), 3),
    round(unname(cm_rf$byClass["Sensitivity"]), 3),
    round(unname(cm_rf$byClass["Specificity"]), 3)
  ),
  XGBoost = c(
    round(as.numeric(auc(roc_xgb)), 3),
    round(unname(cm_xgb$overall["Accuracy"]), 3),
    round(unname(cm_xgb$byClass["Sensitivity"]), 3),
    round(unname(cm_xgb$byClass["Specificity"]), 3)
  ),
  check.names = FALSE
)
rownames(metric_df) <- NULL

tab <- table(heart_nomis$Heart.Attack.Risk..Binary.)
prop <- prop.table(tab)

outcome_summary_df <- data.frame(
  RiskGroup = c("Low Risk", "High Risk"),
  Count = as.numeric(tab),
  Proportion = round(as.numeric(prop) * 100, 1)
)

manual_cv_metrics <- function(formula, data, seed = 123) {
  aucs <- numeric(5)
  kappas <- numeric(5)

  for (i in 1:5) {
    set.seed(seed + i)
    idx <- createDataPartition(data$Heart.Attack.Risk..Binary., p = 0.8, list = FALSE)
    train_fold <- data[idx, ]
    test_fold  <- data[-idx, ]

    fit <- glm(formula, data = train_fold, family = binomial)
    prob <- predict(fit, newdata = test_fold, type = "response")
    pred <- factor(ifelse(prob > 0.5, "1", "0"), levels = c("0", "1"))

    roc_obj <- roc(response = test_fold$Heart.Attack.Risk..Binary., predictor = prob, levels = c("0", "1"), quiet = TRUE)
    cm <- confusionMatrix(pred, test_fold$Heart.Attack.Risk..Binary., positive = "1")

    aucs[i] <- as.numeric(auc(roc_obj))
    kappas[i] <- as.numeric(cm$overall["Kappa"])
  }

  c(Mean_AUC = round(mean(aucs), 3), Mean_Kappa = round(mean(kappas), 3))
}

cv_results <- data.frame(
  Model = c("Model 1", "Model 2", "Model 3", "Model 4", "Final Model"),
  Predictors = c(
    "Cholesterol + Sleep Hours + Stress Level + Systolic BP",
    "Cholesterol + Gender",
    "Troponin + CK-MB",
    "Obesity * Cholesterol * Diabetes",
    "Obesity + Cholesterol + Diabetes"
  ),
  rbind(
    manual_cv_metrics(Heart.Attack.Risk..Binary. ~ Cholesterol + Sleep.Hours.Per.Day + Stress.Level + Systolic.blood.pressure, heart_nomis),
    manual_cv_metrics(Heart.Attack.Risk..Binary. ~ Cholesterol + Gender, heart_nomis),
    manual_cv_metrics(Heart.Attack.Risk..Binary. ~ Troponin + CK.MB, heart_nomis),
    manual_cv_metrics(Heart.Attack.Risk..Binary. ~ Obesity * Cholesterol * Diabetes, heart_nomis),
    manual_cv_metrics(Heart.Attack.Risk..Binary. ~ Obesity + Cholesterol + Diabetes, heart_nomis)
  ),
  check.names = FALSE
)

rf_imp <- as.data.frame(importance(rf_model))
rf_imp$Feature <- rownames(rf_imp)
rf_imp <- rf_imp %>%
  arrange(desc(MeanDecreaseGini)) %>%
  slice(1:10)

# ----------------------------
# Theme
# ----------------------------
theme_modern <- bs_theme(
  version = 5,
  bg = "#f8fafc",
  fg = "#0f172a",
  primary = "#2563eb",
  secondary = "#64748b",
  base_font = font_google("Inter"),
  heading_font = font_google("Inter"),
  code_font = font_google("JetBrains Mono")
)

# ----------------------------
# UI
# ----------------------------
ui <- fluidPage(
  theme = theme_modern,

  tags$head(
    tags$style(HTML("
      body { background: #f8fafc; }
      .page-wrap { max-width: 1320px; margin: 0 auto; padding: 28px 20px 40px 20px; }
      .hero { background: linear-gradient(135deg, #eff6ff 0%, #ffffff 55%, #f8fafc 100%); border: 1px solid #e2e8f0; border-radius: 28px; padding: 34px; margin-bottom: 26px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06); }
      .eyebrow { display: inline-block; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #2563eb; background: #dbeafe; border-radius: 999px; padding: 6px 12px; margin-bottom: 14px; }
      .hero-title { font-size: 2.7rem; font-weight: 800; line-height: 1.05; color: #0f172a; margin-bottom: 10px; }
      .hero-subtitle { font-size: 1.05rem; line-height: 1.7; color: #475569; max-width: 760px; margin-bottom: 0; }
      .hero-side { background: rgba(255,255,255,0.9); border: 1px solid #e2e8f0; border-radius: 22px; padding: 22px; height: auto; align-self: flex-start; margin-bottom: 15px; }
      .hero-side h4 { font-weight: 800; font-size: 1.1rem; margin-top: 0; margin-bottom: 14px; color: #0f172a; }
      .hero-side ul { padding-left: 18px; margin-bottom: 0; color: #475569; line-height: 1.7; }
      .section-block { margin-bottom: 26px; }
      .section-header { margin: 6px 0 14px 0; }
      .section-kicker { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 700; margin-bottom: 5px; }
      .section-title { font-size: 1.65rem; font-weight: 800; color: #0f172a; margin: 0; }
      .section-note { color: #64748b; margin-top: 6px; margin-bottom: 0; font-size: 1rem; }
      .stat-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 22px; padding: 22px 22px 20px 22px; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04); margin-bottom: 18px; height: 100%; }
      .stat-label { color: #64748b; font-size: 0.92rem; margin-bottom: 10px; }
      .stat-value { font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1; margin-bottom: 8px; }
      .stat-subtext { color: #475569; font-size: 0.95rem; }
      .panel-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 24px; padding: 22px; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04); margin-bottom: 20px; }
      .panel-card h3 { font-size: 1.1rem; font-weight: 800; margin-top: 0; margin-bottom: 8px; color: #0f172a; }
      .panel-desc { color: #64748b; font-size: 0.95rem; line-height: 1.6; margin-bottom: 14px; }
      .insight-box { background: #fff7ed; border: 1px solid #fed7aa; border-radius: 20px; padding: 18px; color: #7c2d12; }
      .soft-blue { background: #eff6ff; border-color: #bfdbfe; }
      .soft-slate { background: #f8fafc; }
      .mini-badge { display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; margin-right: 8px; margin-bottom: 8px; }
      .badge-purple { background: #ede9fe; color: #6d28d9; }
      .badge-amber { background: #fef3c7; color: #b45309; }
      .badge-emerald { background: #dcfce7; color: #15803d; }
      .badge-indigo { background: #e0e7ff; color: #4338ca; }
      .risk-meter-bg { height: 14px; width: 100%; background: #e2e8f0; border-radius: 999px; overflow: hidden; margin-top: 14px; }
      .metric-chip { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 16px; padding: 14px 16px; margin-bottom: 12px; }
      .metric-chip-label { color: #64748b; font-size: 0.88rem; margin-bottom: 6px; }
      .metric-chip-value { font-size: 1.2rem; font-weight: 800; color: #0f172a; }
      .shiny-input-container { width: 100% !important; }
      .form-block-title { font-size: 0.95rem; font-weight: 800; color: #334155; margin-top: 8px; margin-bottom: 10px; }
      .footer-note { color: #64748b; font-size: 0.92rem; text-align: center; margin-top: 10px; }
      .table-scroll { overflow-x: auto; width: 100%; }
      .table-scroll table { width: 100% !important; min-width: 760px; }
      .table { min-width: 680px; }
      .nav-tabs .nav-link { font-weight: 700; border-radius: 14px !important; margin-right: 8px; }
    "))
  ),

  div(
    class = "page-wrap",

    # Hero section
    div(
      class = "hero",
      fluidRow(
        column(
          7,
          div(class = "eyebrow", "Statistical Methods 2 Final Project"),
          div(class = "hero-title", "Heart Risk Statistical Analysis"),
          div(style = "margin-top:12px; margin-bottom:12px; font-size:0.95rem; color:#64748b;", "By Hannah Chen and Kyndal Schlup"),
          div(class = "hero-subtitle", "An interactive dashboard to display our statistical process, model results, and compare our models with a more interpretable rule-based risk score."),
          div(
            style = "margin-top:18px;",
            span(class = "mini-badge badge-purple", "Logistic Regression"),
            span(class = "mini-badge badge-amber", "Random Forest"),
            span(class = "mini-badge badge-emerald", "XGBoost"),
            span(class = "mini-badge badge-indigo", "Patient Similarity")
          ),
          div(
            class = "hero-side",
            style = "margin-top:20px;",
            h4("Key Takeaways"),
            tags$ul(
              tags$li("All models perform close to chance level on this dataset."),
              tags$li("Troponin and CK-MB stand out as clinically meaningful markers."),
              tags$li("The rule-based score is the most interpretable part of this project."),
              tags$li("The patient similarity section compares the entered profile with the closest observation in the dataset."),
              tags$li("This dashboard presents findings and is not sufficient for clinical use.")
            )
          )
        ),
        column(
          5,
          div(
            class = "hero-side",
            h4("About The Data"),
            tags$ul(
              tags$li("Heart Attack Risk Prediction Cleaned Dataset from Kaggle"),
              tags$li("Dataset designed for predicting heart attack risk"),
              tags$li("All numerical values are normalized or scaled"),
              tags$li("Response Variable: Heart Attack Risk (Binary)"),
              tags$li("Predictors are weak")
            )
          ),
          div(
            class = "hero-side",
            h4("Statistical Process"),
            tags$ul(
              tags$li("Exploratory Data Analysis"),
              tags$li("Logistic Regression Model"),
              tags$li("Random Forest Model"),
              tags$li("XGBoost Model"),
              tags$li("Rule-based Scoring System"),
              tags$li("Patient Similarity Comparison"),
              tags$li("Dashboard")
            )
          )
        )
      )
    ),

    # Snapshot cards
    div(
      class = "section-block",
      div(
        class = "section-header",
        div(class = "section-kicker", "Snapshot"),
        h2(class = "section-title", "The Cleaned Data"),
        p(class = "section-note", "A quick overview of the cleaned dataset used in the analysis.")
      ),
      fluidRow(
        column(3, div(class = "stat-card", div(class = "stat-label", "Complete observations"), div(class = "stat-value", textOutput("vb_n")), div(class = "stat-subtext", "Rows remaining after removing missing values"))),
        column(3, div(class = "stat-card", div(class = "stat-label", "High-risk cases"), div(class = "stat-value", textOutput("vb_high")), div(class = "stat-subtext", "Share labeled high risk"))),
        column(3, div(class = "stat-card", div(class = "stat-label", "Low-risk cases"), div(class = "stat-value", textOutput("vb_low")), div(class = "stat-subtext", "Share labeled low risk"))),
        column(3, div(class = "stat-card", div(class = "stat-label", "Best model AUC"), div(class = "stat-value", textOutput("vb_aucbest")), div(class = "stat-subtext", "Highest AUC among fitted models")))
      )
    ),

    # Tabs
    navset_card_tab(
      nav_panel(
        "Overview",
        br(),
        fluidRow(
          column(
            7,
            div(
              class = "panel-card",
              h3("Project Framing"),
              p(class = "panel-desc", HTML("<strong>The logistic regression, random forest, and XGBoost models all performed near chance level</strong>, with AUC values ranging from approximately 0.5 to 0.6. Due to the weak predictors in the dataset, this project emphasizes interpretation and clinical reasoning rather than claiming strong predictive power.")),
              div(class = "insight-box", HTML("<strong>Important:</strong> This dashboard is intended to be interpreted as a course-project analysis, not a real clinical decision tool. The rule-based score is included to provide a clearer and more explainable summary of known cardiac risk markers."))
            )
          ),
          column(
            5,
            div(
              class = "panel-card soft-blue",
              h3("Our Alternative"),
              p(class = "panel-desc", "Instead of relying only on weak model predictions, we created a rule-based scoring system that gives additional weight to clinically important biomarkers."),
              tags$ul(
                style = "color:#334155; line-height:1.8; margin-bottom:0;",
                tags$li("Troponin receives the strongest weight"),
                tags$li("CK-MB also contributes more heavily"),
                tags$li("Other markers provide supporting evidence"),
                tags$li("The results allow comparison between rule-based logic and model-based predictions.")
              )
            )
          )
        ),
        fluidRow(
          column(7, div(class = "panel-card", h3("Outcome Distribution"), p(class = "panel-desc", "This chart shows the balance between low-risk and high-risk observations in the cleaned dataset."), plotOutput("outcome_plot", height = "320px"))),
          column(
            5,
            div(
              class = "panel-card soft-slate",
              h3("Scoring System Summary"),
              p(class = "panel-desc", "This rule-based scoring system assigns risk points based on clinical indicators. Higher values indicate higher predicted risk. This is a heuristic clinical risk score, not a statistical probability model."),
              div(class = "metric-chip", div(class = "metric-chip-label", "Age 60+ or 70+"), div(class = "metric-chip-value", "+1 point")),
              div(class = "metric-chip", div(class = "metric-chip-label", "Cholesterol, Heart rate, BMI, Triglycerides, Blood sugar"), div(class = "metric-chip-value", "Low = 0.2, Med = 0.5, High = 0.8")),
              div(class = "metric-chip", div(class = "metric-chip-label", "CK-MB"), div(class = "metric-chip-value", "x2 weight")),
              div(class = "metric-chip", div(class = "metric-chip-label", "Troponin"), div(class = "metric-chip-value", "x3 weight")),
              div(class = "metric-chip", div(class = "metric-chip-label", "Smoking / Diabetes / Family history / Obesity"), div(class = "metric-chip-value", "Yes = +0.8 each, No = 0"))
            )
          )
        )
      ),

      nav_panel(
        "Clinical Signals",
        br(),
        div(class = "section-header", div(class = "section-kicker", "Exploration"), h2(class = "section-title", "Clinical Signals"), p(class = "section-note", "Use the controls below to explore how selected variables relate to risk groups.")),
        fluidRow(
          column(12, div(class = "panel-card", h3("Explore a Categorical Variable"), p(class = "panel-desc", "Switch the selected variable to compare how the low-risk and high-risk groups are distributed."), fluidRow(column(4, selectInput("eda_var", "Choose a variable", choices = c("Gender", "Smoking", "Diabetes", "Obesity", "Family.History")))), plotOutput("eda_bar", height = "320px")))
        ),
        fluidRow(
          column(6, div(class = "panel-card", h3("CK-MB by Risk Group"), p(class = "panel-desc", "CK-MB is one of the clinically important markers used when discussing possible heart muscle damage."), plotOutput("ckmb_plot", height = "320px"))),
          column(6, div(class = "panel-card", h3("Troponin by Risk Group"), p(class = "panel-desc", "Troponin is one of the most clinically important markers in detecting heart muscle damage."), plotOutput("troponin_plot", height = "320px")))
        ),
        fluidRow(
          column(12, div(class = "panel-card", h3("Marker Summary Table"), p(class = "panel-desc", "Choose a marker to compare its descriptive statistics across the two risk groups."), fluidRow(column(4, selectInput("summary_var", "Choose a marker", choices = c("Troponin", "CK.MB", "Cholesterol", "Heart.rate", "BMI", "Triglycerides", "Blood.sugar"), selected = "Troponin"))), div(class = "table-scroll", tableOutput("summary_table"))))
        )
      ),

      nav_panel(
        "Model Performance",
        br(),
        div(
          class = "section-header",
          div(class = "section-kicker", "Models"),
          h2(class = "section-title", "Model Performance"),
          p(
            class = "section-note",
            "These results are shown transparently, even though the models do not perform strongly on this dataset."
          )
        ),

        # Put the metrics table on its own full-width row so all model columns show.
        fluidRow(
          column(
            12,
            div(
              class = "panel-card",
              h3("Performance Metrics"),
              p(
                class = "panel-desc",
                "A side-by-side summary of AUC, accuracy, sensitivity, and specificity."
              ),
              div(class = "table-scroll", tableOutput("metrics_table"))
            )
          )
        ),

        # Put ROC curve below the table. This avoids squeezing the table into a narrow column.
        fluidRow(
          column(
            12,
            div(
              class = "panel-card",
              h3("ROC Curves"),
              p(
                class = "panel-desc",
                "The ROC comparison makes it easier to show that the models are close to chance-level discrimination."
              ),
              plotOutput("roc_compare_plot", height = "430px")
            )
          )
        ),

        fluidRow(
          column(
            6,
            div(
              class = "panel-card",
              h3("Logistic Predicted Probabilities"),
              p(
                class = "panel-desc",
                "This histogram shows how the logistic model separates or fails to separate the two actual risk groups."
              ),
              plotOutput("prob_hist", height = "320px")
            )
          ),
          column(
            6,
            div(
              class = "panel-card",
              h3("Top Random Forest Features"),
              p(
                class = "panel-desc",
                "This plot shows a cleaner ranked bar chart for random forest variable importance."
              ),
              plotOutput("rf_importance_plot", height = "320px")
            )
          )
        ),

        fluidRow(
          column(
            12,
            div(
              class = "panel-card",
              h3("Logistic Regression Models We Tested"),
              p(
                class = "panel-desc",
                "We completed 5-fold cross validation for several logistic regression models before choosing the final model."
              ),
              div(class = "table-scroll", tableOutput("cv_table"))
            )
          )
        )
      ),

      nav_panel(
        "Risk Calculator",
        br(),
        div(class = "section-header", div(class = "section-kicker", "Interactive Tool"), h2(class = "section-title", "Risk Calculator"), p(class = "section-note", "This section combines the rule-based score with the model outputs for a selected patient profile.")),
        fluidRow(
          column(
            5,
            div(
              class = "panel-card",
              h3("Enter Patient Profile"),
              div(class = "form-block-title", "Demographics"),
              fluidRow(column(6, selectInput("age", "Age Group", c("Under 40", "40-50", "50-60", "60-70", "70+"))), column(6, selectInput("gender", "Gender", c("Female", "Male")))),
              fluidRow(column(6, selectInput("diabetes", "Diabetes", c("No", "Yes"))), column(6, selectInput("smoking", "Smoking", c("No", "Yes")))),
              fluidRow(column(6, selectInput("obesity", "Obesity", c("No", "Yes"))), column(6, selectInput("family_history", "Family History", c("No", "Yes")))),
              tags$hr(),
              div(class = "form-block-title", "Clinical Markers"),
              fluidRow(column(6, selectInput("cholesterol", "Cholesterol", c("Low", "Med", "High"))), column(6, selectInput("heart_rate", "Heart Rate", c("Low", "Med", "High")))),
              fluidRow(column(6, selectInput("bmi", "BMI", c("Low", "Med", "High"))), column(6, selectInput("triglycerides", "Triglycerides", c("Low", "Med", "High")))),
              fluidRow(column(6, selectInput("blood_sugar", "Blood Sugar", c("Low", "Med", "High"))), column(6, selectInput("ck_mb", "CK-MB", c("Low", "Med", "High")))),
              fluidRow(column(6, selectInput("troponin", "Troponin", c("Low", "Med", "High"))))
            )
          ),
          column(
            7,
            div(
              class = "panel-card",
              h3("Risk Summary"),
              uiOutput("risk_label"),
              tags$hr(),
              fluidRow(
                column(4, div(class = "metric-chip", div(class = "metric-chip-label", "Logistic model output"), div(class = "metric-chip-value", textOutput("log_pred")))),
                column(4, div(class = "metric-chip", div(class = "metric-chip-label", "Random forest output"), div(class = "metric-chip-value", textOutput("rf_pred")))),
                column(4, div(class = "metric-chip", div(class = "metric-chip-label", "XGBoost output"), div(class = "metric-chip-value", textOutput("xgb_pred"))))
              ),
              div(
                class = "insight-box soft-blue",
                style = "margin-top:14px;",
                HTML("<strong>How to read this:</strong><ul style='margin-top:10px; margin-bottom:0; line-height:1.8;'><li>The rule-based score is the most interpretable result here.</li><li>The model outputs are shown only as rough references.</li><li>Low, Med, and High are mapped to real quantiles from the training data.</li><li>This project is for learning and presentation, not for medical use.</li></ul>")
              )
            )
          )
        ),
        fluidRow(
          column(
            12,
            div(
              class = "panel-card",
              h3("Most Similar Patient in Dataset"),
              p(class = "panel-desc", "Based on the selected inputs, the closest matching patient from the dataset is shown below. The similarity score uses both numerical and categorical variables from the calculator."),
              uiOutput("similar_patient")
            )
          )
        )
      )
    ),

    div(class = "footer-note", "Heart Risk Analysis • Final Project Dashboard • Hannah Chen and Kyndal Schlup")
  )
)

# ----------------------------
# Server
# ----------------------------
server <- function(input, output, session) {

  output$vb_n <- renderText({ format(nrow(heart_nomis), big.mark = ",") })
  output$vb_high <- renderText({ paste0(round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "1") * 100, 1), "%") })
  output$vb_low <- renderText({ paste0(round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "0") * 100, 1), "%") })
  output$vb_aucbest <- renderText({ round(max(as.numeric(auc(roc_log)), as.numeric(auc(roc_rf)), as.numeric(auc(roc_xgb))), 3) })

  output$metrics_table <- renderTable(metric_df, striped = TRUE, bordered = FALSE, spacing = "m", width = "100%")
  output$cv_table <- renderTable(cv_results, striped = TRUE, bordered = FALSE, spacing = "m", width = "100%")
  output$summary_table <- renderTable(summary_stats(heart_nomis, input$summary_var), striped = TRUE, bordered = FALSE, spacing = "m", width = "100%")

  output$outcome_plot <- renderPlot({
    ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary., fill = Heart.Attack.Risk..Binary.)) +
      geom_bar(width = 0.62) +
      geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.35, size = 5) +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "#3b82f6", "1" = "#ef4444")) +
      labs(x = NULL, y = "Count") +
      theme_minimal(base_size = 14) +
      theme(legend.position = "none", panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
  })

  output$eda_bar <- renderPlot({
    ggplot(heart_nomis, aes(x = .data[[input$eda_var]], fill = Heart.Attack.Risk..Binary.)) +
      geom_bar(position = "fill", width = 0.68) +
      scale_fill_manual(values = c("0" = "#3b82f6", "1" = "#ef4444"), labels = c("Low Risk", "High Risk")) +
      scale_y_continuous(labels = percent) +
      labs(x = input$eda_var, y = "Proportion", fill = "Risk Group") +
      theme_minimal(base_size = 14) +
      theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
  })

  output$ckmb_plot <- renderPlot({
    ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary., y = CK.MB, fill = Heart.Attack.Risk..Binary.)) +
      geom_boxplot(width = 0.54, alpha = 0.92, outlier.alpha = 0.35) +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "#60a5fa", "1" = "#f87171")) +
      labs(x = "Risk Group", y = "CK-MB") +
      theme_minimal(base_size = 14) +
      theme(legend.position = "none", panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
  })

  output$troponin_plot <- renderPlot({
    ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary., y = Troponin, fill = Heart.Attack.Risk..Binary.)) +
      geom_boxplot(width = 0.54, alpha = 0.92, outlier.alpha = 0.35) +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "#60a5fa", "1" = "#f87171")) +
      labs(x = "Risk Group", y = "Troponin") +
      theme_minimal(base_size = 14) +
      theme(legend.position = "none", panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
  })

  output$roc_compare_plot <- renderPlot({
    plot(roc_log, col = "#2563eb", lwd = 3, main = "", legacy.axes = TRUE)
    plot(roc_rf, col = "#10b981", lwd = 3, add = TRUE)
    plot(roc_xgb, col = "#7c3aed", lwd = 3, add = TRUE)
    abline(a = 0, b = 1, lty = 2, col = "gray65")
    legend(
      "bottomright",
      legend = c(
        paste("Logistic AUC =", round(as.numeric(auc(roc_log)), 3)),
        paste("Random Forest AUC =", round(as.numeric(auc(roc_rf)), 3)),
        paste("XGBoost AUC =", round(as.numeric(auc(roc_xgb)), 3))
      ),
      col = c("#2563eb", "#10b981", "#7c3aed"),
      lwd = 3,
      bty = "n",
      cex = 1
    )
  })

  output$prob_hist <- renderPlot({
    ggplot(test_data, aes(x = log_prob, fill = Heart.Attack.Risk..Binary.)) +
      geom_histogram(binwidth = 0.02, alpha = 0.72, position = "identity") +
      geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = 1, color = "#64748b") +
      scale_fill_manual(values = c("0" = "#3b82f6", "1" = "#ef4444"), labels = c("Low Risk", "High Risk")) +
      labs(x = "Predicted Probability of High Risk", y = "Count", fill = "Actual Risk") +
      theme_minimal(base_size = 14) +
      theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank())
  })

  output$rf_importance_plot <- renderPlot({
    ggplot(rf_imp, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
      geom_col(fill = "#2563eb", width = 0.72) +
      coord_flip() +
      labs(x = NULL, y = "Mean Decrease in Gini") +
      theme_minimal(base_size = 14) +
      theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank())
  })

  new_patient <- reactive({
    data.frame(
      Age = age_group_map[[input$age]],
      Gender = factor(input$gender, levels = levels(train_data$Gender)),
      Diabetes = factor(yes_no_to_factor(input$diabetes), levels = c("0", "1")),
      Smoking = factor(yes_no_to_factor(input$smoking), levels = c("0", "1")),
      Obesity = factor(yes_no_to_factor(input$obesity), levels = c("0", "1")),
      Family.History = factor(yes_no_to_factor(input$family_history), levels = c("0", "1")),
      Cholesterol = map_to_quantile(input$cholesterol, "Cholesterol"),
      Heart.rate = map_to_quantile(input$heart_rate, "Heart.rate"),
      BMI = map_to_quantile(input$bmi, "BMI"),
      Triglycerides = map_to_quantile(input$triglycerides, "Triglycerides"),
      Blood.sugar = map_to_quantile(input$blood_sugar, "Blood.sugar"),
      CK.MB = map_to_quantile(input$ck_mb, "CK.MB"),
      Troponin = map_to_quantile(input$troponin, "Troponin")
    )
  })

  output$risk_label <- renderUI({
    s <- rule_score(
      input$age,
      input$cholesterol,
      input$heart_rate,
      input$bmi,
      input$triglycerides,
      input$blood_sugar,
      input$ck_mb,
      input$troponin,
      input$smoking,
      input$diabetes,
      input$family_history,
      input$obesity
    )

    max_score <- 12.2
    index <- s / max_score

    color <- if (index < 0.33) "#10b981" else if (index < 0.66) "#f59e0b" else "#ef4444"
    label <- if (index < 0.33) "Low Risk" else if (index < 0.66) "Moderate Risk" else "High Risk"
    pct <- round(index * 100)

    div(
      style = "padding: 6px 0 10px 0;",
      div(style = paste0("font-size: 2.35rem; font-weight: 900; color:", color, "; margin-bottom: 6px;"), label),
      div(style = "color:#64748b; font-size:1rem;", paste0("Rule-based score: ", round(index, 2), " / 1.0")),
      div(class = "risk-meter-bg", div(style = paste0("height:100%; width:", pct, "%; background:", color, "; border-radius:999px;")))
    )
  })

  output$log_pred <- renderText({
    prob <- predict(log_model_calculator, new_patient(), type = "response")
    paste0(round(prob * 100, 1), "%")
  })

  output$rf_pred <- renderText({
    prob <- predict(rf_model_calculator, new_patient(), type = "prob")[, "1"]
    paste0(round(prob * 100, 1), "%")
  })

  output$xgb_pred <- renderText({
    pat <- new_patient()
    pat_with_y <- cbind(pat, Heart.Attack.Risk..Binary. = factor("0", levels = c("0", "1")))
    xmat <- model.matrix(model_formula, data = pat_with_y)[, -1, drop = FALSE]
    xmat <- as.matrix(xmat)
    prob <- predict(xgb_model_calculator, newdata = xgb.DMatrix(data = xmat))
    paste0(round(prob * 100, 1), "%")
  })

  output$similar_patient <- renderUI({
    pat <- new_patient()
    result <- find_similar_patient(pat, heart_nomis)
    sim_pat <- result$patient
    sim_score <- result$similarity

    risk_label <- ifelse(sim_pat$Heart.Attack.Risk..Binary. == "1", "High Risk", "Low Risk")
    risk_color <- ifelse(sim_pat$Heart.Attack.Risk..Binary. == "1", "#ef4444", "#10b981")

    div(
      div(
        style = "display:flex; align-items:center; gap:12px; margin-bottom:14px;",
        div(style = "font-size:2rem; font-weight:900; color:#2563eb;", paste0(sim_score, "%")),
        div(style = "color:#64748b; font-size:0.95rem;", "similarity score")
      ),
      div(class = "metric-chip", div(class = "metric-chip-label", "Matched patient outcome"), div(class = "metric-chip-value", style = paste0("color:", risk_color, ";"), risk_label)),
      fluidRow(
        column(3, div(class = "metric-chip", div(class = "metric-chip-label", "Age (scaled)"), div(class = "metric-chip-value", round(sim_pat$Age, 3)))),
        column(3, div(class = "metric-chip", div(class = "metric-chip-label", "Cholesterol"), div(class = "metric-chip-value", round(sim_pat$Cholesterol, 3)))),
        column(3, div(class = "metric-chip", div(class = "metric-chip-label", "Troponin"), div(class = "metric-chip-value", round(sim_pat$Troponin, 3)))),
        column(3, div(class = "metric-chip", div(class = "metric-chip-label", "CK-MB"), div(class = "metric-chip-value", round(sim_pat$CK.MB, 3))))
      ),
      fluidRow(
        column(3, div(class = "metric-chip", div(class = "metric-chip-label", "BMI"), div(class = "metric-chip-value", round(sim_pat$BMI, 3)))),
        column(3, div(class = "metric-chip", div(class = "metric-chip-label", "Diabetes"), div(class = "metric-chip-value", ifelse(sim_pat$Diabetes == "1", "Yes", "No")))),
        column(3, div(class = "metric-chip", div(class = "metric-chip-label", "Smoking"), div(class = "metric-chip-value", ifelse(sim_pat$Smoking == "1", "Yes", "No")))),
        column(3, div(class = "metric-chip", div(class = "metric-chip-label", "Obesity"), div(class = "metric-chip-value", ifelse(sim_pat$Obesity == "1", "Yes", "No"))))
      )
    )
  })
}

# ----------------------------
# Run app
# ----------------------------
shinyApp(ui = ui, server = server)

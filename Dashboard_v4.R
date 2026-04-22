# ============================================================
# Heart Risk Explorer
# Hannah Chen — Final Project Dashboard
# ============================================================

library(shiny)
library(bslib)
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(randomForest)

# ----------------------------
# 1. Load and clean data
# ----------------------------
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")
heart_nomis <- na.omit(heart)

factor_vars <- c(
  "Diabetes",
  "Family.History",
  "Smoking",
  "Obesity",
  "Heart.Attack.Risk..Binary.",
  "Gender"
)
heart_nomis$Alcohol.Consumption <- factor(heart_nomis$Alcohol.Consumption, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Diet <- as.factor(heart_nomis$Diet)
heart_nomis$Previous.Heart.Problems <- factor(heart_nomis$Previous.Heart.Problems, levels = c(0, 1), labels = c("0", "1"))
heart_nomis$Medication.Use <- factor(heart_nomis$Medication.Use, levels = c(0, 1), labels = c("0", "1"))
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

# ----------------------------
# 2. Train / test split
# ----------------------------
set.seed(123)
train_index <- createDataPartition(
  heart_nomis$Heart.Attack.Risk..Binary.,
  p = 0.7,
  list = FALSE
)

train_data <- heart_nomis[train_index, ]
test_data <- heart_nomis[-train_index, ]

# ----------------------------
# 3. Model formula
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

model_formula <- Heart.Attack.Risk..Binary. ~
  Age +
  Gender +
  Diabetes +
  Smoking +
  Obesity +
  Family.History +
  Cholesterol +
  Heart.rate +
  BMI +
  Triglycerides +
  Blood.sugar +
  CK.MB +
  Troponin
# ----------------------------
# 4. Fit models
# ----------------------------
full_log_model <- glm(full_formula, data = train_data, family = binomial)
log_model_full <- stepAIC(full_log_model, direction = "both", trace = FALSE)

set.seed(123)
rf_model_full <- randomForest(
  full_formula,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

log_model <- glm(model_formula, data = train_data, family = binomial)

set.seed(123)
rf_model <- randomForest(
  model_formula,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# ----------------------------
# 5. Evaluate models
# ----------------------------
test_data$log_prob <- predict(log_model_full, newdata = test_data, type = "response")
test_data$log_class <- factor(
  ifelse(test_data$log_prob > 0.5, "1", "0"),
  levels = c("0", "1")
)

cm_log <- confusionMatrix(
  test_data$log_class,
  test_data$Heart.Attack.Risk..Binary.,
  positive = "1"
)

roc_log <- roc(
  response = test_data$Heart.Attack.Risk..Binary.,
  predictor = test_data$log_prob,
  levels = c("0", "1")
)

rf_pred <- predict(rf_model_full, newdata = test_data)
rf_prob <- predict(rf_model_full, newdata = test_data, type = "prob")[, "1"]

cm_rf <- confusionMatrix(
  rf_pred,
  test_data$Heart.Attack.Risk..Binary.,
  positive = "1"
)

roc_rf <- roc(
  response = test_data$Heart.Attack.Risk..Binary.,
  predictor = rf_prob,
  levels = c("0", "1")
)

# ----------------------------
# 6. Helper functions
# ----------------------------
to_numeric <- function(x) {
  switch(x, "Low" = 0.2, "Med" = 0.5, "High" = 0.8, 0.5)
}

age_to_num <- function(x) {
  switch(
    x,
    "Under 40" = 0.2,
    "40-50" = 0.35,
    "50-60" = 0.5,
    "60-70" = 0.65,
    "70+" = 0.85,
    0.5
  )
}

yes_no_to_factor <- function(x) {
  ifelse(x == "Yes", "1", "0")
}

rule_score <- function(age, chol, hr, bmi, trig, bs, ckmb, trop) {
  s <- 0
  if (age %in% c("60-70", "70+")) {
    s <- s + 1
  }
  s <- s + to_numeric(chol)
  s <- s + to_numeric(hr)
  s <- s + to_numeric(bmi)
  s <- s + to_numeric(trig)
  s <- s + to_numeric(bs)
  s <- s + 2 * to_numeric(ckmb)
  s <- s + 3 * to_numeric(trop)
  s
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
      RiskGroup = ifelse(
        Heart.Attack.Risk..Binary. == "0",
        "Low Risk",
        "High Risk"
      )
    ) %>%
    dplyr::select(RiskGroup, Mean, Median, SD, Min, Max)
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

rf_imp <- as.data.frame(importance(rf_model_full))
rf_imp$Feature <- rownames(rf_imp)
rf_imp <- rf_imp %>%
  arrange(desc(MeanDecreaseGini)) %>%
  slice(1:10)

# ----------------------------
# 7. Theme
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
# 8. UI
# ----------------------------
ui <- fluidPage(
  theme = theme_modern,

  tags$head(
    tags$style(HTML(
      "
      body {
        background: #f8fafc;
      }

      .page-wrap {
        max-width: 1320px;
        margin: 0 auto;
        padding: 28px 20px 40px 20px;
      }

      .hero {
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 55%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 28px;
        padding: 34px;
        margin-bottom: 26px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
      }

      .eyebrow {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #2563eb;
        background: #dbeafe;
        border-radius: 999px;
        padding: 6px 12px;
        margin-bottom: 14px;
      }

      .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        line-height: 1.05;
        color: #0f172a;
        margin-bottom: 10px;
      }

      .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #475569;
        max-width: 760px;
        margin-bottom: 0;
      }

      .hero-side {
        background: rgba(255,255,255,0.9);
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 22px;
        height: 100%;
      }

      .hero-side h4 {
        font-weight: 800;
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 14px;
        color: #0f172a;
      }

      .hero-side ul {
        padding-left: 18px;
        margin-bottom: 0;
        color: #475569;
        line-height: 1.7;
      }

      .section-block {
        margin-bottom: 26px;
      }

      .section-header {
        margin: 6px 0 14px 0;
      }

      .section-kicker {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 5px;
      }

      .section-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
      }

      .section-note {
        color: #64748b;
        margin-top: 6px;
        margin-bottom: 0;
        font-size: 1rem;
      }

      .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 22px 22px 20px 22px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        margin-bottom: 18px;
        height: 100%;
      }

      .stat-label {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 10px;
      }

      .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
        margin-bottom: 8px;
      }

      .stat-subtext {
        color: #475569;
        font-size: 0.95rem;
      }

      .panel-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
      }

      .panel-card h3 {
        font-size: 1.1rem;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 8px;
        color: #0f172a;
      }

      .panel-desc {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 14px;
      }

      .insight-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 20px;
        padding: 18px;
        color: #7c2d12;
      }

      .insight-box strong {
        color: #7c2d12;
      }

      .soft-blue {
        background: #eff6ff;
        border-color: #bfdbfe;
      }

      .soft-slate {
        background: #f8fafc;
      }

      .mini-badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
      }

      .badge-blue {
        background: #dbeafe;
        color: #1d4ed8;
      }

      .badge-amber {
        background: #fef3c7;
        color: #b45309;
      }

      .badge-green {
        background: #dcfce7;
        color: #15803d;
      }

      .risk-meter-bg {
        height: 14px;
        width: 100%;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 14px;
      }

      .metric-chip {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
      }

      .metric-chip-label {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 6px;
      }

      .metric-chip-value {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
      }

      .shiny-input-container {
        width: 100% !important;
      }

      .form-block-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: #334155;
        margin-top: 8px;
        margin-bottom: 10px;
      }

      .table {
        background: white;
      }

      .footer-note {
        color: #64748b;
        font-size: 0.92rem;
        text-align: center;
        margin-top: 10px;
      }
    "
    ))
  ),

  div(
    class = "page-wrap",

    # HERO
    div(
      class = "hero",
      fluidRow(
        column(
          8,
          div(class = "eyebrow", "Statistics Course Project"),
          div(class = "hero-title", "Heart Risk Explorer"),
          p(
            class = "hero-subtitle",
            "An interactive dashboard for exploring clinical markers, model performance, and a more interpretable rule-based risk score. This version is designed to feel closer to a modern web page rather than a basic classroom dashboard."
          ),
          div(
            style = "margin-top:16px;",
            span(class = "mini-badge badge-blue", "Interactive"),
            span(class = "mini-badge badge-amber", "Synthetic Dataset"),
            span(class = "mini-badge badge-green", "Interpretability Focus")
          )
        ),
        column(
          4,
          div(
            class = "hero-side",
            h4("Key takeaways"),
            tags$ul(
              tags$li(
                "Both machine-learning models perform close to chance level on this dataset."
              ),
              tags$li(
                "Troponin and CK-MB still stand out as clinically meaningful markers."
              ),
              tags$li(
                "The rule-based score is the most interpretable part of this project."
              ),
              tags$li(
                "This dashboard is for learning and presentation, not real clinical use."
              )
            )
          )
        )
      )
    ),

    # KPI
    div(
      class = "section-block",
      div(
        class = "section-header",
        div(class = "section-kicker", "Snapshot"),
        h2(class = "section-title", "What the data looks like"),
        p(
          class = "section-note",
          "A quick overview of the cleaned dataset used in the analysis."
        )
      ),
      fluidRow(
        column(
          3,
          div(
            class = "stat-card",
            div(class = "stat-label", "Complete observations"),
            div(class = "stat-value", textOutput("vb_n")),
            div(
              class = "stat-subtext",
              "Rows remaining after removing missing values"
            )
          )
        ),
        column(
          3,
          div(
            class = "stat-card",
            div(class = "stat-label", "High-risk cases"),
            div(class = "stat-value", textOutput("vb_high")),
            div(
              class = "stat-subtext",
              "Share of observations labeled high risk"
            )
          )
        ),
        column(
          3,
          div(
            class = "stat-card",
            div(class = "stat-label", "Low-risk cases"),
            div(class = "stat-value", textOutput("vb_low")),
            div(
              class = "stat-subtext",
              "Share of observations labeled low risk"
            )
          )
        ),
        column(
          3,
          div(
            class = "stat-card",
            div(class = "stat-label", "Best model AUC"),
            div(class = "stat-value", textOutput("vb_aucbest")),
            div(
              class = "stat-subtext",
              "Highest AUC among the two fitted models"
            )
          )
        )
      )
    ),

    # OVERVIEW + DISTRIBUTION
    div(
      class = "section-block",
      fluidRow(
        column(
          7,
          div(
            class = "panel-card",
            h3("Project framing"),
            p(
              class = "panel-desc",
              HTML(
                "<strong>Both logistic regression and random forest performed near chance level</strong>, with AUC values close to 0.5. Because of that, this project emphasizes interpretation and clinical reasoning rather than claiming strong predictive power."
              )
            ),
            div(
              class = "insight-box",
              HTML(
                "<strong>⚠️ Important:</strong> This dashboard should be interpreted as a course-project analysis, not a real clinical decision tool. The rule-based score is included to provide a clearer and more explainable summary of known cardiac risk markers."
              )
            )
          )
        ),
        column(
          5,
          div(
            class = "panel-card soft-blue",
            h3("What we did instead"),
            p(
              class = "panel-desc",
              "Instead of relying only on weak model predictions, we created a rule-based scoring system that gives additional weight to clinically important biomarkers."
            ),
            tags$ul(
              style = "color:#334155; line-height:1.8; margin-bottom:0;",
              tags$li("Troponin receives the strongest weight"),
              tags$li("CK-MB also contributes more heavily"),
              tags$li("Other markers provide supporting evidence"),
              tags$li("The result is easier to explain during presentation")
            )
          )
        )
      ),
      fluidRow(
        column(
          7,
          div(
            class = "panel-card",
            h3("Outcome distribution"),
            p(
              class = "panel-desc",
              "This chart shows the balance between low-risk and high-risk observations in the cleaned dataset."
            ),
            plotOutput("outcome_plot", height = "320px")
          )
        ),
        column(
          5,
          div(
            class = "panel-card soft-slate",
            h3("Dataset summary"),
            div(
              class = "metric-chip",
              div(class = "metric-chip-label", "Complete observations"),
              div(class = "metric-chip-value", textOutput("sum_complete"))
            ),
            div(
              class = "metric-chip",
              div(class = "metric-chip-label", "Training rows"),
              div(class = "metric-chip-value", textOutput("sum_train"))
            ),
            div(
              class = "metric-chip",
              div(class = "metric-chip-label", "Test rows"),
              div(class = "metric-chip-value", textOutput("sum_test"))
            ),
            div(
              class = "metric-chip",
              div(class = "metric-chip-label", "High-risk count"),
              div(class = "metric-chip-value", textOutput("sum_high_n"))
            ),
            div(
              class = "metric-chip",
              div(class = "metric-chip-label", "Low-risk count"),
              div(class = "metric-chip-value", textOutput("sum_low_n"))
            )
          )
        )
      )
    ),

    # CLINICAL SIGNALS
    div(
      class = "section-block",
      div(
        class = "section-header",
        div(class = "section-kicker", "Exploration"),
        h2(class = "section-title", "Clinical signals"),
        p(
          class = "section-note",
          "Use the controls below to explore how selected variables relate to risk groups."
        )
      ),
      fluidRow(
        column(
          12,
          div(
            class = "panel-card",
            h3("Explore a categorical variable"),
            p(
              class = "panel-desc",
              "Switch the selected variable to compare how the low-risk and high-risk groups are distributed."
            ),
            fluidRow(
              column(
                4,
                selectInput(
                  "eda_var",
                  "Choose a variable",
                  choices = c(
                    "Gender",
                    "Smoking",
                    "Diabetes",
                    "Obesity",
                    "Family.History"
                  )
                )
              )
            ),
            plotOutput("eda_bar", height = "320px")
          )
        )
      ),
      fluidRow(
        column(
          6,
          div(
            class = "panel-card",
            h3("Risk by gender"),
            p(
              class = "panel-desc",
              "This view gives a quick proportional comparison by gender."
            ),
            plotOutput("gender_plot", height = "320px")
          )
        ),
        column(
          6,
          div(
            class = "panel-card",
            h3("Troponin by risk group"),
            p(
              class = "panel-desc",
              "Troponin is one of the most clinically important markers in this project, so it is highlighted separately."
            ),
            plotOutput("troponin_plot", height = "320px")
          )
        )
      ),
      fluidRow(
        column(
          12,
          div(
            class = "panel-card",
            h3("Marker summary table"),
            p(
              class = "panel-desc",
              "Choose a marker to compare its descriptive statistics across the two risk groups."
            ),
            fluidRow(
              column(
                4,
                selectInput(
                  "summary_var",
                  "Choose a marker",
                  choices = c("Troponin", "CK.MB", "Cholesterol", "Heart.rate"),
                  selected = "Troponin"
                )
              )
            ),
            tableOutput("summary_table")
          )
        )
      )
    ),

    # MODEL PERFORMANCE
    div(
      class = "section-block",
      div(
        class = "section-header",
        div(class = "section-kicker", "Models"),
        h2(class = "section-title", "How the models performed"),
        p(
          class = "section-note",
          "These results are shown transparently, even though the models do not perform strongly on this dataset."
        )
      ),
      fluidRow(
        column(
          4,
          div(
            class = "panel-card",
            h3("Performance metrics"),
            p(
              class = "panel-desc",
              "A side-by-side summary of AUC, accuracy, sensitivity, and specificity."
            ),
            tableOutput("metrics_table")
          )
        ),
        column(
          8,
          div(
            class = "panel-card",
            h3("ROC curves"),
            p(
              class = "panel-desc",
              "The ROC comparison makes it easier to show that both models are close to chance-level discrimination."
            ),
            plotOutput("roc_compare_plot", height = "340px")
          )
        )
      ),
      fluidRow(
        column(
          6,
          div(
            class = "panel-card",
            h3("Logistic predicted probabilities"),
            p(
              class = "panel-desc",
              "This histogram shows how the logistic model separates—or fails to separate—the two actual risk groups."
            ),
            plotOutput("prob_hist", height = "320px")
          )
        ),
        column(
          6,
          div(
            class = "panel-card",
            h3("Top random forest features"),
            p(
              class = "panel-desc",
              "Instead of using the default base R importance plot, this version uses a cleaner ranked bar chart."
            ),
            plotOutput("rf_importance_plot", height = "320px")
          )
        )
      )
    ),

    # CALCULATOR
    div(
      class = "section-block",
      div(
        class = "section-header",
        div(class = "section-kicker", "Interactive tool"),
        h2(class = "section-title", "Risk calculator"),
        p(
          class = "section-note",
          "This section combines the rule-based score with the model outputs for a selected patient profile."
        )
      ),
      fluidRow(
        column(
          6,
          div(
            class = "panel-card",
            h3("Enter patient profile"),

            div(class = "form-block-title", "Demographics"),
            fluidRow(
              column(
                6,
                selectInput(
                  "age",
                  "Age Group",
                  c("Under 40", "40-50", "50-60", "60-70", "70+")
                )
              ),
              column(6, selectInput("gender", "Gender", c("Female", "Male")))
            ),
            fluidRow(
              column(6, selectInput("diabetes", "Diabetes", c("No", "Yes"))),
              column(6, selectInput("smoking", "Smoking", c("No", "Yes")))
            ),
            fluidRow(
              column(6, selectInput("obesity", "Obesity", c("No", "Yes"))),
              column(
                6,
                selectInput("family_history", "Family History", c("No", "Yes"))
              )
            ),

            tags$hr(),

            div(class = "form-block-title", "Clinical markers"),
            fluidRow(
              column(
                6,
                selectInput(
                  "cholesterol",
                  "Cholesterol",
                  c("Low", "Med", "High")
                )
              ),
              column(
                6,
                selectInput("heart_rate", "Heart Rate", c("Low", "Med", "High"))
              )
            ),
            fluidRow(
              column(6, selectInput("bmi", "BMI", c("Low", "Med", "High"))),
              column(
                6,
                selectInput(
                  "triglycerides",
                  "Triglycerides",
                  c("Low", "Med", "High")
                )
              )
            ),
            fluidRow(
              column(
                6,
                selectInput(
                  "blood_sugar",
                  "Blood Sugar",
                  c("Low", "Med", "High")
                )
              ),
              column(6, selectInput("ck_mb", "CK-MB", c("Low", "Med", "High")))
            ),
            fluidRow(
              column(
                6,
                selectInput("troponin", "Troponin", c("Low", "Med", "High"))
              )
            )
          )
        ),
        column(
          6,
          div(
            class = "panel-card",
            h3("Risk summary"),
            uiOutput("risk_label"),
            tags$hr(),
            div(
              class = "metric-chip",
              div(class = "metric-chip-label", "Logistic model output"),
              div(class = "metric-chip-value", textOutput("log_pred"))
            ),
            div(
              class = "metric-chip",
              div(class = "metric-chip-label", "Random forest output"),
              div(class = "metric-chip-value", textOutput("rf_pred"))
            ),
            div(
              class = "insight-box soft-blue",
              style = "margin-top:14px;",
              HTML(
                "
                <strong>How to read this:</strong>
                <ul style='margin-top:10px; margin-bottom:0; line-height:1.8;'>
                  <li>The rule-based score is the most interpretable result here.</li>
                  <li>The model outputs are shown only as rough references.</li>
                  <li>This project is for learning and presentation, not for medical use.</li>
                </ul>
              "
              )
            )
          )
        )
      )
    ),

    div(
      class = "footer-note",
      "Heart Risk Explorer • Final Project Dashboard • Hannah Chen"
    )
  )
)

# ----------------------------
# 9. Server
# ----------------------------
server <- function(input, output, session) {
  output$vb_n <- renderText({
    format(nrow(heart_nomis), big.mark = ",")
  })

  output$vb_high <- renderText({
    paste0(
      round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "1") * 100, 1),
      "%"
    )
  })

  output$vb_low <- renderText({
    paste0(
      round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "0") * 100, 1),
      "%"
    )
  })

  output$vb_aucbest <- renderText({
    round(max(as.numeric(auc(roc_log)), as.numeric(auc(roc_rf))), 3)
  })

  output$sum_complete <- renderText({
    format(nrow(heart_nomis), big.mark = ",")
  })

  output$sum_train <- renderText({
    format(nrow(train_data), big.mark = ",")
  })

  output$sum_test <- renderText({
    format(nrow(test_data), big.mark = ",")
  })

  output$sum_high_n <- renderText({
    format(sum(heart_nomis$Heart.Attack.Risk..Binary. == "1"), big.mark = ",")
  })

  output$sum_low_n <- renderText({
    format(sum(heart_nomis$Heart.Attack.Risk..Binary. == "0"), big.mark = ",")
  })

  output$metrics_table <- renderTable(
    {
      metric_df
    },
    striped = TRUE,
    bordered = FALSE,
    spacing = "m",
    width = "100%"
  )

  output$summary_table <- renderTable(
    {
      summary_stats(heart_nomis, input$summary_var)
    },
    striped = TRUE,
    bordered = FALSE,
    spacing = "m",
    width = "100%"
  )

  output$outcome_plot <- renderPlot({
    ggplot(
      heart_nomis,
      aes(x = Heart.Attack.Risk..Binary., fill = Heart.Attack.Risk..Binary.)
    ) +
      geom_bar(width = 0.62) +
      geom_text(
        stat = "count",
        aes(label = after_stat(count)),
        vjust = -0.35,
        size = 5
      ) +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "#3b82f6", "1" = "#ef4444")) +
      labs(x = NULL, y = "Count") +
      theme_minimal(base_size = 14) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        axis.title.x = element_blank(),
        plot.margin = ggplot2::margin(10, 14, 10, 4)
      )
  })

  output$eda_bar <- renderPlot({
    ggplot(
      heart_nomis,
      aes(x = .data[[input$eda_var]], fill = Heart.Attack.Risk..Binary.)
    ) +
      geom_bar(position = "fill", width = 0.68) +
      scale_fill_manual(
        values = c("0" = "#3b82f6", "1" = "#ef4444"),
        labels = c("Low Risk", "High Risk")
      ) +
      scale_y_continuous(labels = scales::percent) +
      labs(x = input$eda_var, y = "Proportion", fill = "Risk Group") +
      theme_minimal(base_size = 14) +
      theme(
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank()
      )
  })

  output$gender_plot <- renderPlot({
    ggplot(
      heart_nomis,
      aes(x = Gender, fill = Heart.Attack.Risk..Binary.)
    ) +
      geom_bar(position = "fill", width = 0.68) +
      scale_fill_manual(
        values = c("0" = "#3b82f6", "1" = "#ef4444"),
        labels = c("Low Risk", "High Risk")
      ) +
      scale_y_continuous(labels = scales::percent) +
      labs(x = "Gender", y = "Proportion", fill = "Risk Group") +
      theme_minimal(base_size = 14) +
      theme(
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank()
      )
  })

  output$troponin_plot <- renderPlot({
    ggplot(
      heart_nomis,
      aes(
        x = Heart.Attack.Risk..Binary.,
        y = Troponin,
        fill = Heart.Attack.Risk..Binary.
      )
    ) +
      geom_boxplot(width = 0.54, alpha = 0.92, outlier.alpha = 0.35) +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "#60a5fa", "1" = "#f87171")) +
      labs(x = "Risk Group", y = "Troponin") +
      theme_minimal(base_size = 14) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank()
      )
  })

  output$roc_compare_plot <- renderPlot({
    plot(
      roc_log,
      col = "#2563eb",
      lwd = 3,
      main = "",
      legacy.axes = TRUE
    )
    plot(roc_rf, col = "#10b981", lwd = 3, add = TRUE)
    abline(a = 0, b = 1, lty = 2, col = "gray65")
    legend(
      "bottomright",
      legend = c(
        paste("Logistic AUC =", round(as.numeric(auc(roc_log)), 3)),
        paste("Random Forest AUC =", round(as.numeric(auc(roc_rf)), 3))
      ),
      col = c("#2563eb", "#10b981"),
      lwd = 3,
      bty = "n",
      cex = 1
    )
  })

  output$prob_hist <- renderPlot({
    ggplot(test_data, aes(x = log_prob, fill = Heart.Attack.Risk..Binary.)) +
      geom_histogram(binwidth = 0.02, alpha = 0.72, position = "identity") +
      geom_vline(
        xintercept = 0.5,
        linetype = "dashed",
        linewidth = 1,
        color = "#64748b"
      ) +
      scale_fill_manual(
        values = c("0" = "#3b82f6", "1" = "#ef4444"),
        labels = c("Low Risk", "High Risk")
      ) +
      labs(
        x = "Predicted Probability of High Risk",
        y = "Count",
        fill = "Actual Risk"
      ) +
      theme_minimal(base_size = 14) +
      theme(
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank()
      )
  })

  output$rf_importance_plot <- renderPlot({
    ggplot(
      rf_imp,
      aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)
    ) +
      geom_col(fill = "#2563eb", width = 0.72) +
      coord_flip() +
      labs(
        x = NULL,
        y = "Mean Decrease in Gini"
      ) +
      theme_minimal(base_size = 14) +
      theme(
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank()
      )
  })

  new_patient <- reactive({
    data.frame(
      Age = age_to_num(input$age),
      Gender = factor(input$gender, levels = levels(train_data$Gender)),
      Diabetes = factor(yes_no_to_factor(input$diabetes), levels = c("0", "1")),
      Smoking = factor(yes_no_to_factor(input$smoking), levels = c("0", "1")),
      Obesity = factor(yes_no_to_factor(input$obesity), levels = c("0", "1")),
      Family.History = factor(
        yes_no_to_factor(input$family_history),
        levels = c("0", "1")
      ),
      Cholesterol = to_numeric(input$cholesterol),
      Heart.rate = to_numeric(input$heart_rate),
      BMI = to_numeric(input$bmi),
      Triglycerides = to_numeric(input$triglycerides),
      Blood.sugar = to_numeric(input$blood_sugar),
      CK.MB = to_numeric(input$ck_mb),
      Troponin = to_numeric(input$troponin)
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
      input$troponin
    )

    index <- s / 9
    color <- if (index < 0.33) {
      "#10b981"
    } else if (index < 0.66) {
      "#f59e0b"
    } else {
      "#ef4444"
    }
    label <- if (index < 0.33) {
      "Low Risk"
    } else if (index < 0.66) {
      "Moderate Risk"
    } else {
      "High Risk"
    }
    pct <- round(index * 100)

    div(
      style = "padding: 6px 0 10px 0;",
      div(
        style = paste0(
          "font-size: 2.35rem; font-weight: 800; color:",
          color,
          "; margin-bottom: 6px;"
        ),
        label
      ),
      div(
        style = "color:#64748b; font-size:1rem;",
        paste0("Rule-based score: ", round(index, 2), " / 1.0")
      ),
      div(
        class = "risk-meter-bg",
        div(
          style = paste0(
            "height:100%; width:",
            pct,
            "%; background:",
            color,
            "; border-radius:999px;"
          )
        )
      )
    )
  })

  output$log_pred <- renderText({
    prob <- predict(log_model, new_patient(), type = "response")
    paste0(round(prob * 100, 1), "%")
  })

  output$rf_pred <- renderText({
    prob <- predict(rf_model, new_patient(), type = "prob")[, "1"]
    paste0(round(prob * 100, 1), "%")
  })
}

# ----------------------------
# 10. Run app
# ----------------------------
shinyApp(ui = ui, server = server)

# ============================================================
# Heart Attack Risk Dashboard (Stable Replacement Version)
# Hannah Chen — Final Project
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
  "Diabetes", "Family.History", "Smoking", "Obesity",
  "Heart.Attack.Risk..Binary.", "Gender"
)
heart_nomis[factor_vars] <- lapply(heart_nomis[factor_vars], as.factor)

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
test_data  <- heart_nomis[-train_index, ]

# ----------------------------
# 3. Model formula
# ----------------------------
model_formula <- Heart.Attack.Risk..Binary. ~
  Age + Gender + Diabetes + Smoking + Obesity + Family.History +
  Cholesterol + Heart.rate + BMI + Triglycerides +
  Blood.sugar + CK.MB + Troponin

# ----------------------------
# 4. Fit models
# ----------------------------
log_model <- glm(model_formula, data = train_data, family = binomial)

set.seed(123)
rf_model <- randomForest(
  model_formula,
  data = train_data,
  ntree = 200,
  importance = TRUE
)

# ----------------------------
# 5. Evaluate models
# ----------------------------
test_data$log_prob <- predict(log_model, newdata = test_data, type = "response")
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
  response  = test_data$Heart.Attack.Risk..Binary.,
  predictor = test_data$log_prob,
  levels    = c("0", "1")
)

rf_pred <- predict(rf_model, newdata = test_data)
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "1"]

cm_rf <- confusionMatrix(
  rf_pred,
  test_data$Heart.Attack.Risk..Binary.,
  positive = "1"
)

roc_rf <- roc(
  response  = test_data$Heart.Attack.Risk..Binary.,
  predictor = rf_prob,
  levels    = c("0", "1")
)

# ----------------------------
# 6. Helper functions
# ----------------------------
to_numeric <- function(x) {
  switch(x,
         "Low" = 0.2,
         "Med" = 0.5,
         "High" = 0.8,
         0.5)
}

age_to_num <- function(x) {
  switch(x,
         "Under 40" = 0.2,
         "40-50"    = 0.35,
         "50-60"    = 0.5,
         "60-70"    = 0.65,
         "70+"      = 0.85,
         0.5)
}

yes_no_to_factor <- function(x) {
  ifelse(x == "Yes", "1", "0")
}

rule_score <- function(age, chol, hr, bmi, trig, bs, ckmb, trop) {
  s <- 0
  if (age %in% c("60-70", "70+")) s <- s + 1
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
      Mean   = round(mean(.data[[var]], na.rm = TRUE), 3),
      Median = round(median(.data[[var]], na.rm = TRUE), 3),
      SD     = round(sd(.data[[var]], na.rm = TRUE), 3),
      Min    = round(min(.data[[var]], na.rm = TRUE), 3),
      Max    = round(max(.data[[var]], na.rm = TRUE), 3),
      .groups = "drop"
    ) %>%
    mutate(RiskGroup = ifelse(Heart.Attack.Risk..Binary. == "0", "Low Risk", "High Risk")) %>%
    select(RiskGroup, Mean, Median, SD, Min, Max)
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

dataset_summary_df <- data.frame(
  Item = c("Complete observations", "Training rows", "Test rows"),
  Value = c(nrow(heart_nomis), nrow(train_data), nrow(test_data))
)

tab <- table(heart_nomis$Heart.Attack.Risk..Binary.)
prop <- prop.table(tab)

outcome_summary_df <- data.frame(
  RiskGroup = c("Low Risk", "High Risk"),
  Count = as.numeric(tab),
  Proportion = round(as.numeric(prop) * 100, 1)
)

# ----------------------------
# 7. Theme
# ----------------------------
theme_modern <- bs_theme(
  version = 5,
  bootswatch = "flatly",
  primary = "#1f4e79",
  bg = "#f6f8fb",
  fg = "#1f2937"
)

# ----------------------------
# 8. UI
# ----------------------------
ui <- fluidPage(
  theme = theme_modern,

  tags$head(
    tags$style(HTML("
      body {
        background-color: #f6f8fb;
      }

      .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #0f172a;
      }

      .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.2rem;
      }

      .card-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
      }

      .kpi-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
      }

      .kpi-label {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 8px;
      }

      .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
      }

      .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #0f172a;
      }

      .note-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 14px;
        padding: 14px;
        margin-top: 12px;
      }

      .risk-meter-bg {
        height: 12px;
        width: 100%;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 10px;
      }

      .table {
        background: white;
      }
    "))
  ),

  div(
    class = "main-title",
    "Heart Attack Risk Dashboard"
  ),
  div(
    class = "subtitle",
    "A cleaner and more stable version for your final project dashboard."
  ),

  fluidRow(
    column(
      3,
      div(
        class = "kpi-box",
        div(class = "kpi-label", "Complete Rows"),
        div(class = "kpi-value", textOutput("vb_n"))
      )
    ),
    column(
      3,
      div(
        class = "kpi-box",
        div(class = "kpi-label", "High Risk Cases"),
        div(class = "kpi-value", textOutput("vb_high"))
      )
    ),
    column(
      3,
      div(
        class = "kpi-box",
        div(class = "kpi-label", "Low Risk Cases"),
        div(class = "kpi-value", textOutput("vb_low"))
      )
    ),
    column(
      3,
      div(
        class = "kpi-box",
        div(class = "kpi-label", "Best Model AUC"),
        div(class = "kpi-value", textOutput("vb_aucbest"))
      )
    )
  ),

  tabsetPanel(
    id = "tabs",

    tabPanel(
      "Overview",
      br(),
      fluidRow(
        column(
          8,
          div(
            class = "card-box",
            div(class = "section-title", "Project Summary"),
            p(HTML("<strong>Both logistic regression and random forest performed near chance level</strong>, with AUC values close to 0.5.")),
            p(HTML("<strong>What we did instead:</strong> we added a rule-based score that gives extra weight to clinically important markers like Troponin and CK-MB.")),
            div(
              class = "note-box",
              HTML("<strong>⚠️ Important:</strong> This dashboard is for a statistics course project and is not a real clinical decision tool.")
            )
          )
        ),
        column(
          4,
          div(
            class = "card-box",
            div(class = "section-title", "Dataset Summary"),
            tableOutput("dataset_summary")
          )
        )
      ),

      fluidRow(
        column(
          6,
          div(
            class = "card-box",
            div(class = "section-title", "Outcome Distribution"),
            plotOutput("outcome_plot", height = "300px")
          )
        ),
        column(
          6,
          div(
            class = "card-box",
            div(class = "section-title", "Outcome Summary"),
            tableOutput("outcome_summary")
          )
        )
      )
    ),

    tabPanel(
      "Clinical Signals",
      br(),
      fluidRow(
        column(
          12,
          div(
            class = "card-box",
            div(class = "section-title", "Explore a Variable"),
            selectInput(
              "eda_var",
              "Compare with Heart Attack Risk:",
              choices = c("Gender", "Smoking", "Diabetes", "Obesity", "Family.History"),
              width = "300px"
            ),
            plotOutput("eda_bar", height = "320px")
          )
        )
      ),
      fluidRow(
        column(
          6,
          div(
            class = "card-box",
            div(class = "section-title", "Risk by Gender"),
            plotOutput("gender_plot", height = "300px")
          )
        ),
        column(
          6,
          div(
            class = "card-box",
            div(class = "section-title", "Troponin by Risk Group"),
            plotOutput("troponin_plot", height = "300px")
          )
        )
      ),
      fluidRow(
        column(
          12,
          div(
            class = "card-box",
            div(class = "section-title", "Clinical Marker Summaries"),
            selectInput(
              "summary_var",
              "Choose a marker:",
              choices = c("Troponin", "CK.MB", "Cholesterol", "Heart.rate"),
              selected = "Troponin",
              width = "250px"
            ),
            tableOutput("summary_table")
          )
        )
      )
    ),

    tabPanel(
      "Model Performance",
      br(),
      fluidRow(
        column(
          6,
          div(
            class = "card-box",
            div(class = "section-title", "Model Metrics"),
            tableOutput("metrics_table")
          )
        ),
        column(
          6,
          div(
            class = "card-box",
            div(class = "section-title", "ROC Curves"),
            plotOutput("roc_compare_plot", height = "320px")
          )
        )
      ),
      fluidRow(
        column(
          6,
          div(
            class = "card-box",
            div(class = "section-title", "Logistic Predicted Probabilities"),
            plotOutput("prob_hist", height = "320px")
          )
        ),
        column(
          6,
          div(
            class = "card-box",
            div(class = "section-title", "Random Forest Variable Importance"),
            plotOutput("rf_importance_plot", height = "320px")
          )
        )
      )
    ),

    tabPanel(
      "Risk Calculator",
      br(),
      fluidRow(
        column(
          5,
          div(
            class = "card-box",
            div(class = "section-title", "Enter Patient Profile"),

            h5("Demographics"),
            selectInput("age", "Age Group", c("Under 40", "40-50", "50-60", "60-70", "70+")),
            selectInput("gender", "Gender", c("Female", "Male")),
            selectInput("diabetes", "Diabetes", c("No", "Yes")),
            selectInput("smoking", "Smoking", c("No", "Yes")),
            selectInput("obesity", "Obesity", c("No", "Yes")),
            selectInput("family_history", "Family History", c("No", "Yes")),

            hr(),

            h5("Clinical Markers"),
            selectInput("cholesterol", "Cholesterol", c("Low", "Med", "High")),
            selectInput("heart_rate", "Heart Rate", c("Low", "Med", "High")),
            selectInput("bmi", "BMI", c("Low", "Med", "High")),
            selectInput("triglycerides", "Triglycerides", c("Low", "Med", "High")),
            selectInput("blood_sugar", "Blood Sugar", c("Low", "Med", "High")),
            selectInput("ck_mb", "CK-MB", c("Low", "Med", "High")),
            selectInput("troponin", "Troponin", c("Low", "Med", "High"))
          )
        ),
        column(
          7,
          div(
            class = "card-box",
            div(class = "section-title", "Risk Summary"),
            uiOutput("risk_label"),
            hr(),
            h4(textOutput("log_pred")),
            h4(textOutput("rf_pred")),
            div(
              class = "note-box",
              tags$ul(
                style = "margin-bottom: 0;",
                tags$li("The rule-based score is the most interpretable result here."),
                tags$li("The model outputs are shown only as rough references."),
                tags$li("This project is for learning, not for medical use.")
              )
            )
          )
        )
      )
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
    paste0(round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "1") * 100, 1), "%")
  })

  output$vb_low <- renderText({
    paste0(round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "0") * 100, 1), "%")
  })

  output$vb_aucbest <- renderText({
    round(max(as.numeric(auc(roc_log)), as.numeric(auc(roc_rf))), 3)
  })

  output$dataset_summary <- renderTable({
    dataset_summary_df
  }, striped = TRUE, bordered = TRUE, spacing = "m")

  output$outcome_summary <- renderTable({
    outcome_summary_df
  }, striped = TRUE, bordered = TRUE, spacing = "m")

  output$metrics_table <- renderTable({
    metric_df
  }, striped = TRUE, bordered = TRUE, spacing = "m")

  output$summary_table <- renderTable({
    summary_stats(heart_nomis, input$summary_var)
  }, striped = TRUE, bordered = TRUE, spacing = "m")

  output$outcome_plot <- renderPlot({
    ggplot(
      heart_nomis,
      aes(x = Heart.Attack.Risk..Binary., fill = Heart.Attack.Risk..Binary.)
    ) +
      geom_bar(width = 0.65) +
      geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3, size = 5) +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "#4f46e5", "1" = "#d1495b")) +
      labs(x = NULL, y = "Count") +
      theme_minimal(base_size = 14) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank()
      )
  })

  output$eda_bar <- renderPlot({
    ggplot(
      heart_nomis,
      aes(x = .data[[input$eda_var]], fill = Heart.Attack.Risk..Binary.)
    ) +
      geom_bar(position = "fill", width = 0.7) +
      scale_fill_manual(
        values = c("0" = "#4f46e5", "1" = "#d1495b"),
        labels = c("Low Risk", "High Risk")
      ) +
      labs(x = input$eda_var, y = "Proportion", fill = "Risk Group") +
      theme_minimal(base_size = 14) +
      theme(panel.grid.minor = element_blank())
  })

  output$gender_plot <- renderPlot({
    ggplot(
      heart_nomis,
      aes(x = Gender, fill = Heart.Attack.Risk..Binary.)
    ) +
      geom_bar(position = "fill", width = 0.7) +
      scale_fill_manual(
        values = c("0" = "#4f46e5", "1" = "#d1495b"),
        labels = c("Low Risk", "High Risk")
      ) +
      labs(x = "Gender", y = "Proportion", fill = "Risk Group") +
      theme_minimal(base_size = 14) +
      theme(panel.grid.minor = element_blank())
  })

  output$troponin_plot <- renderPlot({
    ggplot(
      heart_nomis,
      aes(x = Heart.Attack.Risk..Binary., y = Troponin, fill = Heart.Attack.Risk..Binary.)
    ) +
      geom_boxplot(width = 0.55, alpha = 0.9) +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "#4f46e5", "1" = "#d1495b")) +
      labs(x = "Risk Group", y = "Troponin") +
      theme_minimal(base_size = 14) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank()
      )
  })

  output$roc_compare_plot <- renderPlot({
    plot(roc_log, col = "#4f46e5", lwd = 3, main = "ROC Curves")
    plot(roc_rf, col = "#2a9d8f", lwd = 3, add = TRUE)
    abline(a = 0, b = 1, lty = 2, col = "gray60")
    legend(
      "bottomright",
      legend = c(
        paste("Logistic AUC =", round(as.numeric(auc(roc_log)), 3)),
        paste("Random Forest AUC =", round(as.numeric(auc(roc_rf)), 3))
      ),
      col = c("#4f46e5", "#2a9d8f"),
      lwd = 3,
      bty = "n"
    )
  })

  output$prob_hist <- renderPlot({
    ggplot(test_data, aes(x = log_prob, fill = Heart.Attack.Risk..Binary.)) +
      geom_histogram(binwidth = 0.02, alpha = 0.7, position = "identity") +
      geom_vline(xintercept = 0.5, linetype = "dashed", linewidth = 1) +
      scale_fill_manual(
        values = c("0" = "#4f46e5", "1" = "#d1495b"),
        labels = c("Low Risk", "High Risk")
      ) +
      labs(
        x = "Predicted Probability of High Risk",
        y = "Count",
        fill = "Actual Risk"
      ) +
      theme_minimal(base_size = 14) +
      theme(panel.grid.minor = element_blank())
  })

  output$rf_importance_plot <- renderPlot({
    varImpPlot(rf_model, main = "Random Forest Variable Importance")
  })

  new_patient <- reactive({
    data.frame(
      Age            = age_to_num(input$age),
      Gender         = factor(input$gender, levels = levels(train_data$Gender)),
      Diabetes       = factor(yes_no_to_factor(input$diabetes), levels = c("0", "1")),
      Smoking        = factor(yes_no_to_factor(input$smoking), levels = c("0", "1")),
      Obesity        = factor(yes_no_to_factor(input$obesity), levels = c("0", "1")),
      Family.History = factor(yes_no_to_factor(input$family_history), levels = c("0", "1")),
      Cholesterol    = to_numeric(input$cholesterol),
      Heart.rate     = to_numeric(input$heart_rate),
      BMI            = to_numeric(input$bmi),
      Triglycerides  = to_numeric(input$triglycerides),
      Blood.sugar    = to_numeric(input$blood_sugar),
      CK.MB          = to_numeric(input$ck_mb),
      Troponin       = to_numeric(input$troponin)
    )
  })

  output$risk_label <- renderUI({
    s <- rule_score(
      input$age, input$cholesterol, input$heart_rate,
      input$bmi, input$triglycerides, input$blood_sugar,
      input$ck_mb, input$troponin
    )
    index <- s / 9
    color <- if (index < 0.33) "#2a9d8f" else if (index < 0.66) "#f4a261" else "#d1495b"
    label <- if (index < 0.33) "Low Risk" else if (index < 0.66) "Moderate Risk" else "High Risk"
    pct <- round(index * 100)

    div(
      style = "padding: 8px 0;",
      div(
        style = paste0("font-size: 2.2rem; font-weight: 800; color:", color, ";"),
        label
      ),
      div(
        style = "color:#64748b; font-size:1rem; margin-top:6px;",
        paste0("Rule-based score: ", round(index, 2), " / 1.0")
      ),
      div(
        class = "risk-meter-bg",
        div(
          style = paste0(
            "height:100%; width:", pct, "%; background:", color,
            "; border-radius:999px;"
          )
        )
      )
    )
  })

  output$log_pred <- renderText({
    prob <- predict(log_model, new_patient(), type = "response")
    paste("Logistic model output:", round(prob * 100, 1), "%")
  })

  output$rf_pred <- renderText({
    prob <- predict(rf_model, new_patient(), type = "prob")[, "1"]
    paste("Random forest output:", round(prob * 100, 1), "%")
  })
}

# ----------------------------
# 10. Run app
# ----------------------------
shinyApp(ui = ui, server = server)
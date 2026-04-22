# ============================================================
# Heart Attack Risk Dashboard
# Hannah Chen — Final Project
# ============================================================

library(shiny)
library(shinydashboard)
library(tidyverse)
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
# 4. Fit models once at startup
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
# 5. Evaluate models on test data
# ----------------------------
test_data$log_prob  <- predict(log_model, newdata = test_data, type = "response")
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

# Data is already scaled 0-1, so Low/Med/High map directly
to_numeric <- function(x) {
  switch(x,
    "Low"  = 0.2,
    "Med"  = 0.5,
    "High" = 0.8,
    0.5
  )
}

# Age also scaled 0-1 in the dataset
age_to_num <- function(x) {
  switch(x,
    "Under 40" = 0.2,
    "40-50"    = 0.35,
    "50-60"    = 0.5,
    "60-70"    = 0.65,
    "70+"      = 0.85,
    0.5
  )
}

# Convert Yes/No input back to "1"/"0" for the models
yes_no_to_factor <- function(x) {
  ifelse(x == "Yes", "1", "0")
}

# Rule-based clinical score
# Weights: Troponin = 3x, CK-MB = 2x, Age 60+ = +1 bonus
# Max possible raw score = 9, normalised to 0-1
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
  return(s)
}

# ----------------------------
# 7. UI
# ----------------------------
ui <- dashboardPage(
  skin = "blue",

  dashboardHeader(title = "HeartRisk"),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview",           tabName = "overview", icon = icon("house")),
      menuItem("EDA",                tabName = "eda",      icon = icon("chart-bar")),
      menuItem("Model Comparison",   tabName = "compare",  icon = icon("scale-balanced")),
      menuItem("Patient Calculator", tabName = "calc",     icon = icon("user-doctor"))
    )
  ),

  dashboardBody(
    tabItems(

      # ============================================================
      # OVERVIEW TAB
      # ============================================================
      tabItem(
        tabName = "overview",
        fluidRow(
          valueBoxOutput("vb_n",       width = 4),
          valueBoxOutput("vb_high",    width = 4),
          valueBoxOutput("vb_aucbest", width = 4)
        ),
        fluidRow(
          box(
            title = "Project Summary",
            width = 7,
            status = "primary",
            solidHeader = TRUE,

            p("This dashboard explores heart attack risk prediction using a cleaned dataset of 9,377 complete patient records."),

            p(tags$b("What we found:"),
              " Both our logistic regression and random forest models achieved an AUC of
                approximately 0.5 — meaning they performed no better than random guessing.
                This is an honest and common finding with synthetic datasets, where
                variables may not carry true clinical signal."),

            p(tags$b("What we did instead:"),
              " Because the statistical models struggled, we designed a rule-based clinical
                scoring system that weights known cardiac risk markers — especially Troponin
                and CK-MB — based on their real-world clinical importance."),

            p(tags$em(
              "This dashboard should be interpreted as a course-project analysis,
               not a real clinical tool."
            ), style = "color:#888;")
          ),

          box(
            title = "Outcome Distribution",
            width = 5,
            status = "warning",
            solidHeader = TRUE,
            plotOutput("outcome_plot", height = 260)
          )
        )
      ),

      # ============================================================
      # EDA TAB
      # ============================================================
      tabItem(
        tabName = "eda",
        fluidRow(
          box(
            title = "Explore a Variable",
            width = 12,
            status = "info",
            solidHeader = TRUE,
            selectInput(
              "eda_var",
              "Compare with Heart Attack Risk:",
              choices = c(
                "Gender", "Smoking", "Diabetes", "Obesity",
                "Family.History", "Previous.Heart.Problems",
                "Medication.Use"
              ),
              width = "300px"
            ),
            plotOutput("eda_bar", height = 300)
          )
        ),
        fluidRow(
          box(
            title = "Risk by Gender",
            width = 6,
            status = "info",
            solidHeader = TRUE,
            plotOutput("gender_plot", height = 280)
          ),
          # Replaces the duplicate Gender chart with Troponin by Risk Group
          box(
            title = "Troponin by Risk Group",
            width = 6,
            status = "info",
            solidHeader = TRUE,
            plotOutput("troponin_plot", height = 280)
          )
        )
      ),

      # ============================================================
      # MODEL COMPARISON TAB
      # ============================================================
      tabItem(
        tabName = "compare",
        fluidRow(
          valueBoxOutput("vb_log_auc", width = 3),
          valueBoxOutput("vb_rf_auc",  width = 3),
          valueBoxOutput("vb_log_acc", width = 3),
          valueBoxOutput("vb_rf_acc",  width = 3)
        ),
        fluidRow(
          box(
            title = "ROC Curves",
            width = 8,
            status = "danger",
            solidHeader = TRUE,
            plotOutput("roc_compare_plot", height = 330),
            p(
              tags$em(
                "An AUC near 0.5 means the model performs close to random chance.
                 The dashed diagonal line represents a coin flip, and our models
                 stay close to that line."
              ),
              style = "color:#888; font-size:12px; margin-top:8px;"
            )
          ),
          box(
            title = "Performance Metrics",
            width = 4,
            status = "danger",
            solidHeader = TRUE,
            tableOutput("metrics_table")
          )
        ),
        fluidRow(
          box(
            title = "Logistic Predicted Probabilities",
            width = 6,
            status = "danger",
            solidHeader = TRUE,
            plotOutput("prob_hist", height = 280)
          ),
          box(
            title = "Random Forest Variable Importance",
            width = 6,
            status = "danger",
            solidHeader = TRUE,
            plotOutput("rf_importance_plot", height = 280)
          )
        )
      ),

      # ============================================================
      # PATIENT CALCULATOR TAB
      # ============================================================
      tabItem(
        tabName = "calc",
        fluidRow(
          box(
            title = "Enter Patient Profile",
            width = 6,
            status = "primary",
            solidHeader = TRUE,

            tags$p(
              tags$b("Demographics"),
              style = "font-size:11px; text-transform:uppercase;
                       letter-spacing:0.05em; color:#888;"
            ),
            fluidRow(
              column(6, selectInput("age",    "Age Group:",
                                    c("Under 40", "40-50", "50-60", "60-70", "70+"))),
              column(6, selectInput("gender", "Gender:", c("Female", "Male")))
            ),
            fluidRow(
              column(6, selectInput("diabetes",       "Diabetes:",       c("No", "Yes"))),
              column(6, selectInput("smoking",        "Smoking:",        c("No", "Yes")))
            ),
            fluidRow(
              column(6, selectInput("obesity",        "Obesity:",        c("No", "Yes"))),
              column(6, selectInput("family_history", "Family History:", c("No", "Yes")))
            ),

            tags$hr(),

            tags$p(
              tags$b("Clinical Markers"),
              style = "font-size:11px; text-transform:uppercase;
                       letter-spacing:0.05em; color:#888;"
            ),
            fluidRow(
              column(6, selectInput("cholesterol",   "Cholesterol:",   c("Low", "Med", "High"))),
              column(6, selectInput("heart_rate",    "Heart Rate:",    c("Low", "Med", "High")))
            ),
            fluidRow(
              column(6, selectInput("bmi",           "BMI:",           c("Low", "Med", "High"))),
              column(6, selectInput("triglycerides", "Triglycerides:", c("Low", "Med", "High")))
            ),
            fluidRow(
              column(6, selectInput("blood_sugar",   "Blood Sugar:",   c("Low", "Med", "High"))),
              column(6, selectInput("ck_mb",         "CK-MB:",         c("Low", "Med", "High")))
            ),
            fluidRow(
              column(6, selectInput("troponin", "Troponin:", c("Low", "Med", "High")))
            )
          ),

          box(
            title = "Risk Summary",
            width = 6,
            status = "success",
            solidHeader = TRUE,

            uiOutput("risk_label"),

            hr(),

            h4(textOutput("log_pred")),
            h4(textOutput("rf_pred")),

            div(
              style = "background:#fffbea; border:1px solid #f5e47a;
                       border-radius:8px; padding:12px; margin-top:14px;",
              p(tags$b("⚠️ How to read this:"),
                style = "margin-bottom:6px; font-size:13px;"),
              tags$ul(
                style = "font-size:13px; color:#555; margin:0; padding-left:16px;",
                tags$li("The ", tags$b("Rule-Based score"),
                        " uses clinical knowledge — it is the most meaningful
                          output on this page."),
                tags$li("The ", tags$b("Logistic and Random Forest"),
                        " percentages come from models that performed weakly.
                          They should only be treated as rough reference points."),
                tags$li("This dashboard should be interpreted as a course-project
                          analysis, not a real clinical tool.")
              )
            )
          )
        )
      )

    )
  )
)

# ----------------------------
# 8. Server
# ----------------------------
server <- function(input, output) {

  # ---- Overview ----
  output$vb_n <- renderValueBox({
    valueBox(
      formatC(nrow(heart_nomis), big.mark = ","),
      "Complete Rows",
      icon  = icon("database"),
      color = "blue"
    )
  })

  output$vb_high <- renderValueBox({
    pct_high <- round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "1") * 100, 1)
    valueBox(
      paste0(pct_high, "%"),
      "High Risk",
      icon  = icon("triangle-exclamation"),
      color = "red"
    )
  })

  output$vb_aucbest <- renderValueBox({
    best_auc <- round(max(as.numeric(auc(roc_log)), as.numeric(auc(roc_rf))), 3)
    valueBox(
      best_auc,
      "Best AUC",
      icon  = icon("chart-line"),
      color = "green"
    )
  })

  output$outcome_plot <- renderPlot({
    ggplot(heart_nomis,
           aes(x = Heart.Attack.Risk..Binary.,
               fill = Heart.Attack.Risk..Binary.)) +
      geom_bar() +
      geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3) +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato")) +
      labs(x = NULL, y = "Count") +
      theme_minimal() +
      theme(legend.position = "none")
  })

  # ---- EDA ----
  output$eda_bar <- renderPlot({
    ggplot(heart_nomis,
           aes(x = .data[[input$eda_var]],
               fill = Heart.Attack.Risk..Binary.)) +
      geom_bar(position = "fill") +
      scale_fill_manual(
        values = c("0" = "steelblue", "1" = "tomato"),
        labels = c("Low Risk", "High Risk")
      ) +
      labs(x = input$eda_var, y = "Proportion", fill = "Risk Group") +
      theme_minimal()
  })

  output$gender_plot <- renderPlot({
    ggplot(heart_nomis,
           aes(x = Gender, fill = Heart.Attack.Risk..Binary.)) +
      geom_bar(position = "fill") +
      scale_fill_manual(
        values = c("0" = "steelblue", "1" = "tomato"),
        labels = c("Low Risk", "High Risk")
      ) +
      labs(x = "Gender", y = "Proportion", fill = "Risk Group") +
      theme_minimal()
  })

  # NEW: Troponin by Risk Group — shows distribution of your key cardiac marker
  output$troponin_plot <- renderPlot({
    ggplot(heart_nomis,
           aes(x = Heart.Attack.Risk..Binary., y = Troponin,
               fill = Heart.Attack.Risk..Binary.)) +
      geom_boxplot() +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato")) +
      labs(x = "Risk Group", y = "Troponin") +
      theme_minimal() +
      theme(legend.position = "none")
  })

  # ---- Model comparison ----
  output$vb_log_auc <- renderValueBox({
    valueBox(
      round(as.numeric(auc(roc_log)), 3),
      "Logistic AUC",
      icon  = icon("wave-square"),
      color = "blue"
    )
  })

  output$vb_rf_auc <- renderValueBox({
    valueBox(
      round(as.numeric(auc(roc_rf)), 3),
      "RF AUC",
      icon  = icon("tree"),
      color = "green"
    )
  })

  output$vb_log_acc <- renderValueBox({
    valueBox(
      round(cm_log$overall["Accuracy"], 3),
      "Logistic Accuracy",
      icon  = icon("check"),
      color = "yellow"
    )
  })

  output$vb_rf_acc <- renderValueBox({
    valueBox(
      round(cm_rf$overall["Accuracy"], 3),
      "RF Accuracy",
      icon  = icon("check-double"),
      color = "purple"
    )
  })

  output$roc_compare_plot <- renderPlot({
    plot(roc_log, col = "steelblue", lwd = 2, main = "ROC Curves")
    plot(roc_rf,  col = "darkgreen", lwd = 2, add = TRUE)
    abline(a = 0, b = 1, lty = 2, col = "gray50")
    legend(
      "bottomright",
      legend = c(
        paste("Logistic AUC =", round(as.numeric(auc(roc_log)), 3)),
        paste("Random Forest AUC =", round(as.numeric(auc(roc_rf)), 3))
      ),
      col = c("steelblue", "darkgreen"),
      lwd = 2,
      bty = "n"
    )
  })

  output$metrics_table <- renderTable({
    data.frame(
      Metric = c("AUC", "Accuracy", "Sensitivity", "Specificity"),
      Logistic = c(
        round(as.numeric(auc(roc_log)), 3),
        round(cm_log$overall["Accuracy"], 3),
        round(cm_log$byClass["Sensitivity"], 3),
        round(cm_log$byClass["Specificity"], 3)
      ),
      RandomForest = c(
        round(as.numeric(auc(roc_rf)), 3),
        round(cm_rf$overall["Accuracy"], 3),
        round(cm_rf$byClass["Sensitivity"], 3),
        round(cm_rf$byClass["Specificity"], 3)
      ),
      check.names = FALSE
    )
  })

  output$prob_hist <- renderPlot({
    ggplot(test_data,
           aes(x = log_prob, fill = Heart.Attack.Risk..Binary.)) +
      geom_histogram(binwidth = 0.02, alpha = 0.6, position = "identity") +
      geom_vline(xintercept = 0.5, linetype = "dashed") +
      scale_fill_manual(
        values = c("0" = "steelblue", "1" = "tomato"),
        labels = c("Low Risk", "High Risk")
      ) +
      labs(
        x    = "Predicted Probability of High Risk",
        y    = "Count",
        fill = "Actual Risk"
      ) +
      theme_minimal()
  })

  output$rf_importance_plot <- renderPlot({
    varImpPlot(rf_model, main = "Random Forest Variable Importance")
  })

  # ---- Calculator ----
  new_patient <- reactive({
    data.frame(
      Age            = age_to_num(input$age),
      Gender         = factor(input$gender,
                               levels = levels(train_data$Gender)),
      Diabetes       = factor(yes_no_to_factor(input$diabetes),
                               levels = c("0", "1")),
      Smoking        = factor(yes_no_to_factor(input$smoking),
                               levels = c("0", "1")),
      Obesity        = factor(yes_no_to_factor(input$obesity),
                               levels = c("0", "1")),
      Family.History = factor(yes_no_to_factor(input$family_history),
                               levels = c("0", "1")),
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
    s     <- rule_score(input$age, input$cholesterol, input$heart_rate,
                        input$bmi, input$triglycerides, input$blood_sugar,
                        input$ck_mb, input$troponin)
    index <- s / 9
    color <- if (index < 0.33) "#27ae60" else if (index < 0.66) "#f39c12" else "#e74c3c"
    label <- if (index < 0.33) "Low Risk" else if (index < 0.66) "Moderate Risk" else "High Risk"

    HTML(paste0(
      "<div style='text-align:center; padding:10px;'>",
      "<div style='font-size:2.4rem; font-weight:700; color:", color, ";'>",
      label,
      "</div>",
      "<div style='font-size:13px; color:#888; margin-top:4px;'>",
      "Rule score: ", round(index, 2), " / 1.0",
      "</div></div>"
    ))
  })

  output$log_pred <- renderText({
    prob <- predict(log_model, new_patient(), type = "response")
    paste("Logistic Model Output:", round(prob * 100, 1), "%")
  })

  output$rf_pred <- renderText({
    prob <- predict(rf_model, new_patient(), type = "prob")[, "1"]
    paste("Random Forest Output:", round(prob * 100, 1), "%")
  })

}

# ----------------------------
# 9. Run app
# ----------------------------
shinyApp(ui = ui, server = server)
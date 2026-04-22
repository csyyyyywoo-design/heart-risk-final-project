# ============================================================
# Heart Attack Risk Dashboard
# Hannah Chen — Final Project
# Faster Final Version
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
# 3. Use one common, simpler formula
#    (same variables used in calculator)
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
# Logistic regression
test_data$log_prob <- predict(log_model, newdata = test_data, type = "response")
test_data$log_class <- ifelse(test_data$log_prob > 0.5, "1", "0")
test_data$log_class <- factor(test_data$log_class, levels = c("0", "1"))

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

# Random forest
rf_pred <- predict(rf_model, newdata = test_data)
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")[, "1"]

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
# 6. Helper functions for calculator
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
         "Under 40" = 0.15,
         "40-50" = 0.35,
         "50-60" = 0.55,
         "60-70" = 0.75,
         "70+" = 0.90,
         0.50)
}

# ----------------------------
# 7. UI
# ----------------------------
ui <- dashboardPage(
  skin = "blue",

  dashboardHeader(title = "HeartRisk"),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("house")),
      menuItem("EDA", tabName = "eda", icon = icon("chart-bar")),
      menuItem("Model Comparison", tabName = "compare", icon = icon("scale-balanced")),
      menuItem("Patient Calculator", tabName = "calc", icon = icon("user-doctor"))
    )
  ),

  dashboardBody(
    tabItems(

      # ----------------------------
      # Overview tab
      # ----------------------------
      tabItem(
        tabName = "overview",
        fluidRow(
          valueBoxOutput("vb_n", width = 4),
          valueBoxOutput("vb_high", width = 4),
          valueBoxOutput("vb_aucbest", width = 4)
        ),
        fluidRow(
          box(
            title = "Project Summary", width = 7, status = "primary", solidHeader = TRUE,
            p("This dashboard summarizes our heart attack risk final project."),
            p("We cleaned the dataset, removed incomplete rows, and compared a logistic regression model with a random forest model."),
            p("Because model performance was weak, this dashboard should be interpreted as a course-project analysis rather than a real clinical tool.")
          ),
          box(
            title = "Outcome Distribution", width = 5, status = "warning", solidHeader = TRUE,
            plotOutput("outcome_plot", height = 260)
          )
        )
      ),

      # ----------------------------
      # EDA tab
      # ----------------------------
      tabItem(
        tabName = "eda",
        fluidRow(
          box(
            title = "Risk by Gender", width = 6, status = "info", solidHeader = TRUE,
            plotOutput("gender_plot", height = 280)
          ),
          box(
            title = "Cholesterol by Risk Group", width = 6, status = "info", solidHeader = TRUE,
            plotOutput("chol_plot", height = 280)
          )
        )
      ),

      # ----------------------------
      # Model comparison tab
      # ----------------------------
      tabItem(
        tabName = "compare",
        fluidRow(
          valueBoxOutput("vb_log_auc", width = 3),
          valueBoxOutput("vb_rf_auc", width = 3),
          valueBoxOutput("vb_log_acc", width = 3),
          valueBoxOutput("vb_rf_acc", width = 3)
        ),
        fluidRow(
          box(
            title = "ROC Curves", width = 8, status = "danger", solidHeader = TRUE,
            plotOutput("roc_compare_plot", height = 330)
          ),
          box(
            title = "Performance Metrics", width = 4, status = "danger", solidHeader = TRUE,
            tableOutput("metrics_table")
          )
        ),
        fluidRow(
          box(
            title = "Logistic Predicted Probabilities", width = 6,
            status = "danger", solidHeader = TRUE,
            plotOutput("prob_hist", height = 280)
          ),
          box(
            title = "Random Forest Variable Importance", width = 6,
            status = "danger", solidHeader = TRUE,
            plotOutput("rf_importance_plot", height = 280)
          )
        )
      ),

      # ----------------------------
      # Patient calculator tab
      # ----------------------------
      tabItem(
        tabName = "calc",
        fluidRow(
          box(
            title = "Enter Patient Profile", width = 6,
            status = "primary", solidHeader = TRUE,

            selectInput("age", "Age Group:", c("Under 40", "40-50", "50-60", "60-70", "70+")),
            selectInput("gender", "Gender:", c("Female", "Male")),
            selectInput("diabetes", "Diabetes:", c("0", "1")),
            selectInput("smoking", "Smoking:", c("0", "1")),
            selectInput("obesity", "Obesity:", c("0", "1")),
            selectInput("family_history", "Family History:", c("0", "1")),

            selectInput("cholesterol", "Cholesterol:", c("Low", "Med", "High")),
            selectInput("heart_rate", "Heart Rate:", c("Low", "Med", "High")),
            selectInput("bmi", "BMI:", c("Low", "Med", "High")),
            selectInput("triglycerides", "Triglycerides:", c("Low", "Med", "High")),
            selectInput("blood_sugar", "Blood Sugar:", c("Low", "Med", "High")),
            selectInput("ck_mb", "CK-MB:", c("Low", "Med", "High")),
            selectInput("troponin", "Troponin:", c("Low", "Med", "High"))
          ),

          box(
            title = "Predicted Risk", width = 6,
            status = "success", solidHeader = TRUE,
            h4(textOutput("log_pred")),
            h4(textOutput("rf_pred")),
            br(),
            p("These predictions come from simplified course-project models and should be interpreted cautiously.")
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
    valueBox(formatC(nrow(heart_nomis), big.mark = ","), "Complete Rows", icon = icon("database"), color = "blue")
  })

  output$vb_high <- renderValueBox({
    pct_high <- round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "1") * 100, 1)
    valueBox(paste0(pct_high, "%"), "High Risk", icon = icon("triangle-exclamation"), color = "red")
  })

  output$vb_aucbest <- renderValueBox({
    best_auc <- round(max(as.numeric(auc(roc_log)), as.numeric(auc(roc_rf))), 3)
    valueBox(best_auc, "Best AUC", icon = icon("chart-line"), color = "green")
  })

  output$outcome_plot <- renderPlot({
    ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary., fill = Heart.Attack.Risk..Binary.)) +
      geom_bar() +
      geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3) +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato")) +
      labs(x = NULL, y = "Count") +
      theme_minimal() +
      theme(legend.position = "none")
  })

  # ---- EDA ----
  output$gender_plot <- renderPlot({
    ggplot(heart_nomis, aes(x = Gender, fill = Heart.Attack.Risk..Binary.)) +
      geom_bar(position = "fill") +
      scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato"),
                        labels = c("0" = "Low Risk", "1" = "High Risk")) +
      labs(x = "Gender", y = "Proportion", fill = "Risk Group") +
      theme_minimal()
  })

  output$chol_plot <- renderPlot({
    ggplot(heart_nomis, aes(x = Heart.Attack.Risk..Binary., y = Cholesterol, fill = Heart.Attack.Risk..Binary.)) +
      geom_boxplot() +
      scale_x_discrete(labels = c("0" = "Low Risk", "1" = "High Risk")) +
      scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato")) +
      labs(x = "Risk Group", y = "Cholesterol") +
      theme_minimal() +
      theme(legend.position = "none")
  })

  # ---- Model comparison ----
  output$vb_log_auc <- renderValueBox({
    valueBox(round(as.numeric(auc(roc_log)), 3), "Logistic AUC", icon = icon("wave-square"), color = "blue")
  })

  output$vb_rf_auc <- renderValueBox({
    valueBox(round(as.numeric(auc(roc_rf)), 3), "RF AUC", icon = icon("tree"), color = "green")
  })

  output$vb_log_acc <- renderValueBox({
    valueBox(round(cm_log$overall["Accuracy"], 3), "Logistic Accuracy", icon = icon("check"), color = "yellow")
  })

  output$vb_rf_acc <- renderValueBox({
    valueBox(round(cm_rf$overall["Accuracy"], 3), "RF Accuracy", icon = icon("check-double"), color = "purple")
  })

  output$roc_compare_plot <- renderPlot({
    plot(roc_log, col = "steelblue", lwd = 2, main = "ROC Curves")
    plot(roc_rf, col = "darkgreen", lwd = 2, add = TRUE)
    abline(a = 0, b = 1, lty = 2, col = "gray50")
    legend("bottomright",
           legend = c(
             paste("Logistic AUC =", round(as.numeric(auc(roc_log)), 3)),
             paste("Random Forest AUC =", round(as.numeric(auc(roc_rf)), 3))
           ),
           col = c("steelblue", "darkgreen"),
           lwd = 2,
           bty = "n")
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
    ggplot(test_data, aes(x = log_prob, fill = Heart.Attack.Risk..Binary.)) +
      geom_histogram(binwidth = 0.02, alpha = 0.6, position = "identity") +
      geom_vline(xintercept = 0.5, linetype = "dashed") +
      scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato"),
                        labels = c("0" = "Low Risk", "1" = "High Risk")) +
      labs(x = "Predicted Probability of High Risk", y = "Count", fill = "Actual Risk") +
      theme_minimal()
  })

  output$rf_importance_plot <- renderPlot({
    varImpPlot(rf_model, main = "Random Forest Variable Importance")
  })

  # ---- Calculator ----
  new_patient <- reactive({
    data.frame(
      Age = age_to_num(input$age),
      Gender = factor(input$gender, levels = levels(train_data$Gender)),
      Diabetes = factor(input$diabetes, levels = c("0", "1")),
      Smoking = factor(input$smoking, levels = c("0", "1")),
      Obesity = factor(input$obesity, levels = c("0", "1")),
      Family.History = factor(input$family_history, levels = c("0", "1")),
      Cholesterol = to_numeric(input$cholesterol),
      Heart.rate = to_numeric(input$heart_rate),
      BMI = to_numeric(input$bmi),
      Triglycerides = to_numeric(input$triglycerides),
      Blood.sugar = to_numeric(input$blood_sugar),
      CK.MB = to_numeric(input$ck_mb),
      Troponin = to_numeric(input$troponin)
    )
  })

  output$log_pred <- renderText({
    prob <- predict(log_model, new_patient(), type = "response")
    paste("Logistic Regression Risk:", round(prob * 100, 1), "%")
  })

  output$rf_pred <- renderText({
    prob <- predict(rf_model, new_patient(), type = "prob")[, "1"]
    paste("Random Forest Risk:", round(prob * 100, 1), "%")
  })
}

# ----------------------------
# 9. Run app
# ----------------------------
shinyApp(ui = ui, server = server)
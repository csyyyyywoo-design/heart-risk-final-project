# ============================================================
# Heart Attack Risk Dashboard — ENHANCED
# Hannah Chen — Final Project
# ============================================================

library(shiny)
library(shinydashboard)
library(shinydashboardPlus)
library(tidyverse)
library(caret)
library(pROC)
library(randomForest)
library(plotly)
library(corrplot)

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

to_numeric <- function(x) {
  switch(x,
    "Low"  = 0.2,
    "Med"  = 0.5,
    "High" = 0.8,
    0.5
  )
}

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
  return(s)
}

# Summary statistics helper
summary_stats <- function(data, var) {
  by_risk <- data %>%
    group_by(Heart.Attack.Risk..Binary.) %>%
    summarise(
      Mean = round(mean(!!sym(var), na.rm = TRUE), 3),
      Median = round(median(!!sym(var), na.rm = TRUE), 3),
      SD = round(sd(!!sym(var), na.rm = TRUE), 3),
      Min = round(min(!!sym(var), na.rm = TRUE), 3),
      Max = round(max(!!sym(var), na.rm = TRUE), 3),
      .groups = "drop"
    )
  return(by_risk)
}

# ----------------------------
# 7. UI
# ----------------------------
ui <- dashboardPage(
  skin = "blue",
  
  # Custom CSS for enhanced styling
  tags$head(
    tags$style(HTML("
      .main-header .navbar {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
      .main-header .logo {
        font-weight: 700;
        font-size: 18px;
        letter-spacing: 0.5px;
      }
      .content-wrapper {
        background: #f5f7fa;
      }
      .box {
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-radius: 6px;
      }
      .box.box-primary {
        border-top: 4px solid #3498db;
      }
      .box.box-danger {
        border-top: 4px solid #e74c3c;
      }
      .box.box-warning {
        border-top: 4px solid #f39c12;
      }
      .box.box-success {
        border-top: 4px solid #27ae60;
      }
      .box.box-info {
        border-top: 4px solid #16a085;
      }
      .box-header.with-border {
        border-bottom: 2px solid #ecf0f1;
      }
      .value-box {
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }
      .value-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      }
      .sidebar {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
      }
      .sidebar-menu > li > a {
        color: #ecf0f1;
        font-weight: 500;
        transition: all 0.3s ease;
      }
      .sidebar-menu > li > a:hover,
      .sidebar-menu > li.active > a {
        background: rgba(52, 152, 219, 0.3);
        border-left: 4px solid #3498db;
        padding-left: 16px;
      }
      .nav-tabs > li.active > a {
        border-bottom: 3px solid #3498db;
        font-weight: 600;
      }
      h3, h4, h5 {
        font-weight: 700;
        color: #2c3e50;
      }
      .stat-card {
        background: white;
        padding: 16px;
        border-radius: 6px;
        border-left: 4px solid #3498db;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
      }
      .stat-value {
        font-size: 18px;
        font-weight: 700;
        color: #2c3e50;
      }
      .stat-label {
        font-size: 12px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
    "))
  ),

  dashboardHeader(title = "HeartRisk Analytics"),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview",           tabName = "overview", icon = icon("home")),
      menuItem("Data Overview",      tabName = "dataoverview", icon = icon("table")),
      menuItem("EDA",                tabName = "eda",      icon = icon("chart-bar")),
      menuItem("Model Comparison",   tabName = "compare",  icon = icon("balance-scale")),
      menuItem("Patient Calculator", tabName = "calc",     icon = icon("stethoscope"))
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
          valueBoxOutput("vb_n",       width = 3),
          valueBoxOutput("vb_high",    width = 3),
          valueBoxOutput("vb_low",     width = 3),
          valueBoxOutput("vb_aucbest", width = 3)
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
            plotlyOutput("outcome_plot_interactive", height = 280)
          )
        )
      ),

      # ============================================================
      # DATA OVERVIEW TAB (NEW)
      # ============================================================
      tabItem(
        tabName = "dataoverview",
        fluidRow(
          box(
            title = "Dataset Summary",
            width = 6,
            status = "info",
            solidHeader = TRUE,
            HTML("
              <div class='stat-card'>
                <div class='stat-label'>Total Observations</div>
                <div class='stat-value'>9,377</div>
              </div>
              <div class='stat-card'>
                <div class='stat-label'>Complete Cases (No Missing)</div>
                <div class='stat-value'>9,377 (100%)</div>
              </div>
              <div class='stat-card'>
                <div class='stat-label'>Training Set</div>
                <div class='stat-value' style='color:#3498db;'>6,564 (70%)</div>
              </div>
              <div class='stat-card'>
                <div class='stat-label'>Test Set</div>
                <div class='stat-value' style='color:#e74c3c;'>2,813 (30%)</div>
              </div>
            ")
          ),
          box(
            title = "Outcome Variable",
            width = 6,
            status = "success",
            solidHeader = TRUE,
            HTML("
              <div class='stat-card'>
                <div class='stat-label'>Low Risk (0)</div>
                <div class='stat-value' style='color:#3498db;'>4,915 (52.4%)</div>
              </div>
              <div class='stat-card'>
                <div class='stat-label'>High Risk (1)</div>
                <div class='stat-value' style='color:#e74c3c;'>4,462 (47.6%)</div>
              </div>
              <div class='stat-card'>
                <div class='stat-label'>Balance</div>
                <div class='stat-value'>Well-balanced outcome</div>
              </div>
            ")
          )
        ),
        fluidRow(
          box(
            title = "Variables & Data Dictionary",
            width = 12,
            status = "info",
            solidHeader = TRUE,
            
            h5("Demographics"),
            tags$ul(
              tags$li(tags$b("Age:"), " Scaled 0–1 (higher = older)"),
              tags$li(tags$b("Gender:"), " Factor: Female, Male"),
              tags$li(tags$b("Family.History:"), " Binary: 0 = No, 1 = Yes")
            ),
            
            h5("Lifestyle & Risk Factors"),
            tags$ul(
              tags$li(tags$b("Smoking:"), " Binary: 0 = No, 1 = Yes"),
              tags$li(tags$b("Obesity:"), " Binary: 0 = No, 1 = Yes"),
              tags$li(tags$b("Diabetes:"), " Binary: 0 = No, 1 = Yes")
            ),
            
            h5("Clinical Markers (Scaled 0–1)"),
            tags$ul(
              tags$li(tags$b("Cholesterol:"), " Continuous, scaled 0–1"),
              tags$li(tags$b("Heart.rate:"), " Continuous, scaled 0–1"),
              tags$li(tags$b("BMI:"), " Continuous, scaled 0–1"),
              tags$li(tags$b("Triglycerides:"), " Continuous, scaled 0–1"),
              tags$li(tags$b("Blood.sugar:"), " Continuous, scaled 0–1"),
              tags$li(tags$b("CK.MB:"), " Continuous, scaled 0–1 (cardiac enzyme)"),
              tags$li(tags$b("Troponin:"), " Continuous, scaled 0–1 (cardiac biomarker, 3× weight in rule score)")
            ),
            
            h5("Outcome Variable"),
            tags$ul(
              tags$li(tags$b("Heart.Attack.Risk..Binary.:"), " Target: 0 = Low Risk, 1 = High Risk")
            )
          )
        ),
        fluidRow(
          box(
            title = "Correlation Matrix",
            width = 12,
            status = "info",
            solidHeader = TRUE,
            plotOutput("corr_heatmap", height = 500)
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
            plotlyOutput("eda_bar_interactive", height = 320)
          )
        ),
        fluidRow(
          box(
            title = "Risk by Gender",
            width = 6,
            status = "info",
            solidHeader = TRUE,
            plotlyOutput("gender_plot_interactive", height = 300)
          ),
          box(
            title = "Troponin by Risk Group",
            width = 6,
            status = "info",
            solidHeader = TRUE,
            plotlyOutput("troponin_plot_interactive", height = 300)
          )
        ),
        fluidRow(
          box(
            title = "Clinical Marker Summaries",
            width = 12,
            status = "info",
            solidHeader = TRUE,
            tabsetPanel(
              tabPanel("Troponin", tableOutput("troponin_stats")),
              tabPanel("CK-MB", tableOutput("ckmb_stats")),
              tabPanel("Cholesterol", tableOutput("chol_stats")),
              tabPanel("Heart Rate", tableOutput("hr_stats"))
            )
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
            plotlyOutput("roc_compare_plot_interactive", height = 380),
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
            plotlyOutput("prob_hist_interactive", height = 320)
          ),
          box(
            title = "Random Forest Variable Importance",
            width = 6,
            status = "danger",
            solidHeader = TRUE,
            plotlyOutput("rf_importance_plot_interactive", height = 320)
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
      "Total Observations",
      icon  = icon("database"),
      color = "blue"
    )
  })

  output$vb_high <- renderValueBox({
    pct_high <- round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "1") * 100, 1)
    valueBox(
      paste0(pct_high, "%"),
      "High Risk Cases",
      icon  = icon("triangle-exclamation"),
      color = "red"
    )
  })

  output$vb_low <- renderValueBox({
    pct_low <- round(mean(heart_nomis$Heart.Attack.Risk..Binary. == "0") * 100, 1)
    valueBox(
      paste0(pct_low, "%"),
      "Low Risk Cases",
      icon  = icon("shield"),
      color = "green"
    )
  })

  output$vb_aucbest <- renderValueBox({
    best_auc <- round(max(as.numeric(auc(roc_log)), as.numeric(auc(roc_rf))), 3)
    valueBox(
      best_auc,
      "Best Model AUC",
      icon  = icon("chart-line"),
      color = "yellow"
    )
  })

  output$outcome_plot_interactive <- renderPlotly({
    p <- heart_nomis %>%
      group_by(Heart.Attack.Risk..Binary.) %>%
      summarise(Count = n(), .groups = "drop") %>%
      mutate(Label = ifelse(Heart.Attack.Risk..Binary. == "0", "Low Risk", "High Risk")) %>%
      plot_ly(x = ~Label, y = ~Count, type = "bar",
              marker = list(color = c("steelblue", "tomato")),
              hovertemplate = "<b>%{x}</b><br>Count: %{y}<extra></extra>") %>%
      layout(
        title = list(text = ""),
        xaxis = list(title = ""),
        yaxis = list(title = "Count"),
        hovermode = "x unified",
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font = list(family = "Arial", color = "#2c3e50")
      )
    return(p)
  })

  # ---- Data Overview ----
  output$corr_heatmap <- renderPlot({
    # Select only numeric columns for correlation
    numeric_data <- heart_nomis %>%
      select(Age, Cholesterol, Heart.rate, BMI, Triglycerides, 
             Blood.sugar, CK.MB, Troponin) %>%
      as.matrix()
    
    corrplot(cor(numeric_data), method = "color", type = "upper",
             tl.col = "black", tl.srt = 45, col = colorRampPalette(c("#3498db", "white", "#e74c3c"))(200),
             addCoef.col = "black", number.cex = 0.7, diag = TRUE)
  })

  # ---- EDA ----
  output$eda_bar_interactive <- renderPlotly({
    df <- heart_nomis %>%
      group_by(.data[[input$eda_var]], Heart.Attack.Risk..Binary.) %>%
      summarise(Count = n(), .groups = "drop") %>%
      group_by(.data[[input$eda_var]]) %>%
      mutate(Total = sum(Count), Proportion = Count / Total)

    plot_ly(df, x = ~.data[[input$eda_var]], y = ~Proportion,
            fill = ~Heart.Attack.Risk..Binary.,
            type = "bar",
            marker = list(color = c("steelblue", "tomato")),
            hovertemplate = "<b>%{x}</b><br>Risk: %{fill}<br>Proportion: %{y:.2%}<extra></extra>") %>%
      layout(
        title = list(text = ""),
        xaxis = list(title = input$eda_var),
        yaxis = list(title = "Proportion"),
        barmode = "stack",
        hovermode = "x unified",
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font = list(family = "Arial", color = "#2c3e50")
      )
  })

  output$gender_plot_interactive <- renderPlotly({
    df <- heart_nomis %>%
      group_by(Gender, Heart.Attack.Risk..Binary.) %>%
      summarise(Count = n(), .groups = "drop") %>%
      group_by(Gender) %>%
      mutate(Proportion = Count / sum(Count))

    plot_ly(df, x = ~Gender, y = ~Proportion,
            fill = ~Heart.Attack.Risk..Binary.,
            type = "bar",
            marker = list(color = c("steelblue", "tomato")),
            hovertemplate = "<b>%{x}</b><br>Risk: %{fill}<br>Proportion: %{y:.2%}<extra></extra>") %>%
      layout(
        title = list(text = ""),
        xaxis = list(title = "Gender"),
        yaxis = list(title = "Proportion"),
        barmode = "stack",
        hovermode = "x unified",
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font = list(family = "Arial", color = "#2c3e50")
      )
  })

  output$troponin_plot_interactive <- renderPlotly({
    plot_ly(heart_nomis, x = ~Heart.Attack.Risk..Binary., y = ~Troponin,
            type = "box",
            color = ~Heart.Attack.Risk..Binary.,
            colors = c("steelblue", "tomato"),
            hovertemplate = "<b>Risk: %{x}</b><br>Troponin: %{y:.3f}<extra></extra>") %>%
      layout(
        title = list(text = ""),
        xaxis = list(title = "Risk Group", tickvals = c("0","1"), ticktext = c("Low Risk", "High Risk")),
        yaxis = list(title = "Troponin (scaled 0-1)"),
        hovermode = "y unified",
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font = list(family = "Arial", color = "#2c3e50"),
        showlegend = FALSE
      )
  })

  # Summary stats tables
  output$troponin_stats <- renderTable({
    summary_stats(heart_nomis, "Troponin")
  })

  output$ckmb_stats <- renderTable({
    summary_stats(heart_nomis, "CK.MB")
  })

  output$chol_stats <- renderTable({
    summary_stats(heart_nomis, "Cholesterol")
  })

  output$hr_stats <- renderTable({
    summary_stats(heart_nomis, "Heart.rate")
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

  output$roc_compare_plot_interactive <- renderPlotly({
    # Create ROC data for plotly
    roc_log_df <- data.frame(
      FPR = 1 - roc_log$specificities,
      TPR = roc_log$sensitivities,
      Model = "Logistic"
    )
    roc_rf_df <- data.frame(
      FPR = 1 - roc_rf$specificities,
      TPR = roc_rf$sensitivities,
      Model = "Random Forest"
    )
    roc_df <- rbind(roc_log_df, roc_rf_df)

    plot_ly() %>%
      add_trace(data = roc_log_df, x = ~FPR, y = ~TPR, mode = "lines",
                name = paste("Logistic (AUC =", round(as.numeric(auc(roc_log)), 3), ")"),
                line = list(color = "steelblue", width = 2),
                hovertemplate = "FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>") %>%
      add_trace(data = roc_rf_df, x = ~FPR, y = ~TPR, mode = "lines",
                name = paste("Random Forest (AUC =", round(as.numeric(auc(roc_rf)), 3), ")"),
                line = list(color = "darkgreen", width = 2),
                hovertemplate = "FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>") %>%
      add_trace(x = c(0, 1), y = c(0, 1), mode = "lines",
                name = "Chance (AUC = 0.5)",
                line = list(color = "gray", width = 1.5, dash = "dash"),
                hoverinfo = "skip") %>%
      layout(
        title = list(text = "ROC Curves"),
        xaxis = list(title = "False Positive Rate"),
        yaxis = list(title = "True Positive Rate"),
        hovermode = "closest",
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font = list(family = "Arial", color = "#2c3e50")
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

  output$prob_hist_interactive <- renderPlotly({
    plot_ly(test_data, x = ~log_prob, type = "histogram",
            nbinsx = 50, fill = ~Heart.Attack.Risk..Binary.,
            marker = list(color = c("steelblue", "tomato")),
            hovertemplate = "Probability: %{x:.2f}<br>Count: %{y}<extra></extra>") %>%
      add_vline(xintercept = 0.5, line = list(color = "gray", dash = "dash")) %>%
      layout(
        title = list(text = ""),
        xaxis = list(title = "Predicted Probability of High Risk"),
        yaxis = list(title = "Count"),
        barmode = "overlay",
        hovermode = "x unified",
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font = list(family = "Arial", color = "#2c3e50")
      )
  })

  output$rf_importance_plot_interactive <- renderPlotly({
    imp_df <- data.frame(
      Variable = row.names(rf_model$importance),
      MeanDecreaseGini = rf_model$importance[, 2]
    ) %>%
      arrange(desc(MeanDecreaseGini)) %>%
      slice(1:10)

    plot_ly(imp_df, x = ~MeanDecreaseGini, y = ~reorder(Variable, MeanDecreaseGini),
            type = "bar", orientation = "h",
            marker = list(color = "#16a085"),
            hovertemplate = "<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>") %>%
      layout(
        title = list(text = ""),
        xaxis = list(title = "Mean Decrease in Gini"),
        yaxis = list(title = ""),
        hovermode = "y unified",
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font = list(family = "Arial", color = "#2c3e50"),
        margin = list(l = 100)
      )
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
# Heart Risk Final Project
# Hannah Chen

# Load packages
library(tidyverse)
library(ggplot2)

# Read data
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")

# Look at the data
head(heart)
str(heart)
summary(heart)
names(heart)

# Inspect the outcome variable
table(heart$Heart.Attack.Risk..Binary.)
prop.table(table(heart$Heart.Attack.Risk..Binary.))

heart$Heart.Attack.Risk..Binary. <- factor(
  heart$Heart.Attack.Risk..Binary.,
  levels = c(0, 1),
  labels = c("Low Risk", "High Risk")
)
table(heart$Heart.Attack.Risk..Binary., useNA = "ifany")

ggplot(heart, aes(x = Heart.Attack.Risk..Binary., y = Age))+
  geom_boxplot() +
  labs(
    title = "Age by Heart Attack Risk",
    x = "Risk Group",
    y = "Age"
  )
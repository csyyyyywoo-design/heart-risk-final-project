# Heart Risk Final Project
# Hannah Chen

# Load packages
library(tidyverse)
library(ggplot2)
library(dplyr)

# Load data
heart <- read.csv("heart-attack-risk-prediction-dataset.csv")

# Look at the data
head(heart) # shows first few rows
dim(heart) # tells rows and columns
names(heart) # column names
str(heart) # variable types
summary(heart) # quick summary

# Inspect the outcome variable
table(heart$Heart.Attack.Risk..Binary.)
prop.table(table(heart$Heart.Attack.Risk..Binary.))

# Check for missing values
colSums(is.na(heart))
sum(!complete.cases(heart)) # number of missing values
heart_missing <- heart[!complete.cases(heart), ]
dim(heart_missing)

heart_nomis <- na.omit(heart) # Removes every row that has at leat one missing value
colSums(is.na(heart_nomis))
dim(heart_nomis)

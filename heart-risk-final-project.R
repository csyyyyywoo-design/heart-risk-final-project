# Heart Risk Final Project
# Hannah Chen

# Load packages
library(tidyverse)
library(ggplot2)
library(dplyr)

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

# Save the incomplete rows
heart_missing <- heart[!complete.cases(heart), ] # pulls out those incomplete rows
dim(heart_missing)

# Check the missing-data pattern
colSums(is.na(heart_missing)) # which columns are missing in the incomplete rows

# Optional check:
# if this gives 1, then all incomplete rows have the same missing pattern
nrow(unique(is.na(heart_missing)))

# Remove incomplete rows
heart_nomis <- na.omit(heart) # Removes rows whith at least one missing value

# Check the cleaned data
colSums(is.na(heart_nomis)) # should all be 0
dim(heart_nomis) # number of rows and columns after removal



# Clean Gender
table(heart_nomis$Gender, useNA = "ifany")

# Turn Gender into a factor
heart_nomis$Gender <- as.factor(heart_nomis$Gender)
str(heart_nomis$Gender)

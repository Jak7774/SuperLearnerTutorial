#-----------------------------------------------------------
#  Imperial Clinical Trials Unit, Imperial College London, UK
#  Program name:     XGBoost Example
#  Project:          SuperLearner
#  Written by:       Jack Elkes 
#  Date of creation: 20JUL2020
#  Description:      Worked example to understand how XGBoost works
#-----------------------------------------------------------

#install.packages("xgboost")
require(xgboost)
require(DiagrammeR)
library(Matrix)

#------------------------
# Setup
#------------------------

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # Current WD
data <- read.csv("../../../Data/datasets_180_408_data.csv")
final <- subset(data, select=-id:-diagnosis)
xvars <- subset(final, select=-X)
train_obs <- sample(floor(0.7 * nrow(data)))
x_train <- xvars[train_obs, ] 
x_test <- xvars[-train_obs, ]

outcome <- as.factor(data$diagnosis)
y_train = ifelse(outcome[train_obs]=="M",1,0)
y_test = ifelse(outcome[-train_obs]=="M",1,0)

#---------------------------
# Standard Logistic Regression
#---------------------------

X <- model.matrix(~ -1 + ., x_train)
mylogit <- glm(y_train ~ X, data = x_train, family = "binomial") ## Didn't Converege?
summary(mylogit)

#---------------------------
# XGBoost
#---------------------------

xtrain <- Matrix(as.matrix(x_train), sparse = TRUE)
ytrain <- Matrix(as.matrix(y_train), sparse = TRUE)

xtest <- Matrix(as.matrix(x_test), sparse = TRUE)
ytest <- Matrix(as.matrix(y_test), sparse = TRUE)

# Train the Data
bstSparse <- xgboost(data = xtrain, label = ytrain, # Label = Outcome 
                     max.depth = 4, eta = 0.5, nthread = 2, nrounds = 2, objective = "binary:logistic")

#---------------------------
# Perform the Prediction
#---------------------------

pred <- predict(bstSparse, xtest) ## predict the outcome using the train outcome
print(length(pred))
print(head(pred)) # Not Binary!! 

prediction <- as.numeric(pred > 0.5) # Convert to Binary
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != ytest) # Mean number of incorrect predictions
print(paste("test-error=", err))

#---------------------------
# Dataset Manipulation
#---------------------------

importance_matrix <- xgb.importance(model = bstSparse)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

# Reset Results
xgb.dump(bstSparse, with_stats = TRUE)
xgb.plot.tree(model = bstSparse)

# Detailed view of tree
xgb.model.dt.tree(xtrain@Dimnames[[2]], model = bstSparse)

# Cover. 
# The number of times a feature is used to split the data across all trees weighted by the number of training data points that go through those splits.
# Gain. 
# The average training loss reduction gained when using a feature for splitting.

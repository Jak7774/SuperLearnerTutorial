#-----------------------------------------------------------
#  Imperial Clinical Trials Unit, Imperial College London, UK
#  Program name:     XGBoost Example
#  Project:          SuperLearner
#  Written by:       Jack Elkes 
#  Date of creation: 20JUL2020
#  Description:      Worked example to understand how XGBoost works
#-----------------------------------------------------------

install.packages("xgboost")
require(xgboost)
require(DiagrammeR)

# Example to predict whether a mushroom can be eaten or not
# Dataset = Example already with XGBoost package

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test

str(train)
dim(train$data) # View dimensions of the train data
dim(test$data) # Dimensions of the test data

class(train$data)[1] # dgCMatrix means it is sparse (0's are missing to reduce size) - assume means no missing allowed?

#---------------------------
# Train the Data
#---------------------------

train$label[1:10] # View first 10 obs
colnames(train$data) # View the variables A.K.A Features

bstSparse <- xgboost(data = train$data, label = train$label, # Label = Outcome 
                     max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

# If preferred to use a standard R data set (with 1/0's) 
# bstDense <- xgboost(data = as.matrix(train$data), label = train$label, 
#                    max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

dtrain <- xgb.DMatrix(data = train$data, label = train$label)
bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 2)


#---------------------------
# Perform the Prediction
#---------------------------

pred <- predict(bstSparse, test$data) ## predict the outcome using the train outcome
print(length(pred))
print(head(pred)) # Not Binary!! 

prediction <- as.numeric(pred > 0.5) # Convert to Binary
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != test$label) # Mean number of incorrect predictions
print(paste("test-error=", err))


#---------------------------
# Advanced Features 
#---------------------------

dtrain <- xgb.DMatrix(data = train$data, label=train$label)
dtest <- xgb.DMatrix(data = test$data, label=test$label)

# This gives XGBoost a view of the data to know how well it's performed
watchlist <- list(train=dtrain, test=dtest)
bst <- xgb.train(data=dtrain, max.depth=2, eta=1, nthread = 2, nrounds=2, 
                 watchlist=watchlist, objective = "binary:logistic")

# Use eval.metric to monitor selected metrics
bst <- xgb.train(data=dtrain, booster = "gblinear", max.depth=2, eta=1, nthread = 2, nrounds=2, 
                 watchlist=watchlist, eval.metric = "error", eval.metric = "logloss", objective = "binary:logistic")

#---------------------------
# Dataset Manipulation
#---------------------------

xgb.DMatrix.save(dtrain, "dtrain.buffer")
dtrain2 <- xgb.DMatrix("dtrain.buffer")
bst <- xgb.train(data=dtrain2, max.depth=2, eta=1, nthread = 2, nrounds=2, watchlist=watchlist, objective = "binary:logistic")

label = getinfo(dtest, "label")
pred <- predict(bst, dtest)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))

importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

# Reset Results
xgb.dump(bst, with_stats = TRUE)
xgb.plot.tree(model = bst)

# save model to binary local file
xgb.save(bst, "xgboost.model")

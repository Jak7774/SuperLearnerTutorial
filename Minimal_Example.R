# Minimal working example with the iris dataset
library(SuperLearner)
library(ggplot2)
library(plyr)

rm(list = ls())

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
data=read.csv("./../Data/Iris/IRIS.csv")
data=data[!data$species=="Iris-versicolor",]

set.seed(4444)
i_rows=sample(nrow(data))
data=data[i_rows,]

outcome=data$species
# outcome_cat=as.factor(outcome)
# outcome_cat=as.numeric(as.factor(outcome))
outcome_bin=revalue(outcome,c("Iris-virginica"=1))
outcome_bin=revalue(outcome_bin,c("Iris-setosa"=0))
outcome_bin=as.numeric(outcome_bin)

N_sample_train=70
X=subset(data,select=-species)
X_train=X[1:N_sample_train,]
X_val=X[-(1:N_sample_train),]

# y_train=outcome_cat[1:N_sample_train]
# y_val=outcome_cat[-(1:N_sample_train)]
# sl=SuperLearner(Y=y_train,X=X_train,family=gaussian(),SL.library="SL.glm")

y_train=outcome_bin[1:N_sample_train]
y_val=outcome_bin[-(1:N_sample_train)]
sl=SuperLearner(Y=y_train,X=X_train,family=binomial(),SL.library="SL.glm")

pred=predict(sl,X_val,onlySL=TRUE)
qplot(y_val,pred$pred[,])

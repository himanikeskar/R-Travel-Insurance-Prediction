rm(list=ls())
library(plyr)
install.packages("readr")
install.packages("glmnet")
library(dplyr)
library(caret)
library(ggplot2)
install.packages("repr")

data = read.csv("TravelInsurancePrediction.csv", header= T)
head(data)
dim(data)
data = na.omit(data)
dim(data)

data$Employment.Type = ifelse(data$Employment.Type== "Government Sector",1,0) #Government Sector =1, Private Sector/Self Employed = 0
data$GraduateOrNot = ifelse(data$GraduateOrNot == "Yes", 1,0) # yes=1, no=0
data$FrequentFlyer = ifelse(data$FrequentFlyer == "Yes",1,0) #yes=1, no=0
data$EverTravelledAbroad = ifelse(data$EverTravelledAbroad == "Yes",1,0) #yes =1, no=0

#basic analysis
head(data)
summary(data)
sd(data$Age)
sd(data$AnnualIncome)
sd(data$FamilyMembers)
hist(data$Age, main = "Histogram of Age")
hist(data$AnnualIncome, main = "Histogram of Annual Income")
hist(data$FamilyMembers, main = "Histogram of FamilyMembers")

##Scatter Plots
pairs(TravelInsurance~ Age+ AnnualIncome+ChronicDiseases, data = data) 

#Correlation Matrix
cor(data[10], data[5:6])
cor(data[10], data[2])

## Logistic Regression
glm.fit = glm(TravelInsurance~ Age+ Employment.Type+GraduateOrNot+AnnualIncome+FamilyMembers+ChronicDiseases+FrequentFlyer+EverTravelledAbroad, data = data)
summary(glm.fit)
coef(glm.fit)

## Splitting the data into 70% training data and 30% testing data
set.seed(1949)
data$response= 1
data$response[data$TravelInsurance=="0"]= 0
##n= dim(data)[1]
n = nrow(data)
sample = sample(1987, 1379) ## Cross validation approach (Random distribution)
train.index = sample(1:n, (0.7*n))
train = data[sample, ]
test = data[-sample, ]
test.y = test$response

##Perform a logistic regression with training data only, using TravelInsurance as the response and other variables as predictors.
glm.fit = glm(TravelInsurance~ Age+ Employment.Type+GraduateOrNot+AnnualIncome+FamilyMembers+ChronicDiseases+FrequentFlyer+EverTravelledAbroad, data = train)
summary(glm.fit)
coef(glm.fit)

library(ISLR2)
## Computing the confusion matrix with 0.5 as the threshold and overall fraction of correct prediction.
glm.probs = predict(glm.fit,test, type = "response")
glm.class = rep("0",nrow(test))
glm.class[glm.probs > .5] = "1"
library(caret)
glm.sum=confusionMatrix(data= as.factor(glm.class), reference=as.factor(test$TravelInsurance), positive='1')
glm.sum

#ROC curve
library(pROC)
glm.roc=roc(test$TravelInsurance ~ glm.probs, plot = TRUE, print.auc = TRUE)  
auc(glm.roc)

## LDA with 0.5 as the threshold
## Linear Discriminant Analysis
library(MASS)
lda.fit = lda(TravelInsurance~ Age+ Employment.Type+GraduateOrNot+AnnualIncome+FamilyMembers+ChronicDiseases+FrequentFlyer+EverTravelledAbroad, data = train)
lda.fit
plot(lda.fit)

##confusion matrix and overall fraction of correct predictions for LDA
lda.pred = predict(lda.fit, test,type= "response")
lda.class = lda.pred$class  #prediction. 
lda.sum= confusionMatrix(data= as.factor(lda.class), reference=as.factor(test$TravelInsurance), positive="1")
lda.sum
lda.roc = roc(response= test.y, predictor=lda.pred$posterior[,1], plot = TRUE, print.auc = TRUE)  #ROC curve
auc(lda.roc)

##QDA
qda.fit = qda(TravelInsurance~ Age+ Employment.Type+GraduateOrNot+AnnualIncome+FamilyMembers+ChronicDiseases+FrequentFlyer+EverTravelledAbroad, data = train)
qda.fit
###confusion matrix and overall fraction of correct predictions for QDA
qda.pred = predict(qda.fit, test)
qda.class = qda.pred$class
qda.sum= confusionMatrix(data= as.factor(qda.class), reference=as.factor(test$TravelInsurance), positive="1")
qda.sum
qda.roc = roc(response= test.y, predictor=qda.pred$posterior[,1], plot = TRUE, print.auc = TRUE)  #ROC curve
auc(qda.roc)

# Naive Bayes
library(e1071)
nb.fit = naiveBayes(TravelInsurance ~ Age+ Employment.Type+GraduateOrNot+AnnualIncome+FamilyMembers+ChronicDiseases+FrequentFlyer+EverTravelledAbroad, data = train)
nb.fit
nb.class = predict(nb.fit, test)
nb.sum=confusionMatrix(data=as.factor(nb.class), reference= as.factor(test$TravelInsurance), positive="1")
nb.sum
mean(nb.class == as.factor(test$TravelInsurance))
nb.pred = predict(nb.fit, test, type = "raw")
nb.pred[2:9, ]
predictor=nb.pred[2:9, ]
nb.roc = roc(response=as.factor(test$TravelInsurance) , predictor=nb.pred[,2])
auc(nb.roc)
ggroc(nb.roc)

library(class)
train.X = train[, 2:9]
test.X = test[, 2:9]
train.TravelInsurance = train[, 10]
test.TravelInsurance = test[, 10]


#Ridge regression

library(glmnet)

x = model.matrix(TravelInsurance ~ Age+ Employment.Type+GraduateOrNot+AnnualIncome+FamilyMembers+ChronicDiseases+FrequentFlyer+EverTravelledAbroad, data)[, -1]
y = data$TravelInsurance
train.x= x[train.index,]
test.x= x[-train.index,]
train.y= y[train.index]
test.y = y[-train.index]
grid = 10^seq(10, -2, length = 100)
###

#set.seed(1) #select the optimal lambda
cv.out = cv.glmnet(train.x, train.y, alpha = 0)
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam
ridge.mod = glmnet(train.x, train.y, alpha = 0, lambda = grid, thresh = 1e-12)  
ridge.pred = predict(ridge.mod, s = bestlam, newx = test.x)   #prediction
mean((ridge.pred - test.y)^2)
out = glmnet(x, y, alpha = 0)   #fit the model with all data
predict(out, type = "coefficients", s = bestlam)[1:9, ]

#Lasso regression
lasso.mod = glmnet(train.x, train.y, alpha = 1, lambda = grid)
plot(lasso.mod)
cv.out = cv.glmnet(train.x, train.y, alpha = 1)
plot(cv.out)
bestlam = cv.out$lambda.min
lasso.pred = predict(lasso.mod, s = bestlam, newx = test.x)
mean((lasso.pred - test.y)^2) 
lasso.coef = predict(lasso.mod, type = "coefficients", s = bestlam)
lasso.coef

### random forest
install.packages("randomForest")
library(randomForest)

set.seed(1)
sample = sample(1987, 1379) ## Cross validation approach (Random distribution)
n = nrow(data)
train.index = sample(1:n, (0.7*n))
train = data[sample, ]
test = data[-sample, ]
y.test = test$TravelInsurance

rf= randomForest(TravelInsurance~., data = train, mtry = 8, importance = TRUE)
yhat.rf = predict(rf, newdata = test)
mean((y.test- yhat.rf)^2)
plot(y.test, yhat.rf)
abline(0, 1)
importance(rf)
varImpPlot(rf)


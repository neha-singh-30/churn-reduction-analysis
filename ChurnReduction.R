rm(list = ls())

getwd()

setwd("/home/neha/home/Project_1")

x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees',"readr","class")

lapply(x, require, character.only=T)

rm(x)

#Read the data
data_train = read.csv("Train_data.csv")
 
data_test = read.csv("Test_data.csv") 

#Let's combine the train and test data for further process
churn_data = rbind(data_train, data_test)

############Explore the data############
str(churn_data)

dim(churn_data)

churn_data$international.plan=as.factor(churn_data$international.plan)
churn_data$voice.mail.plan=as.factor(churn_data$voice.mail.plan)
churn_data$area.code=as.factor(churn_data$area.code)
churn_data$Churn=as.factor(churn_data$Churn)
churn_data$state=as.factor(churn_data$state)

#Missing Value Analysis
sum(is.na(churn_data)) 
# o/p : 0 
#Hence we do not have any missing value in this dataset


#Data Manupulation; convert string categories into factor numeric
for(i in 1:ncol(churn_data)){
  
  if(class(churn_data[,i]) == 'factor'){
    
    churn_data[,i] = factor(churn_data[,i], labels=(1:length(levels(factor(churn_data[,i])))))
    
  }
}

######################Outlier Analysis#######################
numeric_index = sapply(churn_data,is.numeric) 

numeric_data = churn_data[,numeric_index]

cnames = colnames(numeric_data)

for (i in 1:length(cnames)) {
  assign(paste0("gn",i), ggplot(aes_string( y = (cnames[i]), x= "Churn") , data = subset(churn_data)) +
           stat_boxplot(geom = "errorbar" , width = 0.5) +
           geom_boxplot(outlier.color = "red", fill = "grey", outlier.shape = 20, outlier.size = 1, notch = FALSE)+
           theme(legend.position = "bottom")+
           labs(y = cnames[i], x= "Churn")+
           ggtitle(paste("Boxplot" , cnames[i])))
}

# Plotting plots together
gridExtra::grid.arrange(gn1, gn2,gn3, ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6, ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9, ncol =3)
gridExtra::grid.arrange(gn10,gn11, ncol =3 )

#Replace all outliers with NA and impute

for(i in cnames){
  val = churn_data[,i][churn_data[,i] %in% boxplot.stats(churn_data[,i])$out]
  #print(length(val))
  churn_data[,i][churn_data[,i] %in% val] = NA
}

churn_data = knnImputation(churn_data, k = 3)

##########################Feature Selection#######################
corrgram(churn_data[,numeric_index], order = F, upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

## Chi-squared Test of Independence
factor_index = sapply(churn_data,is.factor)

factor_data = churn_data[,factor_index]

for (i in 1:6){
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}

# Dimension Reduction
churn_data = subset(churn_data, select = -c(total.day.charge,total.eve.charge,total.night.charge,total.intl.charge, 
                                          area.code,phone.number))

########################Feature Scaling#######################
#Normality Check
cnames = colnames(churn_data[,sapply(churn_data,is.numeric)])

for(i in cnames){
  churn_data[,i] = (churn_data[,i] - min(churn_data[,i])) / (max(churn_data[,i] - min(churn_data[,i])))
}

##################Model Development##########################

#Divide data into train and test using stratified sampling method
train.index = createDataPartition(churn_data$Churn, p = .80, list = FALSE)
train = churn_data[ train.index,]
test  = churn_data[-train.index,]

#1.DECISION TREE CLASSIFIER
#Develop Model on training data

DT_model = C5.0(Churn ~., train, trials = 100, rules = TRUE)

#Summary of DT model
summary(DT_model)

#write rules into disk
write(capture.output(summary(DT_model)), "DTRules.txt")

#Lets predict for test cases
DT_Predictions = predict(DT_model, test[,-15], type = "class")

#Evaluate the performance of classification model
ConfMatrix_DT = table(test$Churn, DT_Predictions)

confusionMatrix(ConfMatrix_DT)

#False Negative rate
FNR = FN/FN+TP

#Accuracy : 93.09%
#FNR : 42.5

#2.Random Forest
RF_model = randomForest(Churn ~ ., train, importance = TRUE, ntree = 500)

#Extract rules fromn random forest
#transform rf object to an inTrees' format
treeList = RF2List(RF_model)

#Extract rules
exec = extractRules(treeList, train[,-15])

#Visualize some rules
exec[1:2,]

#Make rules more readable:
readableRules = presentRules(exec, colnames(train))

readableRules[1:2,]

#Get rule metrics
ruleMetric = getRuleMetric(exec, train[,-15], train$Churn)

ruleMatric[1:2,]

#Predict test data using random forest model
RF_Predictions = predict(RF_model, test[,-15])

#Evaluate the performance of classification model
ConfMatrix_RF = table(test$Churn, RF_Predictions)

confusionMatrix(ConfMatrix_RF)

#False Negative rate
FNR = FN/FN+TP 

#Accuracy = 91.79
#FNR = 51.06

#3.Logistic Regression

logit_model = glm(Churn ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_logit = table(test$Churn, logit_Predictions)

#False Negative rate
FNR = FN/FN+TP 

#Accuracy: 87.18
#FNR: 80.85

#4.KNN Implementation
library(class)

#Predict test data
KNN_Predictions = knn(train[, 1:14], test[, 1:14], train$Churn, k = 7)

#Confusion matrix
ConfMatrix_KNN = table(KNN_Predictions, test$Churn)
confusionMatrix(ConfMatrix_KNN)

#Accuracy
sum(diag(ConfMatrix_KNN))/nrow(test)

#False Negative rate
FNR = FN/FN+TP 

#Accuracy = 85.38
#FNR = 60.86

#naive Bayes
library(e1071)

#Develop model
NB_model = naiveBayes(Churn ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:14], type = 'class')

#Look at confusion matrix
ConfMatrix_NB = table(observed = test[,15], predicted = NB_Predictions)
confusionMatrix(ConfMatrix_NB)

#Accuracy: 87.59
#FNR: 80.14
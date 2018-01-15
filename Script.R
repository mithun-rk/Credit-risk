# Reading raw data 
RawTraindata<- read.csv("C:/Users/mithun/Desktop/Temp-R files/training.csv", header = TRUE, sep = ",",stringsAsFactors = TRUE)
dim(RawTraindata)
str(RawTraindata)
RawTraindata$SeriousDlqin2yrs<- factor(RawTraindata$SeriousDlqin2yrs)
summary(RawTraindata)

# Reading the data after cleaning
traindata<- read.csv("C:/Users/mithun/Desktop/Temp-R files/newtrain.csv", header = TRUE, sep = ",",stringsAsFactors = TRUE)
head(traindata)
dim(traindata)
str(traindata)
View(traindata)
summary(traindata)
table(prop.table(traindata$SeriousDlqin2yrs))
table(traindata$SeriousDlqin2yrs)
traindata$SeriousDlqin2yrs<- as.numeric(traindata$SeriousDlqin2yrs)
levels(traindata$SeriousDlqin2yrs)
table(prop.table(traindata$SeriousDlqin2yrs))

# Converting from binary to Yes and No
traindata$SeriousDlqin2yrs <- ifelse(traindata$SeriousDlqin2yrs == 1,"Yes","No")

traindata$SeriousDlqin2yrs<- as.factor(traindata$SeriousDlqin2yrs)

# Using SMOTE for correcting data imbalance. 
library(DMwR)

frasmotedata<- SMOTE(SeriousDlqin2yrs~.,data = traindata,perc.over = 450,k=5,perc.under = 300)

dim(frasmotedata)
table(frasmotedata$SeriousDlqin2yrs)

# Removing nominal value "Case number" 
varset<- c("SeriousDlqin2yrs","RevolvingUtilizationOfUnsecuredLines","DebtRatio","NumberOfOpenCreditLinesAndLoans","NumberOfDependents")

modeltraindata<- frasmotedata[,varset]
str(modeltraindata)
write.csv(x = modeltraindata, file = "SmotedFRAformodeling.csv")

# Using H2o ML library for modelling. 
library(h2o)

h2o.init(ip = "localhost", port = 54321, nthreads= -1,max_mem_size = "6g")
traindata <- h2o.importFile(path = "C:/Users/mithun/Desktop/Temp-R files/SmotedFRAformodeling.csv" )
testdata<- h2o.importFile(path = "C:/Users/mithun/Desktop/Temp-R files/fracleantest.csv")

dim(traindata)
str(traindata)
h2o.table(traindata$SeriousDlqin2yrs)
h2o.table(testdata$SeriousDlqin2yrs)
dim(testdata)

traindata$SeriousDlqin2yrs<- as.factor(traindata$SeriousDlqin2yrs)

testdata$SeriousDlqin2yrs<- as.factor(testdata$SeriousDlqin2yrs)

# identifying response & predictor variables

y = "SeriousDlqin2yrs"
x = c("RevolvingUtilizationOfUnsecuredLines","DebtRatio","NumberOfOpenCreditLinesAndLoans","NumberOfDependents")
x = c("RevolvingUtilizationOfUnsecuredLines","DebtRatio","NumberOfOpenCreditLinesAndLoans")


#GLM

glm_fit <- h2o.glm(x = x, 
                   y = y, 
                   training_frame = traindata,
                   model_id = "glm_fit",
                   family = "binomial") 
summary(glm_fit)

# Let's compare the performance of the GLMs
glm_perf1 <- h2o.performance(model = glm_fit,
                             newdata = testdata)

print(glm_perf1)

h2o.auc(glm_perf1)
h2o.confusionMatrix(glm_perf1)
h2o.auc(glm_perf1)

plot(glm_perf1)
plot(glm_perf1,type = "roc", main= "GLM ROC CURVE")

#Random Forest
rf_fit1 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = traindata,
                            model_id = "rf_fit1", max_depth = 5, mtries = 1,
                            ntrees = 100)

print(summary(rf_fit1))

rf_perf1 <- h2o.performance(model = rf_fit1,
                            newdata = testdata)

print(rf_perf1)
h2o.auc(rf_perf1)
h2o.confusionMatrix(rf_perf1)
h2o.auc(glm_perf1)


plot(rf_perf1,main= "Random Forest ROC CURVE")
print(rf_fit1)
print(rf_perf1)

# GBM

gbm_fit<- h2o.gbm(x = x,y = y,training_frame = traindata,model_id = "gbm_fit",seed = 1000)


summary(gbm_fit)

gbm_perf1 <- h2o.performance(model = gbm_fit,
                             newdata = testdata)
print(gbm_perf1)
h2o.auc(gbm_perf1)
h2o.confusionMatrix(gbm_perf1)
plot(gbm_perf1,main= "GBM ROC CURVE")


plot(rf_perf1,main= "Random Forest ROC CURVE")
print(rf_fit1)
print(rf_perf1)

h2o.shutdown()

# Decision tree
dttraindata <- read.csv("C:/Users/mithun/Desktop/Temp-R files/SmotedFRAformodeling.csv", header = TRUE, sep = ",") 
dttestdata<- read.csv("C:/Users/mithun/Desktop/Temp-R files/fracleantest.csv", header = TRUE, sep = ",")

dttraindata<- dttraindata[,varset]

library(rpart)
r.ctrl= rpart.control(minsplit = 900,minbucket = 300,cp = 0,xval = 10)
dt_fit<- rpart(dttraindata$SeriousDlqin2yrs~.,data = dttraindata, method = "class",control = r.ctrl)

plot(dt_fit)
text(dt_fit)
install.packages('rpart.plot')
library(rattle)
library(rpart.plot)
library(RColorBrewer)
install.packages('rx')
fancyRpartPlot(dt_fit)
summary(dt_fit)

dt_fit$variable.importance

plot(dt_fit$variable.importance,names="variables")

pred<- predict(dt_fit,dttestdata,type = "class")
rp<-
  predict(dt_fit,dttestdata,type = "prob")
pred

library(gplots)
library(ROCR)
library(psych)
roc_pred <- prediction(rp[,2],dttestdata$SeriousDlqin2yrs)
plot(performance(roc_pred, measure="tpr", x.measure="fpr"),colorize=TRUE,main="ROC CURVE")
abline(0,1,lty=2)

auc.tmp <- performance(roc_pred,"auc"); auc <- as.numeric(auc.tmp@y.values)
auc.tmp
plot(performance(roc_pred, measure="lift", x.measure="rpp"), colorize=TRUE)
plot(performance(roc_pred, measure="sens", x.measure="spec"), colorize=TRUE)
plot(performance(roc_pred, measure="prec", x.measure="rec"), colorize=TRUE)

t<- table(predictions = pred, actual= dttestdata$SeriousDlqin2yrs)
t
pred
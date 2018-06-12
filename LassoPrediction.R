# Set seed to enable replication of the results
set.seed(1203)

## Load packeges
#install.packages("mlr")
library(mlr)
library(here)

## (1) Read and clean data
train     <- read.csv(here("Data", "train.csv"))
test      <- read.csv(here("Data", "test.csv"))
test$medv <- 0
testID    <- test$ID
train$ID  <- test$ID <- NULL

## (2) Define the task
trainTask <- makeRegrTask(data = train, target = "medv")
testTask  <-makeRegrTask(data = test, target = "medv")

## (3) Preprocessing
train$rad  <- as.factor(train$rad)
test$rad   <- as.factor(test$rad)
trainTask  <- createDummyFeatures(trainTask)
testTask   <- createDummyFeatures(testTask)
trainTask  <- normalizeFeatures(trainTask)
testTask   <- normalizeFeatures(testTask)

## (4) Define the learner
lrn <- makeLearner("regr.glmnet")

## (5) Tune hyperparameter (s: lambda)
ps <- makeParamSet(
  makeNumericParam("s", lower = -5, upper = 2, trafo = function(x) 10^x)
)
ctrl  <- makeTuneControlRandom(maxit = 100L)
rdesc <- makeResampleDesc("CV", iters = 10L)
res   <- tuneParams(lrn, trainTask, rdesc, measures = rmse, par.set = ps, control = ctrl)

## (6) Fit the model
lrn    <- setHyperPars(makeLearner("regr.glmnet"), par.vals = res$x)
model  <-  train(lrn, trainTask)

## (7) Make predictions
pred   <- predict(model, testTask)
preds  <- pred$data$response
table(preds)

## (8) create submission file
submission       <- data.frame(ID = testID)
submission$medv  <- preds
write.csv(submission, "Submissions/LassoSubmission.csv", row.names = FALSE)
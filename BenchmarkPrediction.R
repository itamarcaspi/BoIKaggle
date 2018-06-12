# Set seed to enable replication of the results
set.seed(1203)

## Load packeges
#install.packages("mlr")
#install.packages("here")
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
testTask  <- makeRegrTask(data = test, target = "medv")

## (3) Preprocessing
train$rad <-  as.factor(train$rad)
test$rad  <-  as.factor(test$rad)
trainTask <-  createDummyFeatures(trainTask)
testTask  <-  createDummyFeatures(testTask)

## (4) Define the learner (OLS)
lrn <- makeLearner("regr.lm")

## (5) Fit the model
model <- train(lrn, trainTask)

## (6) Make predictions
pred  <- predict(model, testTask)
preds <- pred$data$response
table(preds)

## (7) create submission file
submission       <- data.frame(ID = testID)
submission$medv  <- preds
write.csv(submission, "Submissions/BenchmarkSubmission.csv", row.names = FALSE)
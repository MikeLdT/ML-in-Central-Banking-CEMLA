########### Course Machine Learning and Central Banking 2022 #########
########### Prof. Gabriela Alves Werb, Sebastian Seltmann ############



#### Load Necessary Packages ####

# Data Handling and Plotting
suppressPackageStartupMessages({
  library(data.table)
  library(Hmisc)
  library(ggplot2)
  library(colorspace) 
  library(dplyr)
  library(scales)
  library(plotmo)
  library(rpart.plot)
  library(RColorBrewer)})

# Regression + Machine Learning + Parallel Computing
suppressPackageStartupMessages({
  library(rpart)
  library(randomForest)
  library(gbm)
  library(xgboost)
  library(caret)
  library(e1071)
  library(glmnet)
  library(adabag)
  library(doParallel)
  library(parallelMap)
  library(partykit)
  library(grf)
  library(e1071)})

# Model Evaluation
suppressPackageStartupMessages({
  library(caret)
  library(e1071)
  library(pROC)
  library(PRROC)})


#### Load Loans Data Set ####

# In RStudio Cloud your working directory is /cloud/project

loans <- fread("loans.csv", fill=TRUE, na.strings=c(""," ","NA"))

# Take a look at the data set
describe(loans) # this describes all variables
describe(loans$default) # this is our target variable

##### Transform all Character Variables to Factors ####

loans <-loans %>% mutate_if(sapply(loans, is.character), as.factor) %>% data.table()


##### Split data set into Train, Validation and Test Subsamples ####

# 60% training, 20% validation, 20% test

# Use stratified sampling, so that all levels of the categorical variables are present in each subsample
set.seed(1234) # set random seed for reproducibility
# randomly select 60% of the observations for training
loans_train <- loans %>% group_by(grade,sub_grade,emp_length,home_ownership,purpose,addr_state,
                                  verification_status) %>% sample_frac(.6, replace=FALSE) %>% data.table() 

# make a temporary data set for the remaining 40%
loans_temp <- data.table(anti_join(loans,loans_train)) 

# randomly select half of the remaining observations for validation (~50%*40% = ~20%)
loans_valid <- loans_temp %>% group_by(grade,sub_grade,emp_length,home_ownership,purpose,addr_state,
                                       verification_status) %>% sample_frac(.51, replace=FALSE) %>% data.table() 

#20% test: the remaining 20% observations
loans_test <- data.table(anti_join(loans_temp,loans_valid)) 

# Compare data sets
describe(loans$default)
describe(loans_train$default)
describe(loans_valid$default)
describe(loans_test$default)

# final shares: 67.5%, 21.5%, 11%
nrow(loans_train)/nrow(loans)
nrow(loans_valid)/nrow(loans)
nrow(loans_test)/nrow(loans)

# remove temporary data set and full data set
rm(loans_temp, loans)

##### Alternative: Use a small version of the data for faster computation ####
set.seed(1234)
loans_train_small <- loans_train %>% sample_frac(.1, replace=FALSE) %>% data.table()


######### Day 1 Methods ##########

##### Estimate a Simple Logistic Regression ####
# Transform the dependent variable to 0/1

#loans_train$default_num <- ifelse(loans_train$default=="yes",1,0) # version with the entire training data
loans_train_small$default_num <- ifelse(loans_train_small$default=="yes",1,0) # version with the "small" training data

# Run a logistic regression using the numeric dependent variable 

# logistic <- glm(default_num~., # regress default_num on all other variables
#               data=loans_train[,-"default"], # version with the entire training data excluding the original default variable
#               family=binomial) # logit link

logistic <- glm(default_num~., # regress default_num on all other variables
                data=loans_train_small[,-"default"], # version with the "small" training data excluding the original default variable
                family=binomial) # logit link

summary(logistic) 
# not a really good model, plaged by multicollinearity!
# also, all observations with missings are dropped

# Generate predictions in the train and validation data

### For the Train Data
# Make table for each data set (we will fill all other predictions there later on)

# Predicted probabilities
#pred_train <- data.table(logit_prob=predict(logistic,type="response")) # version with the entire training data  

pred_train_small <- data.table(logit_prob=predict(logistic,type="response")) # version with the "small" training data

# Rule: if predicted probability >=0.5, default otherwise no-default
#pred_train$logit_class <- ifelse(pred_train$logit_prob>=0.5,"yes","no") # version with the entire training data

pred_train_small$logit_class <- ifelse(pred_train_small$logit_prob>=0.5,"yes","no") # version with the "small" training data

### For the Validation Data

# Make table for each data set (we will fill all other predictions there later on)
pred_valid <- data.table(logit_prob=predict(logistic,newdata=loans_valid,type="response")) # predicted probabilities

# Rule: if predicted probability >=0.5, default otherwise no-default
pred_valid$logit_class <- ifelse(pred_valid$logit_prob>=0.5,"yes","no")

##### Confusion Matrix ####

### For the Train Data

# conf_logit_train <- confusionMatrix(as.factor(pred_train$logit_class),as.factor(loans_train$default), # version with the entire training data
#                                   positive="yes", # which value is what we're trying to predict? Here, default yes
#                                   dnn = c("Prediction", "Actual Data"))

conf_logit_train <- confusionMatrix(as.factor(pred_train_small$logit_class),as.factor(loans_train_small$default), # version with the "small" training data
                                    positive="yes", # which value is what we're trying to predict? Here, default yes
                                    dnn = c("Prediction", "Actual Data"))
conf_logit_train

### For the Validation Data

conf_logit_valid <- confusionMatrix(as.factor(pred_valid$logit_class),as.factor(loans_valid$default), 
                                    positive="yes", # which value is what we're trying to predict? Here, default yes
                                    dnn = c("Prediction", "Actual Data"))
conf_logit_valid

##### Model Evaluation Measures ####

# Get each measure from the stored values in the confusion matrix

### For the Train Data

#Accuracy
conf_logit_train$overall["Accuracy"]

#Precision
conf_logit_train$byClass["Precision"]

#Recall or Sensitivity
conf_logit_train$byClass["Recall"]
conf_logit_train$byClass["Sensitivity"]

# Specificity
conf_logit_train$byClass["Specificity"]

#F1 Score
conf_logit_train$byClass["F1"]

### For the Validation Data

#Accuracy
conf_logit_valid$overall["Accuracy"]

#Precision
conf_logit_valid$byClass["Precision"]

#Recall or Sensitivity
conf_logit_valid$byClass["Recall"]
conf_logit_valid$byClass["Sensitivity"]

# Specificity
conf_logit_valid$byClass["Specificity"]

#F1 Score
conf_logit_valid$byClass["F1"]


### Compute some of them "by hand"

# Accuracy
(conf_logit_valid$table[1,1]+conf_logit_valid$table[2,2])/sum(conf_logit_valid$table)
conf_logit_valid$overall["Accuracy"]

# Kappa = (accuracy - chance)/(1 - chance)
# Chance: use Bayes' rule to find the prob. of getting this accuracy by chance (multiply marginal probabilities)
chance=(conf_logit_valid$table[1,1]+conf_logit_valid$table[1,2])/sum(conf_logit_valid$table) * 
  (conf_logit_valid$table[1,1]+conf_logit_valid$table[2,1])/sum(conf_logit_valid$table) + # this is the probability of getting true negative by chance
  (conf_logit_valid$table[2,1]+conf_logit_valid$table[2,2])/sum(conf_logit_valid$table) * 
  (conf_logit_valid$table[1,2]+conf_logit_valid$table[2,2])/sum(conf_logit_valid$table) # this is the probability of getting true positive by chance

(conf_logit_valid$overall["Accuracy"]-chance)/(1-chance)
conf_logit_valid$overall["Kappa"]

# F1 score
2*(conf_logit_valid$byClass["Sensitivity"]*conf_logit_valid$byClass["Precision"])/
  (conf_logit_valid$byClass["Sensitivity"]+conf_logit_valid$byClass["Precision"])
conf_logit_valid$byClass["F1"]

##### ROC Curve  ####

### For Validation Data

roc_logit <- roc(response=loans_valid$default, # actual data (default yes/no)
                 predictor=pred_valid$logit_prob) # predicted probabilities of default (logit model)

# Plot the Curve
plot(roc_logit)

# Calculate the AUC
roc_logit$auc

# Get coordinates for best threshold
coords(roc_logit, "best", ret=c("threshold", "specificity", "1-npv"))

# Plot the Curve ggplot2 prettier graphics
ggroc(roc_logit) + scale_x_reverse(name ="True Negative Rate (Specificity)") +
  scale_y_continuous(name="True Positive Rate (Sensitivity)") + 
  theme(legend.position="none",
        axis.title = element_text(color="#666666", face="bold", size=14),
        axis.text=element_text(size=10,face="bold")) +
  geom_segment(aes(x = 1, y = 0, xend = 0, yend = 1), color="black") +
  annotate("text", x = .95, y = .6, label = sprintf("AUC Logit = %0.2f", roc_logit$auc),hjust=0, size=4,
           color="darkblue", fontface=2) 

# Same result using the PRROC package (but inputs are less intuitive)

roc_logit_2 <- roc.curve(scores.class0 = pred_valid[loans_valid$default=="yes",logit_prob],# these are the predictions for default=yes, the positive class
                         scores.class1 = pred_valid[loans_valid$default=="no",logit_prob], # these are the predictions for default=no, the negative class
                         curve=TRUE)

plot(roc_logit_2)

# We can use the results from roc.curve to make a prettier plot in ggplot2 too

ggplot(data.frame(roc_logit_2$curve),aes(x=X1,y=X2,color=X3)) +
  geom_line(size=2) + labs(x="1 - True Negative Rate (Specificity)",y="True Positive Rate (Sensitivity)",colour="Threshold") +
  annotate("text", x = .5, y = .6, label = sprintf("AUC-ROC Logit = %0.2f", roc_logit_2$auc),hjust=0, size=4,
           color="darkblue", fontface=2) +
  scale_colour_gradient2(low="yellow", mid="green",high="blue") +
  theme(axis.title = element_text(color="#666666", face="bold", size=18),
        axis.text=element_text(size=16,face="bold"))

##### PR Curve ####

### For the Validation Data
pr_logit <- pr.curve(scores.class0=pred_valid[loans_valid$default=="yes",logit_prob],# these are the predictions for default=no
                     scores.class1 = pred_valid[loans_valid$default=="no",logit_prob], # these are the predictions for default=yes
                     curve=TRUE)
plot(pr_logit)

# We can use the results from pr.curve to make a prettier plot in ggplot2 too

ggplot(data.frame(pr_logit$curve),aes(x=X1,y=X2,color=X3)) +
  geom_line(size=2) + labs(x="Recall",y="Precision",colour="Threshold") +
  annotate("text", x = .5, y = .6, label = sprintf("AUC-PR Logit = %0.2f", pr_logit$auc.integral),hjust=0, size=4,
           color="darkblue", fontface=2) +
  scale_colour_gradient2(low="yellow", mid="green",high="blue") +
  theme(axis.title = element_text(color="#666666", face="bold", size=18),
        axis.text=element_text(size=16,face="bold"))


#### Day 2 Methods #####


##### Decision Trees (CART) ####

# rpart_tree_simple <- rpart(formula=default ~ . , # Dependent variable and independent variables ("." for all variables in the data)
#                          data=loans_train[,-"default_num"], # Version with the entire training data
#                          na.action = na.rpart, # What should be done with missings?
#                          method="class", # Decision problem (here: categorical variable, so "classification")
#                          parms = list(split = 'gini'), # Gini Index as split criterion
#                          control=rpart.control(minsplit=20, # Minimal number of obs. in a node before a split is attempted
#                                                cp = 0.01, # Complexity param.: min. increase in split criterion to grow the tree
#                                                usesurrogate = 2, # How many surrogates are used
#                                                maxdepth = 10)) #max. tree depth

rpart_tree_simple <- rpart(formula=default ~ . , # Dependent variable and independent variables ("." for all variables in the data)
                           data=loans_train_small[,-"default_num"], # Version with the "small" training data
                           na.action = na.rpart, # What should be done with missings?
                           method="class", # Decision problem (here: categorical variable, so "classification")
                           parms = list(split = 'gini'), # Gini Index as split criterion
                           control=rpart.control(minsplit=20, # Minimal number of obs. in a node before a split is attempted
                                                 cp = 0.01, # Complexity param.: min. increase in split criterion to grow the tree
                                                 usesurrogate = 2, # How many surrogates are used
                                                 maxdepth = 10)) #max. tree depth


# See model summary
summary(rpart_tree_simple)

# See pretty plot of the tree  
rpart.plot(rpart_tree_simple, cex = 0.8)

#### Decrease in node impurity from first split  

index_all <- 1-(68773/333301)^2-(264528/333301)^2
index_left <- 1-(39918/275913)^2-(235995/275913)^2
index_right <- 1-(28533/57388)^2-(28855/57388)^2
delta_index <- index_all-(275913/333301)*index_left-(57388/333301)*index_right
delta_index*333301 # This is the "improve" measure reported for the first split
rpart_tree_simple$splits[1,"improve"] 

#### Predictions for the Train Data

rpart.plot(rpart_tree_simple,box.palette = "GnRd")

# pred_train <- data.table(tree_prob=predict(rpart_tree_simple,type="prob")[,2]) # Probabilities
# pred_train$tree_class <- predict(rpart_tree_simple, type="class") # Classes (default yes/no)

# conf_tree_train <- confusionMatrix(as.factor(pred_train$tree_class),as.factor(loans_train$default), # Version with the entire training data
#                                    positive="yes", # which value is what we're trying to predict? Here, default yes
#                                    dnn = c("Prediction", "Actual Data"))

pred_train_small <- data.table(tree_prob=predict(rpart_tree_simple,type="prob")[,2]) # Probabilities
pred_train_small$tree_class <- predict(rpart_tree_simple, type="class") # Classes (default yes/no)

conf_tree_train <- confusionMatrix(as.factor(pred_train_small$tree_class),as.factor(loans_train_small$default), # Version with the "small" training data
                                   positive="yes", # which value is what we're trying to predict? Here, default yes
                                   dnn = c("Prediction", "Actual Data"))
conf_tree_train


#### Predictions for the Validation Data

pred_valid$tree_prob<-predict(rpart_tree_simple, newdata=loans_valid,type="prob")[,2] # Probabilities
pred_valid$tree_class<-predict(rpart_tree_simple, newdata=loans_valid,type="class") # Classes (default yes/no)

conf_tree_valid <- confusionMatrix(as.factor(pred_valid$tree_class),as.factor(loans_valid$default), 
                                   positive="yes", # which value is what we're trying to predict? Here, default yes
                                   dnn = c("Prediction", "Actual Data"))
conf_tree_valid

conf_tree_valid$overall["Accuracy"]
conf_tree_valid$byClass["Precision"]
conf_tree_valid$byClass["Recall"]
conf_tree_valid$byClass["F1"]

##### Missings: missings in total_pymnt #####

# Now generate 30% missings in total_pymnt without removing loan_amnt (best surrogate together with funded_amnt)

set.seed(1234) # random number generator (for reproducibility)

#sample_mis <- loans_train # create new data set (copy the original one)
sample_mis <- loans_train_small # create new data set (copy the original one)

mis_id <- sample(1:nrow(sample_mis), size = 0.3*nrow(sample_mis), replace=FALSE) # select obs. that will have the missings
sample_mis[mis_id,"total_pymnt"]<-NA

# Grow the tree with the new data set (30% missings in total_pymnt) #

rpart_tree_miss <- rpart(formula=default ~ . , # Dependent variable and independent variables ("." for all variables in the data)
                         data=sample_mis[,-"default_num"], # Training data set with missings
                         na.action = na.rpart, # What should be done with missings? 
                         method="class", # Decision problem (here: categorical variable, so "classification") 
                         parms = list(split = 'gini'), # Gini Index as split criterion
                         control=rpart.control(minsplit=20, # Minimal number of obs. in a node before a split is attempted
                                               cp = 0.01, # Complexity param.: min. increase in split criterion to grow the tree 
                                               usesurrogate = 2, # How many surrogates are used
                                               maxdepth = 3)) #max. tree depth

# See the results
summary(rpart_tree_miss)

# Plot the tree
rpart.plot(rpart_tree_miss,cex=0.8)

# Note how total_pymnt and loan_amnt contain similar information, as the payments depend on the loan amount 
# If we generate missings in total_pymnt, the tree chooses loan_amnt for the primary split instead. 

##### Example of Overfitting - No stop criterion ####

# rpart_tree_overfit <- rpart(formula=default ~ . , # Dependent variable and independent variables ("." for all variables in the data)
#                             data=loans_train[,-"default_num"], # Version with the entire training data
#                             na.action = na.rpart, # What should be done with missings?
#                             method="class", # Decision problem (here: categorical variable, so "classification")
#                             parms = list(split = 'gini'), # Gini Index as split criterion
#                             control=rpart.control(minsplit=1, # Minimal number of obs. in a node before a split is attempted
#                                                   cp = 0, # Complexity param.: min. increase in split criterion to grow the tree
#                                                   usesurrogate = 2, # How many surrogates are used
#                                                   maxdepth = 30)) #max. tree depth


rpart_tree_overfit <- rpart(formula=default ~ . , # Dependent variable and independent variables ("." for all variables in the data)
                            data=loans_train_small[,-"default_num"], # Version with the "small" training data
                            na.action = na.rpart, # What should be done with missings?
                            method="class", # Decision problem (here: categorical variable, so "classification")
                            parms = list(split = 'gini'), # Gini Index as split criterion
                            control=rpart.control(minsplit=1, # Minimal number of obs. in a node before a split is attempted
                                                  cp = 0, # Complexity param.: min. increase in split criterion to grow the tree
                                                  usesurrogate = 2, # How many surrogates are used
                                                  maxdepth = 30)) #max. tree depth

rpart.plot(rpart_tree_overfit,cex=0.08)
summary(rpart_tree_overfit)

# Goodness of fit in the train data

# pred_train$rpart_overfit_prob <- predict(rpart_tree_overfit,type="prob")[,2] # Probabilities
# pred_train$rpart_overfit_class <- predict(rpart_tree_overfit, type="class") # Classes (default yes/no)

# conf_tree_overfit_train <- confusionMatrix(as.factor(pred_train$rpart_overfit_class),
#                                            as.factor(loans_train$default), # Version with the entire training data
#                                            positive="yes", # which value is what we're trying to predict? Here, default yes
#                                            dnn = c("Prediction", "Actual Data"))

pred_train_small$rpart_overfit_prob <- predict(rpart_tree_overfit,type="prob")[,2] # Probabilities
pred_train_small$rpart_overfit_class <- predict(rpart_tree_overfit, type="class") # Classes (default yes/no)

conf_tree_overfit_train <- confusionMatrix(as.factor(pred_train_small$rpart_overfit_class),
                                           as.factor(loans_train_small$default), # Version with the "small" training data
                                           positive="yes", # which value is what we're trying to predict? Here, default yes
                                           dnn = c("Prediction", "Actual Data"))
conf_tree_overfit_train

# We seem to have a perfect model!!

# But....

# Goodness of fit in the validation data

pred_valid$rpart_overfit_prob <- predict(rpart_tree_overfit, newdata=loans_valid, type="prob")[,2] # Probabilities
pred_valid$rpart_overfit_class <- predict(rpart_tree_overfit, newdata=loans_valid, type="class") # Classes (default yes/no)

conf_tree_overfit_valid <- confusionMatrix(as.factor(pred_valid$rpart_overfit_class),as.factor(loans_valid$default), 
                                           positive="yes", # which value is what we're trying to predict? Here, default yes
                                           dnn = c("Prediction", "Actual Data"))
conf_tree_overfit_valid


##### Conditional Inference Trees #####

settings <- ctree_control(testtype = "Bonferroni", # Test Statistic - correction for multiple testing 
                          minsplit=20, # Min. Size to Perform a Split
                          minbucket=10, # Min. Size per Terminal Node
                          maxdepth=10, # Max. Tree Depth
                          maxsurrogate = 2, # Number of Surrogate Splits to Evaluate (default: 0, no surrogates)
                          mincriterion=0.99) # Significance Level for Permutation Tests (1- p-value)

# ctree <- ctree(default ~ ., # Dependent Variable and Independent Variables ("." for all variables in the data)
#                data = loans_train[,-c("default_num", # Version with the entire training data
#                                             "sub_grade","addr_state")], # Remove (alternatively: consolidate to 30 levels each)
#                control = settings) # Add Settings

ctree <- ctree(default ~ ., # Dependent Variable and Independent Variables ("." for all variables in the data)
               na.action = na.pass, # What should be done with missings?
               data = loans_train_small[,-c("default_num", # Version with the "small" training data 
                                            "sub_grade","addr_state")], # Remove (alternatively: consolidate to 30 levels each)
               control = settings) # Add Settings

windows()
plot(ctree, type="simple", gp = gpar(fontsize = 9)) # Plot the resulting tree


# Goodness of fit in the train data

# pred_train$ctree_prob <- predict(ctree,type="prob")[,2] # Probabilities
# pred_train$ctree_class <- predict(ctree, type="response") # Classes (default yes/no)
# 
# conf_ctree_train <- confusionMatrix(as.factor(pred_train$ctree_class),
#                                            as.factor(loans_train$default), # Version with the entire training data
#                                            positive="yes", # which value is what we're trying to predict? Here, default yes
#                                            dnn = c("Prediction", "Actual Data"))

pred_train_small$ctree_prob <- predict(ctree,type="prob")[,2] # Probabilities
pred_train_small$ctree_class <- predict(ctree, type="response") # Classes (default yes/no)

conf_ctree_train <- confusionMatrix(as.factor(pred_train_small$ctree_class),
                                    as.factor(loans_train_small$default), # Version with the "small" training data
                                    positive="yes", # which value is what we're trying to predict? Here, default yes
                                    dnn = c("Prediction", "Actual Data"))
conf_ctree_train

# Goodness of fit in the validation data

pred_valid$ctree_prob <- predict(ctree, newdata=loans_valid, type="prob")[,2] # Probabilities
pred_valid$ctree_class <- predict(ctree, newdata=loans_valid, type="response") # Classes (default yes/no)

conf_ctree_valid <- confusionMatrix(as.factor(pred_valid$ctree_class),as.factor(loans_valid$default), 
                                    positive="yes", # which value is what we're trying to predict? Here, default yes
                                    dnn = c("Prediction", "Actual Data"))
conf_ctree_valid

conf_ctree_valid$byClass["F1"]

##### Bagging Trees ####

# Set parameters for parallel computing
parallelStartSocket(cpus = detectCores()-1)

set.seed(1234)

# bagged_trees <- bagging(default ~., # variables in the model
#                         data=loans_train[,-"default_num"], # data: version with the entire training data
#                         mfinal=5, # number of bagged trees
#                         par=TRUE, # run in parallel?
#                         control=rpart.control(maxdepth=3, minsplit=20)) # pruning parameters


bagged_trees <- bagging(default ~., # variables in the model
                        data=loans_train_small[,-"default_num"], # data: Version with the "small" training data
                        mfinal=5, # number of bagged trees
                        par=TRUE, # run in parallel?
                        control=rpart.control(maxdepth=3, minsplit=20)) # pruning parameters

parallelStop()

### Predictions for the Train Data
#pred_bagged_trees_train <- predict.bagging(bagged_trees, newdata = loans_train[,-"default_num"]) # Version with the entire training data

# pred_train$bagging_prob <- pred_bagged_trees_train$prob[,2]
# pred_train$bagging_class <- pred_bagged_trees_train$class

pred_bagged_trees_train <- predict.bagging(bagged_trees, newdata = loans_train_small[,-"default_num"]) # Version with the "small" training data

pred_train_small$bagging_prob <- pred_bagged_trees_train$prob[,2]
pred_train_small$bagging_class <- pred_bagged_trees_train$class


# Goodness of fit in the training data

# conf_bagged_trees_train <- confusionMatrix(as.factor(pred_train$bagging_class),as.factor(loans_train$default),  # Version with the entire training data
#                                            positive="yes", # which value is what we're trying to predict? Here, default yes
#                                            dnn = c("Prediction", "Actual Data"))

conf_bagged_trees_train <- confusionMatrix(as.factor(pred_train_small$bagging_class),as.factor(loans_train_small$default), # Version with the "small" training data
                                           positive="yes", # which value is what we're trying to predict? Here, default yes
                                           dnn = c("Prediction", "Actual Data"))
conf_bagged_trees_train


### Predictions for the Validation Data

pred_bagged_trees_valid <- predict.bagging(bagged_trees, newdata=loans_valid)

pred_valid$bagging_prob <- pred_bagged_trees_valid$prob[,2]
pred_valid$bagging_class <- pred_bagged_trees_valid$class


# Goodness of fit in the validation data
conf_bagged_trees_valid <- confusionMatrix(as.factor(pred_valid$bagging_class),as.factor(loans_valid$default), 
                                           positive="yes", # which value is what we're trying to predict? Here, default yes
                                           dnn = c("Prediction", "Actual Data"))
conf_bagged_trees_valid


### Cross-Validation

parallelStartSocket(cpus = detectCores()-1)

set.seed(1234)

# bagged_trees_cv <- bagging.cv(default ~., # variables in the model
#                               data=loans_train[,-"default_num"], # Version with the entire training data
#                               mfinal=5, # number of bagged trees
#                               par=TRUE, # run in parallel?
#                               v=10, # 10-fold cross validation
#                               control=rpart.control(maxdepth=3, minsplit=20)) # pruning parameters


bagged_trees_cv <- bagging.cv(default ~., # variables in the model
                              data=loans_train_small[,-"default_num"], # Version with the "small" training data
                              mfinal=5, # number of bagged trees
                              par=TRUE, # run in parallel?
                              v=10, # 10-fold cross validation
                              control=rpart.control(maxdepth=3, minsplit=20)) # pruning parameters

parallelStop()


##### Boosting Trees ####


# boosted_trees <- boosting(default ~., # variables in the model
#                           data=loans_train[,-"default_num"], # data: version with the entire training data
#                           mfinal=5, # number of boosted trees
#                           coeflearn = "Freund", # Rule to update the weights (other options available too)
#                           control=rpart.control(maxdepth=3, minsplit=20)) # pruning parameters


boosted_trees <- boosting(default ~., # variables in the model
                          data=loans_train_small[,-"default_num"], # data: Version with the "small" training data
                          mfinal=5, # number of boosted trees
                          coeflearn = "Freund", # Rule to update the weights (other options available too)
                          control=rpart.control(maxdepth=3, minsplit=20)) # pruning parameters


### Predictions for the Train Data
#pred_boosted_trees_train <- predict.boosting(boosted_trees, newdata = loans_train[,-"default_num"]) # Version with the entire training data

# pred_train$boosting_prob <- pred_boosted_trees_train$prob[,2]
# pred_train$boosting_class <- pred_boosted_trees_train$class

pred_boosted_trees_train <- predict.boosting(boosted_trees, newdata = loans_train_small[,-"default_num"]) # Version with the "small" training data

pred_train_small$boosting_prob <- pred_boosted_trees_train$prob[,2]
pred_train_small$boosting_class <- pred_boosted_trees_train$class


# Goodness of fit in the training data

# conf_boosted_trees_train <- confusionMatrix(as.factor(pred_train$boosting_class),as.factor(loans_train$default), # Version with the entire training data
#                                             positive="yes", # which value is what we're trying to predict? Here, default yes
#                                             dnn = c("Prediction", "Actual Data"))

conf_boosted_trees_train <- confusionMatrix(as.factor(pred_train_small$boosting_class),as.factor(loans_train_small$default), # Version with the "small" training data
                                            positive="yes", # which value is what we're trying to predict? Here, default yes
                                            dnn = c("Prediction", "Actual Data"))
conf_boosted_trees_train


### Predictions for the Validation Data

pred_boosted_trees_valid <- predict.boosting(boosted_trees, newdata=loans_valid)

pred_valid$boosting_prob <- pred_boosted_trees_valid$prob[,2]
pred_valid$boosting_class <- pred_boosted_trees_valid$class


# Goodness of fit in the validation data
conf_boosted_trees_valid <- confusionMatrix(as.factor(pred_valid$boosting_class),as.factor(loans_valid$default), 
                                            positive="yes", # which value is what we're trying to predict? Here, default yes
                                            dnn = c("Prediction", "Actual Data"))
conf_boosted_trees_valid


### Cross-Validation

parallelStartSocket(cpus = detectCores()-1)

set.seed(1234)

# Version with the entire training data
# boosted_trees_cv <- boosting.cv(default ~., # variables in the model
#                                 data=loans_train[,-"default_num"], # Version with the entire training data
#                                 mfinal=5, # number of boosted trees
#                                 par=TRUE, # run in parallel?
#                                 v=10, # 10-fold cross validation
#                                 coeflearn = "Freund", # Rule to update the weights (other options available too)
#                                 control=rpart.control(maxdepth=3, minsplit=20)) # pruning parameters

# Version with the "small" training data
boosted_trees_cv <- boosting.cv(default ~., # variables in the model
                                data=loans_train_small[,-"default_num"], # Version with the "small" training data
                                mfinal=5, # number of boosted trees
                                par=TRUE, # run in parallel?
                                v=10, # 10-fold cross validation
                                coeflearn = "Freund", # Rule to update the weights (other options available too)
                                control=rpart.control(maxdepth=3, minsplit=20)) # pruning parameters

parallelStop()


### Day 3 Methods #####


##### Random Forests ####

# Random forests in R needs complete observations (no missings)
# We can impute missing values using random forest proximities 
# set.seed(1234)
# train_imputed<-rfImpute(default~.,data=loans_train[,-"default_num"],iter=5,ntree=500)
# valid_imputed<-rfImpute(default~.,data=loans_valid,iter=5,ntree=500)
# test_imputed<-rfImpute(default~.,data=loans_test,iter=5,ntree=500)


# Set parameters for parallel computing
parallelStartSocket(cpus = detectCores()-1)

set.seed(1234)

# Version with the entire training data
# rf<- randomForest::randomForest(formula=default~., # variables
#                                 data=loans_train[,-"default_num"], # data set without missings, Version with the entire training data
#                                 replace=TRUE, # sampling of observations with replacement (Y/N)
#                                 ntree=500, # number of trees in the forest
#                                 mtry=6, # number of explanatory variables considered for a split
#                                 nodesize=1, # min. terminal node size
#                                 maxnodes=NULL, # max. number of terminal nodes
#                                 importance=TRUE) # calculate importance measure (Y/N)

# Version with the "small" training data
rf <- randomForest::randomForest(formula=default~., # variables
                                 data=loans_train_small[,-"default_num"], # data set without missings, Version with the "small" training data
                                 replace=TRUE, # sampling of observations with replacement (Y/N)
                                 ntree=50, # number of trees in the forest
                                 mtry=6, # number of explanatory variables considered for a split
                                 nodesize=1, # min. terminal node size
                                 maxnodes=NULL, # max. number of terminal nodes
                                 importance=TRUE) # calculate importance measure (Y/N)


print(rf)

plot(rf)

plot(margin(rf))

varImpPlot(rf)

#### Predictions for the Train Data

# Version with the entire training data
# pred_train$rf_prob <- predict(rf,type="prob")[,2] # Probabilities
# pred_train$rf_class <- predict(rf, type="class") # Classes (default yes/no)

# Version with the "small" training data
pred_train_small$rf_prob <- predict(rf,type="prob")[,2] # Probabilities
pred_train_small$rf_class <- predict(rf, type="class") # Classes (default yes/no)


# Goodness of fit in the training data

# conf_rf_train <- confusionMatrix(as.factor(pred_train$rf_class),as.factor(loans_train$default), # Version with the entire training data
#                                  positive="yes", # which value is what we're trying to predict? Here, default yes
#                                  dnn = c("Prediction", "Actual Data"))

conf_rf_train <- confusionMatrix(as.factor(pred_train_small$rf_class),as.factor(loans_train_small$default), # Version with the "small" training data
                                 positive="yes", # which value is what we're trying to predict? Here, default yes
                                 dnn = c("Prediction", "Actual Data"))
conf_rf_train


### Predictions for the Validation Data

pred_valid$rf_prob <- predict(rf, newdata=loans_valid,type="prob")[,2] # Probabilities
pred_valid$rf_class <- predict(rf, newdata=loans_valid,type="class") # Classes (default yes/no)

# Goodness of fit in the validation data

conf_rf_valid <- confusionMatrix(as.factor(pred_valid$rf_class),as.factor(loans_valid$default), 
                                 positive="yes", # which value is what we're trying to predict? Here, default yes
                                 dnn = c("Prediction", "Actual Data"))
conf_rf_valid


##### Causal Random Forests ####

# Check for the effect of income verification status (attention: not an exogenous variable, so doutbful as a "causal" effect)

### Create "treatment variable"

#loans_train$verification_status_new <- ifelse(loans_train$verification_status=="Not Verified",0,1) # Version with the entire training data

loans_train_small$verification_status_new <- ifelse(loans_train_small$verification_status=="Not Verified",0,1) # Version with the "small" training data

loans_valid$verification_status_new <- ifelse(loans_valid$verification_status=="Not Verified",0,1) 


### Transform covariate x matrix, factors must be converted to dummies

# Version with the entire training data
# x_matrix_train <- as.data.frame(model.matrix(as.formula(paste("~ ",
#                                                               paste(colnames(loans_train[,-c("default","default_num","verification_status","verification_status_new")]),
#                                                                     collapse = "+"), "-1")), #-1: no intercept column create
#                                              data = loans_train))

# Version with the "small" training data
x_matrix_train <- as.data.frame(model.matrix(as.formula(paste("~ ", 
                                                              paste(colnames(loans_train_small[,-c("default","default_num","verification_status","verification_status_new")]), 
                                                                    collapse = "+"), "-1")), #-1: no intercept column create
                                             data = loans_train_small))

x_matrix_valid <- as.data.frame(model.matrix(as.formula(paste("~ ", 
                                                              paste(colnames(loans_train_small[,-c("default","default_num","verification_status","verification_status_new")]), 
                                                                    collapse = "+"), "-1")), #-1: no intercept column create
                                             data = loans_valid))

### Causal Random Forest

# Version with the entire training data
# cf <- grf::causal_forest(X = x_matrix_train, # Covariates in the model, Version with the entire training data
#                          Y = loans_train$default_num, # Outcome to predict (default 0/1, must be numeric)
#                          W = loans_train$verification_status_new, # Treatment variable (numeric binary or continuous, no factors)
#                          mtry=6, # number of explanatory variables considered for a split
#                          num.trees = 100) # Number of trees in the forest

# Version with the "small" training data
cf <- grf::causal_forest(X = x_matrix_train, # Covariates in the model, Version with the "small" training data
                         Y = loans_train_small$default_num, # Outcome to predict (default 0/1, must be numeric)
                         W = loans_train_small$verification_status_new, # Treatment variable (numeric binary or continuous, no factors)
                         mtry=6, # number of explanatory variables considered for a split
                         num.trees = 100) # Number of trees in the forest
cf

variable_importance(cf) %>% plot()

### Important: predict does not predict default yes/no - it predicts the conditional average treatment effect (CATE) using the OOB predictions
### Therefore, these predictions are not comparable with our previous results from other models.

pred_treat_effect <- data.table(treat_pred = predict(cf)$predictions)

ggplot(pred_treat_effect, aes(x=treat_pred)) + geom_histogram() + xlab("Predicted CATE") + ylab("Number of Observations")

### Average "treatment" effect (all observations) - CATE

average_treatment_effect(cf, target.sample = "all")

### Average "treatment" effect (only on the treated observations)

average_treatment_effect(cf, target.sample = "treated")


##### Gradient Boosting ####


# Version with the full data set
# set.seed(1234)
# gradient_boosting <- gbm(formula=default_num~., # here we also need our dep. variable to be 0/1
#                          data=loans_train[,-"default"], # train data (exclude the default yes/no variable)
#                          distribution="bernoulli",
#                          shrinkage=0.005, #shrinkage parameter (learning rate)
#                          interaction.depth=3, # max. tree depth (models interactions)
#                          n.trees=500, # number of trees
#                          bag.fraction=1, # If <1, uses bagged samples for each tree
#                          n.minobsinnode = 10, #min. number of observations in leafs
#                          cv.folds=10, # perform cross-validation? (Y/N)
#                          verbose=TRUE)

# Version with the "small" training data
set.seed(1234)
gradient_boosting <- gbm(formula=default_num~., # here we also need our dep. variable to be 0/1
                         data=loans_train_small[,-"default"], # data: loans_train for full version
                         distribution="bernoulli",
                         shrinkage=0.005, #shrinkage parameter (learning rate)
                         interaction.depth=3, # max. tree depth (models interactions)
                         n.trees=50, # number of trees
                         bag.fraction=0.9, # If <1, uses bagged samples for each tree
                         n.minobsinnode = 10, #min. number of observations in leafs
                         cv.folds=3, # perform cross-validation? (Y/N)
                         verbose=TRUE)


gradient_boosting

summary(gradient_boosting)

summary(gradient_boosting, cBars = 10,method = relative.influence)
summary(gradient_boosting, cBars = 10, method = permutation.test.gbm)

gbm.perf(gradient_boosting, method = "cv")

plot(gradient_boosting, i.var="loan_amnt", type="response")
plot(gradient_boosting, i.var="total_pymnt", type="response")
plot(gradient_boosting, i.var="installment", type="response")

### Visualize a single tree ###
summary(gradient_boosting,n.trees=1)

### Check performance ###

gbm.perf(gradient_boosting,plot.it=TRUE)

#### Predictions for the Train Data

# pred_train$gbm_prob <- predict(gradient_boosting,type="response") # Probabilities
# pred_train$gbm_class <- ifelse(pred_train$gbm_prob>=0.5,"yes","no") # Classes (default yes/no)

# conf_gbm_train <- confusionMatrix(as.factor(pred_train$gbm_class),as.factor(loans_train$default), #Version with the full data set
#                                   positive="yes", # which value is what we're trying to predict? Here, default yes
#                                   dnn = c("Prediction", "Actual Data"))

pred_train_small$gbm_prob<-predict(gradient_boosting,type="response") # Probabilities
pred_train_small$gbm_class<-ifelse(pred_train$gbm_prob>=0.5,"yes","no") # Classes (default yes/no)

conf_gbm_train <- confusionMatrix(as.factor(pred_train_small$gbm_class),as.factor(loans_train_small$default), # Version with the "small" training data
                                  positive="yes", # which value is what we're trying to predict? Here, default yes
                                  dnn = c("Prediction", "Actual Data"))
conf_gbm_train


### Predictions for the Validation Data

pred_valid$gbm_prob <- predict(gradient_boosting, newdata=loans_valid,type="response") # Probabilities
pred_valid$gbm_class <- ifelse(pred_valid$gbm_prob>=0.5,"yes","no") # Classes (default yes/no)

conf_gbm_valid <- confusionMatrix(as.factor(pred_valid$gbm_class),as.factor(loans_valid$default), 
                                  positive="yes", # which value is what we're trying to predict? Here, default yes
                                  dnn = c("Prediction", "Actual Data"))
conf_gbm_valid

conf_gbm_valid$overall["Accuracy"]
conf_gbm_valid$byClass["Precision"]
conf_gbm_valid$byClass["Recall"]
conf_gbm_valid$byClass["F1"]

##### Extreme Gradient Boosting ####

# Prepare sparse matrices for xgboost

# Version with the full data set
# train_xgb <- xgb.DMatrix(model.matrix.lm(default~.-1,loans_train[,-c("default_num")],na.action="na.pass"),
#                          label=loans_train$default_num)

# Version with the "small" training data
train_xgb_small <- xgb.DMatrix(model.matrix.lm(default~.-1,loans_train_small[,-c("default_num")],na.action="na.pass"),label=loans_train_small$default_num)

# Validation data
valid_xgb <- xgb.DMatrix(model.matrix.lm(default~.-1,loans_valid,na.action="na.pass"),label=as.numeric(loans_valid$default)-1)

## Version with the full data set
# set.seed(1234)
# xgb <- xgb.train(data = train_xgb, # train data set
#                  nrounds=500, # number of trees
#                  params = list(booster="gbtree", # use trees as the base model
#                                lambda=1, # L2 regularization term on weights
#                                eta = 0.3, # shrinkage hyperparameter
#                                max_depth = 6, # max. tree depth
#                                min_child_weight=1, #min. weight sum in child node
#                                subsample=1, # if <1, bagged samples for each tree
#                                colsample_bytree=1, # if <1, works like mtry in rf
#                                objective = "binary:logistic", # type of problem
#                                eval_metric = "error", # performance metric
#                                early_stopping_rounds=50, #early stopping criteria
#                                nthread = 4), # optional for parallel computing
#                  verbosity=2) # show information during model building

# Version with the "small" training data
set.seed(1234)
xgb <- xgb.train(data = train_xgb_small, # Version with the "small" training data
                 nrounds=50, # number of trees
                 params = list(booster="gbtree", # use trees as the base model
                               lambda=1, # L2 regularization term on weights
                               eta = 0.3, # shrinkage hyperparameter
                               max_depth = 6, # max. tree depth
                               min_child_weight=1, #min. weight sum in child node
                               subsample=1, # if <1, bagged samples for each tree
                               colsample_bytree=1, # if <1, works like mtry in rf
                               objective = "binary:logistic", # type of problem
                               eval_metric = "error", # performance metric
                               early_stopping_rounds=50, #early stopping criteria
                               nthread = 4), # optional for parallel computing
                 verbosity=2) # show information during model building
xgb.importance(model=xgb) %>% xgb.ggplot.importance(top_n=10)

xgb.ggplot.deepness(xgb)

# See how the median absolute leaf weight changes through the iterations (number of trees)
xgb.ggplot.deepness(xgb, which="med.weight")

#### Predictions for the Train Data

# Version with the full data set
# pred_train$xgb_prob<-predict(xgb,newdata=train_xgb,type="prob") # Probabilities
# pred_train$xgb_class<-ifelse(pred_train$xgb_prob>=0.5,"yes","no") # Classes (default yes/no)
# 
# conf_xgb_train <- confusionMatrix(as.factor(pred_train$xgb_class),as.factor(loans_train$default), # data: loans_train for full version
#                                   positive="yes", # which value is what we're trying to predict? Here, default yes
#                                   dnn = c("Prediction", "Actual Data"))

# Version with the "small" training data
pred_train_small$xgb_prob<-predict(xgb,newdata=train_xgb_small,type="prob") # Probabilities
pred_train_small$xgb_class<-ifelse(pred_train_small$xgb_prob>=0.5,"yes","no") # Classes (default yes/no)

conf_xgb_train <- confusionMatrix(as.factor(pred_train_small$xgb_class),as.factor(loans_train_small$default), # data: loans_train for full version
                                  positive="yes", # which value is what we're trying to predict? Here, default yes
                                  dnn = c("Prediction", "Actual Data"))

conf_xgb_train

### Predictions for the Validation Data

pred_valid$xgb_prob<-predict(xgb, newdata=valid_xgb,type="prob") # Probabilities
pred_valid$xgb_class<-ifelse(pred_valid$xgb_prob>=0.5,"yes","no") # Classes (default yes/no)

conf_xgb_valid <- confusionMatrix(as.factor(pred_valid$xgb_class),as.factor(loans_valid$default), 
                                  positive="yes", # which value is what we're trying to predict? Here, default yes
                                  dnn = c("Prediction", "Actual Data"))
conf_xgb_valid

conf_xgb_valid$overall["Accuracy"]
conf_xgb_valid$byClass["Precision"]
conf_xgb_valid$byClass["Recall"]
conf_xgb_valid$byClass["F1"]


#### Day 5 Methods ####

##### Support Vector Machines ####

### Visualize the data with multidimensional scaling (MDS)

# Limit to numeric variables and small data set for quick visualization
# Returns x and y coordinates

cmdscale(dist(loans_train_small[,Filter(is.numeric,.SD)]), # independent variables 
         k = 2) # number of dimensions to reduce to

# Make colors as target (default yes/no)

### SVM with a linear kernel (hyperplane, linear subspace)
svm_linear <- svm(default ~ ., # variables
                  data = loans_train_small[,-c("default_num")], # training data, Version with the "small" training data
                  kernel = "linear", # linear kernel
                  na.action = na.pass, # what to do with missings
                  cost = 10, # cost of constraints violation
                  shrinking = TRUE, # use shrinkage?
                  probability = TRUE, # calculate predicted probabilities
                  scale = FALSE) # whether to scale the variables. If true, the default is to scale x and y to mean 0 and variance 1.

print(svm_linear)

summary(svm_linear)

plot(svm_linear, loans_train_small[,-c("default_num")])



### Predictions for the Test Data ####

test_xgb <- xgb.DMatrix(model.matrix.lm(default~.-1,loans_test,na.action="na.pass"),label=as.numeric(loans_test$default)-1)

pred_test<-data.table(xgb_prob=predict(xgb, newdata=test_xgb,type="prob")) # Probabilities
pred_test$xgb_class<-ifelse(pred_test$xgb_prob>=0.5,"yes","no") # Classes (default yes/no)

conf_xgb_test <- confusionMatrix(as.factor(pred_test$xgb_class),as.factor(loans_test$default), 
                                 positive="yes", # which value is what we're trying to predict? Here, default yes
                                 dnn = c("Prediction", "Actual Data"))
conf_xgb_test
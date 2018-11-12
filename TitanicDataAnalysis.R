#import the data
test <- read.csv('~/GitHub/Titanic/test.csv', header = TRUE)
train <- read.csv('~/GitHub/Titanic/train.csv', header = TRUE)

#add a survived column to test populate with "None"
test.survived <- data.frame(Survived = rep("None", nrow(test)), test[,])
test.survived <- test.survived[, c(2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)]

#combine the two into one big set
data.combined <- rbind(train, test.survived)

#Lets see the data types in the data frame
str(data.combined)

#convert the survived and pclass into factors
data.combined$Survived <- as.factor(data.combined$Survived)
data.combined$Pclass <- as.factor(data.combined$Pclass)

#what values are in sex
table(data.combined$Sex)

#what values are in age
summary(data.combined$Age)
summary(test$Age)
summary(train$Age)

#look at survival rates
table(data.combined$Survived)

#look at pclass
table(data.combined$Pclass)

#Load ggplot2
library(ggplot2)

#Look to see if there is any pattern in pclass
train$Pclass <- as.factor(train$Pclass)
ggplot(train, aes(x = Pclass, fill = factor(Survived))) +
  geom_bar() +
  stat_count(width = 0.5) +
  xlab("Pclass") +
  ylab("Total Count") +
  labs(fill = "Survived")

#As the age is poorly coded - can we use title as a substitute (Miss, Master, Mr, Mrs)
#Check the first few rows
head(as.character(train$Name))

#Load stringr for text manipulation
library(stringr)

misses <- data.combined[which(str_detect(data.combined$Name, "Miss.")),]
misses[1:5,]
summary(misses$Age)

mrs <- data.combined[which(str_detect(data.combined$Name, "Mrs.")),]
mrs[1:5,]
summary(mrs$Age)

#Extract the title from the name

extractTitle <- function(name) {
  name <- as.character(name)
  
  if (length(grep("Miss.", name)) > 0) {
    return("Miss.")
  } else if (length(grep("Master.", name)) > 0) {
    return("Master.")
  } else if (length(grep("Mrs.", name)) > 0) {
    return("Mrs.")
  } else if (length(grep("Mr.", name)) > 0) {
    return("Mr.")
  } else {
    return("Other")
  }
}

titles <- NULL
for (i in 1:nrow(data.combined)) {
  titles <- c(titles, extractTitle(data.combined[i, "Name"]))
}
data.combined$Title <- as.factor(titles)

#Plot this - use only the train portion of data.combined
ggplot(data.combined[1:891,], aes(x = Title, fill = Survived)) +
  geom_bar() +
  stat_count(width = 0.5) +
  facet_wrap(~Pclass) +
  ggtitle("Pclass") +
  xlab("Title") +
  ylab("Total Count") +
  labs(fill = "Survived")

#create a new vaiable for family size based on SibSp + parch + 1
data.combined$FamilySize <- as.factor(data.combined$SibSp + data.combined$Parch + 1)

table(data.combined$FamilySize)

#Plot the new variable against Pclass and Title
ggplot(data.combined[1:891,], aes(x = FamilySize, fill = Survived)) +
  geom_bar() +
  stat_count(width = 0.5) +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass/Title") +
  xlab("Family Size") +
  ylab("Total Count") +
  labs(fill = "Survived")

#Load tidyverse
library(tidyverse)

#Look for misses alone to see if this removes children
missesAlone <- data.combined %>%
  filter(FamilySize == 1 & Title == "Miss.")

summary(missesAlone$Age)

#Look at fare
ggplot(data.combined[1:891,], aes(x = Fare, fill = Survived)) +
  geom_histogram(binwidth = 5) +
  facet_wrap(~Pclass + Title) +
  ggtitle("Pclass/Title") +
  xlab("Fare") +
  ylab("Total Count") +
  ylim(0, 50) +
  labs(fill = "Survived")

#now lets look at an exploratory model to see what we have:
#load random forest
library(randomForest)

#First try with Pclass and Title
rf.train.1 <- data.combined[1:891, c("Pclass", "Title")]
rf.label <- as.factor(train$Survived)

set.seed(1234)
rf.1 <- randomForest(x = rf.train.1, y = rf.label, importance = TRUE, ntree = 1000)
rf.1
varImpPlot(rf.1)

#20.99% error rate (79.01% accurate)

#include family size
rf.train.2 <- data.combined[1:891, c("Pclass", "Title", "FamilySize")]

set.seed(1234)
rf.2 <- randomForest(x = rf.train.2, y = rf.label, importance = TRUE, ntree = 1000)
rf.2
varImpPlot(rf.2)

#18.18% error rate (81.82% accurate)

#include family size + sibsp
rf.train.3 <- data.combined[1:891, c("Pclass", "Title", "SibSp", "FamilySize")]

set.seed(1234)
rf.3 <- randomForest(x = rf.train.3, y = rf.label, importance = TRUE, ntree = 1000)
rf.3
varImpPlot(rf.3)

#19.3% error rate (80.7% accurate)


library(caret)
library(doSNOW)

#Now we do cross validation (CV) to check how accurate we really are as 80.92% is optimistic
#10-fold CV is the defacto standard - no hard and fast rules and this is where the experience
#of Data Scientist (the art of it) comes in to its own

#We are going to leverage caret t create 100 folds but ensure that the ratio of those that
#survived and perished in each fold matches back to the training set. This is known as a
#stratified cross validatio and generally gives better results
set.seed(2348)
cv.10.folds <- createMultiFolds(rf.label, k = 10, times = 10)

#check stratification
table(rf.label)
342/549

table(rf.label[cv.10.folds[[33]]])
308/494

#Set up caret's trainControl object as above
ctrl.1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv.10.folds)

#Set up doSNOW for multi core training. This will be helpful as we are training lots of trees
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#Set seed for reproducablity and train
set.seed(34324)
rf.3.cv.1 <- train(x = rf.train.3, y = rf.label, method = "rf", tuneLength = 3,
                   ntree = 1000, trControl = ctrl.1)
  
#Shutdown cluster
stopCluster(cl)

#check out results
rf.3.cv.1

#THis is only a little more pessimistic than our earlier rf so we need to re-look at it as it nees to
#be more pessimistic
#try with 5 folds instead of 10
set.seed(5983)
cv.5.folds <- createMultiFolds(rf.label, k = 5, times = 10)

ctrl.2 <- trainControl(method = "repeatedcv", number = 5, repeats = 10,
                       index = cv.5.folds)

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#Set seed for reproducablity and train
set.seed(89472)
rf.3.cv.2 <- train(x = rf.train.3, y = rf.label, method = "rf", tuneLength = 3,
                   ntree = 1000, trControl = ctrl.2)

#Shutdown cluster
stopCluster(cl)

#check out results
rf.3.cv.2

#still too high
#try with 3 folds instead of 5
set.seed(37596)
cv.3.folds <- createMultiFolds(rf.label, k = 3, times = 10)

ctrl.3 <- trainControl(method = "repeatedcv", number = 3, repeats = 10,
                       index = cv.3.folds)

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#Set seed for reproducablity and train
set.seed(94622)
rf.3.cv.3 <- train(x = rf.train.3, y = rf.label, method = "rf", tuneLength = 3,
                   ntree = 64, trControl = ctrl.3)

#Shutdown cluster
stopCluster(cl)

#check out results
rf.3.cv.3


#Use rf 2 - Title Pclass and Family Size
set.seed(45986)
cv.4.folds <- createMultiFolds(rf.label, k = 3, times = 10)

ctrl.4 <- trainControl(method = "repeatedcv", number = 3, repeats = 10,
                       index = cv.4.folds)

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

#Set seed for reproducablity and train
set.seed(94622)
rf.4.cv.4 <- train(x = rf.train.2, y = rf.label, method = "rf", tuneLength = 3,
                   ntree = 64, trControl = ctrl.4)

#Shutdown cluster
stopCluster(cl)

#check out results
rf.4.cv.4

#Now look at CART to see whats going on, this is a single tree
library(rpart)
library(rpart.plot)

#use 3 fold x 10 times as previous as this works well with the proportions of our test / train data
rpart.cv <- function(seed, training, labels, ctrl){
  cl <- makeCluster(3, type = "SOCK")
  registerDoSNOW(cl)
  
  set.seed(seed)
  #leverage formula interface for training
  rpart.cv <- train(x = training, y = labels, method = "rpart", tuneLength = 30,
                    trControl = ctrl)
  
  #shutdown cluster
  stopCluster(cl)
  
  return(rpart.cv)
  
}

#grab features
features <- c("Pclass","Title","FamilySize")
rpart.train.1 <- data.combined[1:891, features]

#run CV and check results
rpart.1.cv.1 <- rpart.cv(94622, rpart.train.1, rf.label, ctrl.4)
rpart.1.cv.1

#plot
prp(rpart.1.cv.1$finalModel, type = 0, extra = 1, under = TRUE)


#check title
table(data.combined$Title)

#parse out name and title
data.combined[1:25, "Name"]

name.splits <- str_split(data.combined$Name, ",")
name.splits[1]

last.names <- sapply(name.splits, "[",1)
last.names[1:10]

#add to our df
data.combined$LastName <- last.names

#now for titles
name.splits <- str_split(sapply(name.splits, "[",2)," ")
titles <- sapply(name.splits, "[", 2)
unique(titles)

#what is the "the"
data.combined[which(titles == "the"),]

#Re-Map the titles to be more specific
titles[titles %in% c("Dona.","the")] <- "Lady."
titles[titles %in% c("Ms.","Mlle.")] <- "Miss."
titles[titles == "Mme."] <- "Mrs."
titles[titles %in% c("Jonkheer.","Don.")] <- "Sir."
titles[titles %in% c("Col.","Capt.","Major.")] <- "Officer."
table(titles)

data.combined$NewTitle <- as.factor(titles)

#plot it
ggplot(data.combined[1:891,], aes(x=NewTitle, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) +
  ggtitle("Survival Rates by PClass and new Title")

#collapse the titles based on visual analysis 
indexes <- which(data.combined$NewTitle == "Lady.")

data.combined$NewTitle[indexes] <- "Mrs."

indexes <- which(data.combined$NewTitle == "Dr." |
                   data.combined$NewTitle == "Rev." |
                   data.combined$NewTitle == "Sir." |
                   data.combined$NewTitle == "Officer." )

data.combined$NewTitle[indexes] <- "Mr."

table(data.combined$NewTitle)


#plot it
ggplot(data.combined[1:891,], aes(x=NewTitle, fill = Survived)) +
  geom_bar() +
  facet_wrap(~Pclass) +
  ggtitle("Survival Rates by PClass and new Title")


#lets redo the models with the new and improved titles
features <- c("Pclass","NewTitle","FamilySize")
rpart.train.2 <- data.combined[1:891, features]

#Run CV
rpart.2.cv.1 <- rpart.cv(94622, rpart.train.2, rf.label, ctrl.3)
rpart.2.cv.1

#plot
prp(rpart.2.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

#Dive into Mr. and 1st class
indexes.first.mr <- which(data.combined$NewTitle == "Mr." & data.combined$Pclass == "1")
first.mr.df <- data.combined[indexes.first.mr,]
summary(first.mr.df)

#Female Mr.?
first.mr.df[first.mr.df$Sex == "female",]

#Femlae Dr set to Mrs
indexes <- which(data.combined$NewTitle == "Mr." &
                   data.combined$Sex == "female")

data.combined$NewTitle[indexes] <- "Mrs."

#Check for any other gender slip ups
length(which(data.combined$Sex == "female" & 
               (data.combined$NewTitle == "Master." |
               data.combined$NewTitle == "Mr.")))

#now re do 1st class men data frame
indexes.first.mr <- which(data.combined$NewTitle == "Mr." & data.combined$Pclass == "1")
first.mr.df <- data.combined[indexes.first.mr,]
summary(first.mr.df)

#look at the ones that actually survived
summary(first.mr.df[first.mr.df$Survived == "1",])
View(first.mr.df[first.mr.df$Survived == "1",])

#Get some high fares
indexes <- which(data.combined$Ticket == "PC 17755" |
                 data.combined$Ticket == "PC 17611" |
                 data.combined$Ticket == "113760")

View(data.combined[indexes, ])


#visualise Mr. Survival rates by ticket price
ggplot(first.mr.df, aes(x = Fare, fill = Survived)) +
         geom_density(alpha = 0.5) +
         ggtitle("1st Class Mr. Survived by Fare")


#Engineer feature based on number of people with the same ticket number
ticket.party.size <- rep(0, nrow(data.combined))
avg.fare <- rep(0.0, nrow(data.combined))
tickets <- unique(data.combined$Ticket)

for (i in 1:length(tickets)){
  current.ticket <- tickets[i]
  party.indexes <- which(data.combined$Ticket == current.ticket)
  current.avg.fare <- data.combined[party.indexes[1], "Fare"] / length(party.indexes)
  
  for (k in 1:length(party.indexes)){
    ticket.party.size[party.indexes[k]] <- length(party.indexes)
    avg.fare[party.indexes[k]] <- current.avg.fare
  }
}

data.combined$ticket.party.size <- ticket.party.size
data.combined$avg.fare <- avg.fare

#refresh male df
first.mr.df <- data.combined[indexes.first.mr, ]
summary(first.mr.df)

#visualise the new features
ggplot(first.mr.df[first.mr.df$Survived != "None", ], aes(x = ticket.party.size, fill = Survived)) +
  geom_density(alpha = 0.5) +
  ggtitle("1st Class Mr. Survived by Party Size")


ggplot(first.mr.df[first.mr.df$Survived != "None", ], aes(x = avg.fare, fill = Survived)) +
  geom_density(alpha = 0.5) +
  ggtitle("1st Class Mr. Survived by Avg Fare")

#hypothesis that party sie is highly correlated to avg ticket fare
summary(data.combined$avg.fare)

#missing value
data.combined[is.na(data.combined$Fare), ]

#get details of similar passengers
indexes <- with(data.combined, which(Pclass == "3" & Title == "Mr." & FamilySize == "1" & 
                                       Ticket != "3701" ))

similar.na.passengers <- data.combined[indexes,]
summary(similar.na.passengers)

#use median avg fare
data.combined[is.na(avg.fare), "avg.fare"] <-7.840

#Leverage caret's preProcess funtion to normalise data
preProc.data.combined <- data.combined[, c("ticket.party.size","avg.fare")]
preProc <- preProcess(preProc.data.combined, method = c("center","scale"))

postproc.data.combined <- predict(preProc, preProc.data.combined)

#hypothesis refuted for all data
cor(postproc.data.combined$ticket.party.size, postproc.data.combined$avg.fare)?cor

#how about for 1st Class
indexes <- which(data.combined$Pclass == "1")
cor(postproc.data.combined$ticket.party.size[indexes],
    postproc.data.combined$avg.fare[indexes])

#hypothesis refuted again


#Lets use our new features to see what we get now
features <- c("Pclass", "NewTitle", "FamilySize", "ticket.party.size", "avg.fare")
rpart.train.3 <- data.combined[1:891, features]

#Run CV and check results
rpart.3.cv.1 <- rpart.cv(94622, rpart.train.3, rf.label, ctrl.3)
rpart.3.cv.1

#plot
prp(rpart.3.cv.1$finalModel, type = 0, extra = 1, under = TRUE)


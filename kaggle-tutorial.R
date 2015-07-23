# read training data
train <- read.csv('dataraw/training.csv', stringsAsFactors=F)
raw <- train
str(train)

install.packages('doMC')
library(doMC)
registerDoMC(6)

#convert Image lists to numeric
images <- train$Image
images <- foreach(im = images, .combine = rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}



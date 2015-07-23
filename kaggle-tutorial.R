# read training data

data_setup <- function() {
  train <- read.csv('dataraw/training.csv', stringsAsFactors=F)
  test <- read.csv('dataraw/test.csv', stringsAsFactors=F)
  raw <- train
  str(train)

  install.packages('doMC')
  library(doMC)
  registerDoMC(6)

  #convert Image lists to numeric
  images <- train$Image
  imagesgg <- foreach(im = images, .combine = rbind) %dopar% {
    data.frame(melt(matrix(as.integer(unlist(strsplit(im, " "))), nrow = 96, ncol=96)))
  }

  testimages <- test$Image
  testimages <- foreach(im = testimages, .combine = rbind) %dopar% {
    as.integer(unlist(strsplit(im, " ")))
  }
  save(train, test, images, testimages, file='fr_data.RData')
}

messing_around <- function() {
  mat <- matrix(data=rev(images[1,]), nrow=96, ncol=96)
  system.time(
    image(1:96, 1:96, mat, col = gray((0:255)/255))
  )
  head(train[,-(length(train))])

  # calculate perfectly average face
  means <- c()
  for ( i in 1:ncol(images) ){
    #   print(i)
    means <- c(means, mean(images[,i]))
    #   print(means)
  }
  meanmat <- matrix(data=rev(means), nrow=96, ncol=96)

  # plot face with ggplot
  library(ggplot2);library(ggthemes)
  p <- ggplot(aes(x=X1, y=X2, alpha=-value), data=data.frame(melt(meanmat))) + theme_minimal() + theme(legend.position='none')
  p + geom_raster() + geom_point(x=96-mean(train$left_eye_center_x, na.rm=T), y=96-mean(train$left_eye_center_y, na.rm=T), color='blue', size=4) + geom_point(x=96-mean(train$right_eye_center_x, na.rm=T), y=96-mean(train$right_eye_center_y, na.rm=T), color='blue', size=4)
}






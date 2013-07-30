sigm <- function(x,weights){
  # Sigmoid activation function
  Z <- x %*% weights
  return(1/(1+exp(-Z)))
}

logreg <- function(data,labels,eta,nIter){
  # This function performs binary classification by logistic regression
  # Data is a design matrix, i.e. N rows of the form (x1,...,xd)
  # Labels is a vector of class labels {0,1}
  # Eta is the learning rate
  # nIter is a specified number of iterations
  
  # Append bias
  bias <- rep(1,length(data[,1]))
  data <- cbind(bias,data)

  N <- length(bias) # Number of training samples
  d <- length(data[1,]) # Dimensionality of input

  weights <- rep(0.0001,d)
  
  # Learn the weights that achieve accurate training classification
  for(x in 1:nIter){
    for(n in 1:N){
      x <- data[n,]
      y <- sigm(x,weights)
      t <- labels[n]
      weights <- weights - eta * ((t-y) * (y * (1-y)) * x)

    }
  }
  weights
}

classify <- function(weights,newdata,newlabels){
  # Having learnt the weights, use these to
  # predict the test data
  bias <- rep(1, length(newdata[,1]))
  newdata <- cbind(bias,newdata)
  N <- length(newdata[,1])
  
  predicted <- rep(0,N)
    
  for(n in 1:N){
    x <- newdata[n,]
    y <- sigm(x,weights)
    if(y > 0.5){
      predicted[n] <- 0
    }
    else{
      predicted[n] <- 1
    }
  }
  predicted
}

giveLabeledData <- function(N,dimen){
  dat = matrix(nrow=N,ncol=dimen,data=0)
  labs = rep(0,N)
  for(i in 1:N){
    # Draw sample from one of two distributions
    samp = rbinom(1,1,0.5)
    if(samp == 1){
      dat[i,] <- rnorm(dimen,mean = 10, sd = 0.5)
      labs[i] <- 1
    }
    else{
      dat[i,] <- rnorm(dimen,mean=1,sd=0.5)
      labs[i] <- 0
    }
  }
  return(list(dat,labs))
}

dat <- giveLabeledData(1000,2)

mydata <- dat[[1]]
labels <- dat[[2]]

# Learn weights on training data
ab <- logreg(mydata[1:700,],labels[1:700],0.2,6)

# Plot decision boundary
plot(mydata[1:700,],pch=20)
abline(-ab[1]/ab[3],ab[2]/-ab[3])

# Classify the remaining test data
pred <- classify(ab,mydata[-(1:700),],labels[-(1:700)])

# Test error
1-mean(pred == labels[-(1:700)])
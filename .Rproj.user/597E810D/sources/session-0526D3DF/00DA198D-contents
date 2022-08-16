###
# first order derivative functions

didentity <- function(x){
  return(x$data)
}

dfexp <- function(x){
  m <- as.matrix(x$data) %*% matrix(x$pars[2:length(x$pars)],ncol=1)
  X <- matrix(exp(m),ncol=1)
  for(i in 1:ncol(x$data)){
    X <- cbind(X,x$data[,i]*x$pars[i+1]*exp(m))
  }
  return(X)
}

dfactor <- function(x){
  matrix(model.matrix(~factor(a)-1,data.frame(a=x$data)),nrow=length(x$data))
}

dar1 <- function(x){
  
}

dlog <- function(x){
  
}
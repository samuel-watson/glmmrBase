#' Identity function
#' @param x List with named vector data
didentity <- function(x){
  return(x$data)
}

#' Exponential function
#' @param x List with named elements `data`, a vector of covariate values, and `pars`, a vector of two parameters
dfexp <- function(x){
  m <- as.matrix(x$data) %*% matrix(x$pars[2:length(x$pars)],ncol=1)
  X <- matrix(exp(m),ncol=1)
  for(i in 1:ncol(x$data)){
    X <- cbind(X,x$data[,i]*x$pars[i+1]*exp(m))
  }
  return(X)
}

#' Factor function
#' @param x List with named elements `data`, a vector of covariate values, and `pars`, a vector of two parameters
dfactor <- function(x){
  matrix(model.matrix(~factor(a)-1,data = data.frame(a=x$data)),nrow=length(x$data))
}

# dar1 <- function(x){
# 
# }
# 
# dlog <- function(x){
# 
# }
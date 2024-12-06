#' Generate matrix mapping between data frames
#' 
#' For a data frames `x` and `target`, the function will return a matrix mapping the rows of
#' `x` to those of `target`.
#' 
#' @details 
#' `x` is a data frame with n rows and `target` a data frame with m rows. This function will
#' return a n times m matrix that maps the rows of `x` to those of `target` based on the values
#' in the columns specified by the argument `by`
#' 
#' @param x data.frame 
#' @param target data.frame to map to
#' @param by vector of strings naming columns in `x` and `target`
#' @return A matrix with nrow(x) rows and nrow(target) columns
#' @examples 
#' df <- nelder(~(cl(10)*t(5)) > ind(10))
#' df_unique <- df[!duplicated(df[,c('cl','t')]),]
#' match_rows(df,df_unique,c('cl','t'))
#' @export
match_rows <- function(x,target,by){
  if(!is(x,"data.frame")|!is(target,"data.frame"))stop("x and target must be data frames")
  if(!all(by%in%colnames(x))|!all(by%in%colnames(target)))stop("by must contain column names of x and target")
  
  if(ncol(target)==1){
    tstr <- target[,by]
    xstr <- x[,by]
  } else {
    xstr <- Reduce(paste0,as.data.frame(apply(x[,by],2,function(i)paste0(i,".0000."))))
    tstr <- Reduce(paste0,as.data.frame(apply(target[,by],2,function(i)paste0(i,".0000."))))
  }
  Z <- matrix(0,nrow=length(xstr),ncol=length(tstr))
  mat <- lapply(tstr,function(i)which(xstr==i))
  for(i in 1:length(mat))Z[mat[[i]],i] <- 1
  return(Z)
}

#' Generates a progress bar
#'
#' Prints a progress bar
#'
#' @param i integer. The current iteration.
#' @param n integer. The total number of interations
#' @param len integer. Length of the progress a number of characters
#' @return A character string
#' @examples
#' progress_bar(10,100)
#' @export
progress_bar <- function(i,n,len=30){
  prop <- floor((i*100/n) / (100/len))
  pt1 <- paste0(rep("=",prop), collapse="")
  pt2 <- paste0(rep(" ",len-prop), collapse = "")
  msg <- paste0("|",pt1,pt2,"| ",round((i*100/n),0),"%")
  return(msg)
}


#' Returns the file name and type for MCNR function
#' 
#' Returns the file name and type for MCNR function
#' 
#' @param family family object
#' @param cmdstan Logical indicating whether cmdstan is being used and the function will return the filename
#' @return list with filename and type
mcnr_family <- function(family, cmdstan){
  f1 <- tolower(family[[1]])
  link <- family[[2]]
  gaussian_list <- c("identity")
  binomial_list <- c("logit","log","identity","probit")
  bernoulli_list <- c("logit","log","identity","probit")
  quantile_list <- quantile_scaled_list <- c("identity","log","logit","probit","inverse")
  poisson_list <- c("log")
  gamma_list <- c("identity","inverse","log")
  beta_list <- c("logit")
  if(f1 == "quantile_scaled")f1 <- "quantile"
  type <- which(get(paste0(f1,"_list"))==link)
  if(length(type)==0)stop("link not supported for this family")
  if(cmdstan){
    return(list(file = paste0("mcml_",f1,".stan"),type=type))
  } else {
    return(list(file = paste0("mcml_",f1),type=type))#,".stan"
  }
}

#' Simulated data from a stepped-wedge cluster trial
#'
#' @name SimTrial
#' @docType data
#' @examples
#' #Data were generated with the following code:
#' SimTrial <- nelder(~ (cl(10)*t(7))>i(10))
#' SimTrial$int <- 0
#' SimTrial[SimTrial$t > SimTrial$cl,'int'] <- 1
#' 
#' model <- Model$new(
#'   formula = ~ int + factor(t) - 1 + (1|gr(cl)*ar1(t)),
#'   covariance = c(0.05,0.8),
#'   mean = rep(0,8),
#'   data = SimTrial,
#'   family = gaussian()
#' )
#' 
#' SimTrial$y <- model$sim_data()
NULL

#' Simulated data from a geospatial study with continuous outcomes
#'
#' @name SimGeospat
#' @docType data
#' @examples
#' #Data were generated with the following code:
#' n <- 600
#' SimGeospat <- data.frame(x = runif(n,-1,1), y = runif(n,-1,1))
#' 
#' sim_model <- Model$new(
#'   formula = ~ (1|fexp(x,y)),
#'   data = SimGeospat,
#'   covariance = c(0.25,0.3),
#'   mean = c(0),
#'   family = gaussian()
#' )
#' 
#' SimGeospat$y <- sim_model$sim_data()
NULL

#' Salamanders data
#' 
#' Obtained from 
#' \code{
#' uu <- url("http://www.math.mcmaster.ca/bolker/R/misc/salamander.txt")
#' sdat <- read.table(uu,header=TRUE,colClasses=c(rep("factor",5),"numeric"))
#' }
#' See \url{https://rpubs.com/bbolker/salamander} for more information.
#' @name Salamanders
#' @docType data
NULL

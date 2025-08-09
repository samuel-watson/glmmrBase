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

#' Rescales data to [-1,1] 
#' 
#' Rescales data to [-1,1] for HSGP model fitting
#' 
#' @details 
#' The HSGP covariance function requires that all dimensions are scaled to 
#' [-1,1] as conversion is not automatic. This function will rescale the D
#' covariance variables to [-1,1]^D while preserving their size relative to 
#' one another.
#' @param data A data frame
#' @param columns Vector of integers. The indexes of the columns to be rescaled.
#' @return A copy of the input data frame with rescaled columns
#' @examples 
#' df <- data.frame(x = runif(100,0,2), y = runif(100, -2,2))
#' df <- hsgp_rescale(df, 1:2)
#' @export
hsgp_rescale <- function(data, columns){
  # scale to -1,1 in all dimensions
  ranges <- data.frame(lower = rep(NA,length(columns)), upper = rep(NA, length(columns)))
  for(i in 1:length(columns)){
    ranges[i,] <- range(data[,columns[i]])
  }
  scale_f <- max(ranges$upper - ranges$lower)
  for(i in 1:length(columns)){
    data[,columns[i]] <- -1 + 2*(data[,columns[i]] - ranges$lower[i])/scale_f
  }
  return(data)  
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
  # if(f1 %in% c("gaussian","beta","gamma")) fty <- "cont"
  # if(f1 %in% c("bernoulli","binomial","poisson")) fty <- "int"
  # if(f1 %in% c("quantile","quantile_scaled")) fty <- "quantile"
  if(f1 == "gaussian"){
    if(link == "identity"){
      type <- 1
    } else if(link == "log") {
      type <- 2
    }
  } else if(f1 == "beta"){
    type <- 3
  } else if(f1 == "gamma"){
    if(link == "identity"){
      type <- 4
    } else if(link == "inverse"){
      type <- 5
    } else if(link == "log"){
      type <- 6
    }
  } else if(f1 == "bernoulli"){
    if(link == "logit"){
      type <- 7
    } else if(link == "log"){
      type <- 8
    } else if(link == "identity"){
      type <- 9
    } else if(link == "probit"){
      type <- 10
    }
  } else if(f1 == "binomial"){
    if(link == "logit"){
      type <- 11
    } else if(link == "log"){
      type <- 12
    } else if(link == "identity"){
      type <- 13
    } else if(link == "probit"){
      type <- 14
    }
  } else if(f1 == "poisson"){
    type <- 15
  } else if(f1 %in% c("quantile", "quantile_scaled")){
    if(link == "logit"){
      type <- 18
    } else if(link == "log"){
      type <- 17
    } else if(link == "identity"){
      type <- 16
    } else if(link == "probit"){
      type <- 19
    } else if(link == "inverse"){
      type <- 20
    }
  }
  
  if(length(type)==0)stop("link not supported for this family")
  if(cmdstan){
    return(list(file = "mcml.stan",type=type))
  } else {
    return(list(file = "mcml",type=type))#,".stan"
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
#'   covariance = c(0.25,0.8),
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

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

#' Exponential covariance function
#' 
#' Exponential covariance function
#' 
#' @details 
#' The function:
#' 
#' \dexp{f(x) = \theta_1*exp(-\theta_2*x)}
#' 
#' @param x A list with named elements `pars` and `data`. `pars` is a vector with one parameter values, and
#' `data` is the data `x`
#' @return vector of values of the function
#' @examples 
#' fexp(list(pars = c(1,0.2),data=runif(10)))
#' @export
fexp <- function(x){
  #if(length(x$pars)!=2)stop("two parameters required for fexp")
  #x$pars[1]*exp(-x$pars[2]*x$data)
  exp(-x$pars[1]*x$data)
}

#' Power exponential covariance function
#' 
#' Power exponential covariance function
#' 
#' @details 
#' The function:
#' 
#' \dexp{f(x) = \theta_1^x}
#' 
#' @param x A list with named elements `pars` and `data`. `pars` is a vector with one parameter value, and
#' `data` is the data `x`
#' @return vector of values of the function
#' @examples 
#' fexp(list(pars = c(0.8),data=runif(10)))
#' @export
pexp <- function(x){
  x$pars[1]^x$data
}

#' Group indicator covariance function
#' 
#' Group indicator covariance function
#' 
#' @details 
#' The function:
#' 
#' \dexp{f(x) = 1(x==0)*\theta_1}
#' 
#' @param x A list with named elements `pars` and `data`. `pars` is a vector with one parameter value, and
#' `data` is the data `x`
#' @return vector of values of the function
#' @examples 
#' fexp(list(pars = c(0.8),data=c(0,1,0,2,3,4,0)))
#' @export
gr <- function(x){
  I(x$data==0)*x$pars[1]^2
}

#' Create block matrix
#' 
#' Create a block matrix from the inputs
#' 
#' @details 
#' Takes a sequence of matrices and produces a block matrix. 
blockmat <- function(...){
  matlist <- list(...)
  n <- length(matlist)
  N <- 0:(n-1)
  rlist <- list()
  for(i in 1:n){
    N <- (N+1)%%n
    N[N==0] <- n
    if(i<n)matlistt <- matlist[N]
    rlist[[i]] <- Reduce(cbind,matlistt[N])
  }
  Reduce(rbind,rlist)
}

#' Generate a list of a given structure with new values
#' 
#' Generate a list of a given structure with new values
#' 
#' @param lst List whose structure will be replicated with the new values
#' @param value vector of values to replace in `lst`. The order of replacement is given by `unlist(lst)`
#' @param p Used internally to track level of recursion
#' @return A list of the same structure as `lst` but with the values given by `value`
#' @examples 
#' df <- nelder(~(cl(5)*t(5)) > ind(5))
#' cov <- Covariance$new(formula = ~(1|gr(j)*pexp(t)),
#'                       parameters = list(list(0.05,0.8)),
#'                       data= df)
#' cov$parameters <- relist(cov$parameters,
#'                          c(0.01,0.5))
#' @export
relist <- function(lst,value,p=0){
  if(is(lst,"list")){
    for(i in 1:length(lst)){
      out <- Recall(lst[[i]],value,p=p)
      lst[[i]] <- out[[1]]
      p <- out[[2]]
    }
  } else {
    for(i in 1:length(lst)){
      lst[i] <- value[p+1]
      p <- p + 1
    }
  }
  return(list(lst,p))
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
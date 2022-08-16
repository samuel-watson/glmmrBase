#' Generate crossed block structure
#' 
#' Generate a data frame with crossed rows from two other data frames
#' 
#' @details 
#' For two data frames `df1` and `df2`, the function will return another data frame
#' that crosses them, which has rows with every unique combination of the input data frames
#' @param df1 data frame
#' @param df2 data frame
#' @return data frame
#' @examples 
#' cross_df(data.frame(t=1:4),data.frame(cl=1:3))
#' @export
cross_df <- function(df1,df2){
  if(!is(df1,"data.frame")|!is(df2,"data.frame"))stop("inputs must be data frames")
  
  cnames <- c(colnames(df1),colnames(df2))
  df1 <- as.data.frame(df1[rep(1:nrow(df1),each=nrow(df2)),])
  df3 <- cbind(df1,df2[1:nrow(df2),])
  colnames(df3) <- cnames
  return(df3)
}

#' Generate nested block structure
#' 
#' Generate a data frame that nests one data frame in another
#' 
#' @details 
#' For two data frames `df1` and `df2`, the function will return another data frame
#' that nests `df2` in `df1`. So each row of `df1` will be duplicated `nrow(df2)` times 
#' and matched with `df2`. The values of each `df2` will be unique for each row of `df1`
#' @param df1 data frame
#' @param df2 data frame
#' @return data frame
#' @examples 
#' nest_df(data.frame(t=1:4),data.frame(cl=1:3))
#' @export
nest_df <- function(df1,df2){
  df3 <- cbind(df1[rep(1:nrow(df1),each=nrow(df2)),],df2)
  colnames(df3)[1:ncol(df1)] <- colnames(df1)
  if(ncol(df1)>1)ids <- Reduce(paste0,df3[,1:ncol(df1)]) else ids <- df3[,1]
  df3[,(ncol(df1)+1):(ncol(df1)+ncol(df2))] <- apply(as.data.frame(df3[,(ncol(df1)+1):(ncol(df1)+ncol(df2))]),2,
                                                     function(i)as.numeric(as.factor(paste0(ids,i))))
  colnames(df3[,(ncol(df1)+1):(ncol(df1)+ncol(df2))]) <- colnames(df2)
  return(df3)
}

#' Generates a block experimental structure using Nelder's formula
#' 
#' Generates a data frame expressing a block experimental structure using Nelder's formula
#' 
#' @details 
#' Nelder (1965) suggested a simple notation that could express a large variety of different blocked designs. 
#' The function `nelder()` that generates a data frame of a design using the notation. 

#' There are two operations:
#'  
#' `>` (or \eqn{\to} in Nelder's notation) indicates "clustered in".
#'
#' `*` (or \eqn{\times} in Nelder's notation) indicates a crossing that generates all combinations of two factors.
#' 
#' The implementation of this notation includes a string indicating the name of the variable and a number for the number of levels, 
#' such as `abc(12)`. So for example `~cl(4) > ind(5)` means in each of five levels of `cl` there are five levels of `ind`, and 
#' the individuals are different between clusters. The formula `~cl(4) * t(3)` indicates that each of the four levels of `cl` are 
#' observed for each of the three levels of `t`. Brackets are used to indicate the order of evaluation. Some specific examples:
#' 
#' `~person(5) * time(10)`: A cohort study with five people, all observed in each of ten periods `time`
#'
#' `~(cl(4) * t(3)) > ind(5)`: A repeated-measures cluster study with four clusters (labelled `cl`), each observed in each time 
#' period `t` with cross-sectional sampling and five indviduals (labelled `ind`) in each cluster-period.
#'
#' `~(cl(4) > ind(5)) * t(3)`: A repeated-measures cluster cohort study with four clusters (labelled `cl`) wth five 
#' individuals per cluster, and each cluster-individual combination is observed in each time period `t`.
#'
#' `~((x(100) * y(100)) > hh(4)) * t(2)`: A spatio-temporal grid of 100x100 and two time points, with 4  households per spatial 
#' grid cell.
#' @param formula A model formula. See details
#' @return A list with the first member being the data frame 
#' @examples 
#' nelder(~(j(4) * t(5)) > i(5))
#' nelder(~person(5) * time(10)))
#' @export
nelder <- function(formula){
  if(formula[[1]]=="~")formula <- formula[[2]]
  f1l <- formula[[2]]
  f1r <- formula[[3]]
  
  if(as.character(f1l[[1]])%in%c("*",">")){
    df1 <- Recall(f1l)
  } else if(as.character(f1l[[1]])%in%c("(")){
    df1 <- Recall(f1l[[2]])
  } else {
    df1 <- data.frame(a = seq(1,f1l[[2]]))
    colnames(df1) <- as.character(f1l[[1]])
  }
  
  if(as.character(f1r[[1]])%in%c("(")){
    df2 <- Recall(f1r[[2]])
  } else {
    df2 <- data.frame(a = seq(1,f1r[[2]]))
    colnames(df2) <- as.character(f1r[[1]])
  }
  
  if(formula[[1]] == "*"){
    df <- cross_df(df1,df2)
  } else if(formula[[1]] == ">"){
    df <- nest_df(df1,df2)
  }
  rownames(df) <- NULL
  return(df)
}

#' Generates all the orderings of a
#' 
#' Given input a, returns a length(a)^2 vector by cycling through the values of a
#' @param a vector
#' @return vector
cycles <- function(a){
  fa <- a
  for(i in 1:(length(a)-1)){
    a <- c(a[2:length(a)],a[1])
    fa <- c(fa,a)
  }
  fa
}
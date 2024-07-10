#' Family declaration to support quantile regression
#' 
#' Skeleton list to declare a quantile regression model in a `Model` object. 
#' 
#' @param link Name of the link function - any of `identity`, `log`, `logit`, `inverse`, or `probit`
#' @param scaled Logical indicating whether to include a scale parameter. If FALSE then the scale parameter is one.
#' @param q Scalar in [0,1] declaring the quantile of interest.
#' @return A list with two elements naming the family and link function
#' @export
quantile <- function(link="identity", scaled = FALSE, q = 0.5){
  if(! link %in% c("identity","log","logit","inverse","probit"))stop("Link not supported for quantile regression")
  if(q <= 0 | q >= 1) stop("q outside [0,1]")
  return(list(Family = ifelse(!scaled,"quantile","quantile_scaled"), Link= link, q = q))
}

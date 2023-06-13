#' Beta distribution declaration
#' 
#' Skeleton list to declare a Beta distribution in a `Model` object
#' 
#' @param link Name of link function. Only accepts `logit` currently.
#' @return A list with two elements naming the family and link function
#' @export
Beta <- function(link="logit"){
  if(link != "logit")stop("Only logit currently supported for Beta distribution")
  return(list(Family = "beta", Link= link))
}

Rcpp::loadModule("covariance_cpp",TRUE)
.onLoad <- function(libname, pkgname){
  #covclass <- covariance_cpp$covariance
}
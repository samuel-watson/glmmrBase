# Rcpp::loadModule("covariance_cpp",TRUE)
# .onLoad <- function(libname, pkgname){
#   #covclass <- covariance_cpp$covariance
# }

setClass( "Covariance", representation( pointer = "externalptr" ) )
Covariance_method <- function(name) {
  paste( "Covariance", name, sep = "__" )
}
setMethod( "$", "Covariance", function(x, name ) {
  function(...) .Call( Uniform_method(name) , x@pointer , ... )
} )
setMethod( "initialize", "Covariance", function(.Object, ...) {
  .Object@pointer <- .Call( Covariance_method("new"), ... )
  .Object
} )

#' Summarizes a `Model` object
#' 
#' Summarizes `Model` object. 
#' @param object An `Model` object.
#' @param max_n Integer. The maximum number of rows to print.
#' @param ... Further arguments passed from other methods
#' @return An object of class `logLik`. If both `fixed` and `covariance` are FALSE then it returns NA.
#' @method summary Model
#' @export
summary.Model <- function(object, max_n = 10, ...){
  cat("\nA GLMM Model")
  print(object$family)
  cat("\nFormula: ",object$formula)
  
  if(object$family[[1]] %in% c("binomial","bernoulli"))
    cat("\nTrials: ",object$trials[1:max_n])
  
  cat("\nWeights: ", object$weights[1:max_n])
  
  if(object$.__enclos_env__$private$y_has_been_updated){
    cat("\nLog-likelihood: ", mod$log_likelihood())
  }
  
  cat("\n\nFIXED EFFECTS")
  cat("\nParameter values: ", object$mean$parameters)
  M <- solve(object$information_matrix())
  cat("\nGLS variance-covariance matrix at current parameter values:\n")
  print(M)
  
  cat("\n\nRANDOM EFFECTS")
  cat("\nFormula: ",self$covariance$formula)
  cat("\nParameters: ")
  print(mod$covariance$parameter_table())
  cat("\nCurrent values:\n")
  u <- object$u()
  print(summary(u[,1:(min(max_n,ncol(u)))]))
  
  cat("\n\nSee help(Model) for a detailed list of available methods")
  
  cat("\n")
}

#' Extracts the log-likelihood from an mcml object
#' 
#' Extracts the log-likelihood value from an `Model` object. If no data `y` are specified then it returns NA.
#' @param object An `Model` object.
#' @param ... Further arguments passed from other methods
#' @return An object of class `logLik`. If both `fixed` and `covariance` are FALSE then it returns NA.
#' @method logLik Model
#' @export
logLik.Model <- function(object, ...){
  ll <- tryCatch(object$log_likelihood(),
                 error = function(e)message("No data has been set in the Model. See Model$update_y()"))
  class(ll) <- "logLik"
  attr(ll,"df") <- length(object$mean$parameters) + length(object$covariance$parameters) + I(object$family[[1]] %in% c("gaussian","Beta"))*1
  attr(ll,"nobs") <- object$n()
  attr(ll,"nall") <- object$n()
  return(ll)
}

#' Extracts coefficients from a Model object
#' 
#' Extracts the coefficients from a `Model` object.
#' @param object A `Model` object.
#' @param ... Further arguments passed from other methods
#' @return Fixed effect and covariance parameters extracted from the model object.
#' @method coef Model
#' @export
coef.Model <- function(object,...){
  return(c(object$mean$parameters, object$covariance$parameters))
}

#' Extracts the family from a `Model` object. This information can also be
#' accessed directly from the Model as `Model$family`
#' 
#' Extracts the \link[stats]{family} from a `Model` object.
#' @param object A `Model` object.
#' @param ... Further arguments passed from other methods
#' @return A \link[stats]{family} object.
#' @method family Model
#' @export
family.Model <- function(object,...){
  return(object$family)
}

#' Extracts the formula from a `Model` object
#' 
#' Extracts the \link[stats]{formula} from a `Model` object. This information can also be
#' accessed directly from the Model as `Model$formula`
#' @param object A `Model` object.
#' @param ... Further arguments passed from other methods
#' @return A \link[stats]{formula} object.
#' @method formula Model
#' @export
formula.Model <- function(object,...){
  return(as.formula(object$formula))
}

#' Calculate Variance-Covariance matrix for a `Model` object
#' 
#' Returns the variance-covariance matrix for a `Model` object. Specifically, this function will 
#' return the inverse GLS information matrix for the fixed effect parameters. Small sample corrections 
#' can be accessed directly from the Model using `Model$small_sample_correction()`. The varaince-covariance 
#' matrix including the random effects can be accessed using `Model$information_matrix(include.re = TRUE)`.
#' @param object A `Model` object.
#' @param ... Further arguments passed from other methods
#' @return A variance-covariance matrix.
#' @method vcov Model
#' @export
vcov.Model <- function(object,...){
  V <- solve(object$information_matrix())
  rownames(V) <- colnames(V) <- names(object$mean$parameters)
  return(V)
}

#' Generate predictions at new values from a `Model` object
#' 
#' Generates predicted values from a `Model` object using a new data set to specify covariance 
#' values and values for the variables that define the covariance function.
#' The function will return a list with the linear predictor, conditional 
#' distribution of the new random effects term conditional on the current estimates
#' of the random effects, and some simulated values of the random effects if requested. Typically 
#' this functionality is accessed using `Model$predict()`, which this function provides a wrapper for.
#' @param object A `Model` object.
#' @param newdata A data frame specifying the new data at which to generate predictions
#' @param m Number of samples of the random effects to draw
#' @param offset Optional vector of offset values for the new data
#' @param ... Further arguments passed from other methods
#' @return A list with the linear predictor, parameters (mean and covariance matrices) for
#' the conditional distribution of the random effects, and any random effect samples.
#' @method predict Model
#' @export
predict.Model <- function(object,
                          newdata,
                          offset = rep(0,nrow(newdata)),
                          m=0, ...){
  if(missing(offset)) {
    off <- rep(0, nrow(newdata))
  } else {
    off <- offset
  }
  if(missing(m)){
    mm <- 0
  } else {
    mm <- m
  }
  return(model$predict(newdata, off, mm))
}

#' Extract or generate fitted values from a `Model` object
#'
#' Return fitted values. Does not account for the random effects. This function is a wrapper for `Model$fitted()`, which 
#' also provides a variety of additional options for generating fitted values from mixed models. 
#' For simulated values based on resampling random effects, see also `Model$sim_data()`. To predict the values including random effects at a new location see also
#' `Model$predict()`.
#' @param object A `Model` object.
#' @param ... Further arguments passed from other methods
#' @return Fitted values
#' @method fitted Model
#' @export
fitted.Model <- function(object, ...){
  return(model$fitted())
}

#' Extract residuals from a `Model` object
#'
#' Return the residuals from a `Model` object. This function is a wrapper for `Model$residuals()`.
#' Generates one of several types of residual for the model. If conditional = TRUE then 
#' the residuals include the random effects, otherwise only the fixed effects are included. For type,
#' there are raw, pearson, and standardized residuals. For conditional residuals a matrix is returned 
#' with each column corresponding to a sample of the random effects.
#' @param object A `Model` object.
#' @param type Either "standardized", "raw" or "pearson"
#' @param conditional Logical indicating whether to condition on the random effects (TRUE) or not (FALSE)
#' @param ... Further arguments passed from other methods
#' @return A matrix with either one column is conditional is false, or with number of columns corresponding 
#' to the number of MCMC samples.
#' @method residuals Model
#' @export
residuals.Model <- function(object, type, conditional, ...){
  return(model$residuals(type, conditional))
}


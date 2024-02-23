#' Prints an mcml fit output
#' 
#' Print method for class "`mcml`"
#' 
#' @param x an object of class "`mcml`" as a result of a call to MCML, see \link[glmmrBase]{Model}
#' @param ... Further arguments passed from other methods
#' @details 
#' `print.mcml` tries to replicate the output of other regression functions, such
#' as `lm` and `lmer` reporting parameters, standard errors, and z- and p- statistics.
#' The z- and p- statistics should be interpreted cautiously however, as generalised
#' linear miobjected models can suffer from severe small sample biases where the effective
#' sample size relates more to the higher levels of clustering than individual observations.
#' 
#' Parameters `b` are the mean function beta parameters, parameters `cov` are the
#' covariance function parameters in the same order as `$covariance$parameters`, and
#' parameters `d` are the estimated random effects.
#' @return No return value, called for side effects.
#' @method print mcml
#' @export
print.mcml <- function(x, ...){
  digits <- 4
  cat(ifelse(x$method%in%c("mcem","mcnr"),
             "Markov chain Monte Carlo Maximum Likelihood Estimation\nAlgorithm: ",
             "Maximum Likelihood Estimation with Laplace Approximation\nAlgorithm: "),
      ifelse(x$method%in%c("nloptim","nr"),ifelse(x$method=="nr","Newton-Raphson","BOBYQA"),
             ifelse(x$method=="mcem","Markov Chain Expectation Maximisation",
                                        "Markov Chain Newton-Raphson")),
      ifelse(x$sim_step," with simulated likelihood step\n","\n"))
  
  cat("\nFixed effects formula :",x$mean_form)
  cat("\nCovariance function formula: ",x$cov_form)
  cat("\nFamily: ",x$family,", Link function:",x$link)
  setype <- switch(
    x$se,
    "gls" = "GLS",
    "robust" = "Robust",
    "kr" = "Kenward Roger",
    "kr2" = "Kenward Roger (improved)",
    "bw" = "GLS with between-within correction",
    "bwrobust" = "Robust with between-within correction",
    "box" = "Modified Box correction",
    "sat" = "Satterthwaite"
  )
  cat("\nStandard error: ",setype,"\n")
  if(x$method%in%c("mcem","mcnr"))cat("\nNumber of Monte Carlo simulations per iteration: ",x$m," with tolerance ",x$tol,"\n\n")
  
  dim1 <- dim(x$re.samps)[1]
  pars <- x$coefficients[1:(length(x$coefficients$par)-dim1),2:7]
  colnames(pars) <- c("Estimate","Std. Err.","z value","p value","2.5% CI","97.5% CI")
  if(x$se == "bw" || x$se == "bwrobust" || x$se == "kr" || x$se == "kr2" || x$se == "sat")colnames(pars)[3] <- "t value"
  if(x$se == "box")colnames(pars)[3] <- "F value"
  rnames <- x$coefficients$par[1:(length(x$coefficients$par)-dim1)]
  if(any(duplicated(rnames))){
    did <- unique(rnames[duplicated(rnames)])
    for(i in unique(did)){
      rnames[rnames==i] <- paste0(rnames[rnames==i],".",1:length(rnames[rnames==i]))
    }
  }
  rownames(pars) <- rnames
  total_vars <- x$P+x$Q
  if(x$var_par_family)total_vars <- total_vars + 1
  if(x$se %in% c("kr","bw","bwrobust","kr2","box","sat"))pars$DoF <- c(x$dof, rep(NA,total_vars - x$P))
  pars <- apply(pars,2,round,digits = digits)
  if(x$se %in% c("kr","bw","bwrobust","kr2","box","sat")){
    colrange <- 1:7
  } else if(x$se == "box"){
    colrange <- c(1:4,7)
  } else {
    colrange <- 1:6
  }
  
  cat("\nRandom effects: \n")
  print(pars[(x$P+1):(total_vars),c(1,2)])
  
  cat("\nFixed effects: \n")
  print(pars[1:x$P,colrange])
  
  cat("\ncAIC: ",round(x$aic,digits))
  cat("\nApproximate R-squared: Conditional: ",round(x$Rsq[1],digits)," Marginal: ",round(x$Rsq[2],digits))
  cat("\nLog-likelihood: ",round(x$logl,digits))
  if(!x$converged)cat("\nMCML ALGORITHM DID NOT CONVERGE!")

  return(invisible(pars))
}

#' Summarises an mcml fit output
#' 
#' Summary method for class "`mcml`"
#' 
#' @param object an object of class "`mcml`" as a result of a call to MCML, see \link[glmmrBase]{Model}
#' @param ... Further arguments passed from other methods
#' @details 
#' `print.mcml` tries to replicate the output of other regression functions, such
#' as `lm` and `lmer` reporting parameters, standard errors, and z- and p- statistics.
#' The z- and p- statistics should be interpreted cautiously however, as generalised
#' linear miobjected models can suffer from severe small sample biases where the effective
#' sample size relates more to the higher levels of clustering than individual observations.
#' 
#' Parameters `b` are the mean function beta parameters, parameters `cov` are the
#' covariance function parameters in the same order as `$covariance$parameters`, and
#' parameters `d` are the estimated random effects.
#' @return A list with random effect names and a data frame with random effect mean and credible intervals
#' @method summary mcml
#' @export
summary.mcml <- function(object,...){
  digits <- 2
  pars <- print(object)
  ## summarise random effects
  dfre <- data.frame(Mean = round(apply(object$re.samps,2,mean),digits = digits), 
                     lower = round(apply(object$re.samps,2,function(i)stats::quantile(i,0.025)),digits = digits),
                     upper = round(apply(object$re.samps,2,function(i)stats::quantile(i,0.975)),digits = digits))
  colnames(dfre) <- c("Estimate","2.5% CI","97.5% CI")
  cat("Random effects estimates\n")
  print(dfre)
  ## add in model fit statistics
  return(invisible(list(coefficients = pars,re.terms = dfre)))
}


#' Extracts coefficients from a mcml object
#' 
#' Extracts the coefficients from an `mcml` object returned from a call of `MCML` or `LA` in the \link[glmmrBase]{Model} class.
#' @param object An `mcml` model fit.
#' @param ... Further arguments passed from other methods
#' @return A data frame summarising the parameters including the random effects.
#' @method coef mcml
#' @export
coef.mcml <- function(object,...){
  return(object$coefficients)
}

#' Extracts the computed AIC from an mcml object
#' 
#' Extracts the conditional Akaike Information Criterion from an mcml object returned from call of `MCML` or `LA` in the \link[glmmrBase]{Model} class.
#' @param object An `mcml` model fit.
#' @param ... Further arguments passed from other methods
#' @return A numeric value.
#' @method extractAIC mcml
#' @export
extractAIC.mcml <- function(object,...){
  return(object$aic)
}

#' Extracts the log-likelihood from an mcml object
#' 
#' Extracts the final log-likelihood value from an mcml object returned from call of `MCML` or `LA` in the \link[glmmrBase]{Model} class. The fitting algorithm estimates
#' the fixed effects, random effects, and covariance parameters all separately. The log-likelihood is separable in the fixed and covariance parameters, so one can return 
#' the log-likelihood for either component, or the overall log-likelihood.
#' @param object An `mcml` model fit.
#' @param fixed Logical whether to include the log-likelihood value from the fixed effects.
#' @param covaraince Logical whether to include the log-likelihood value from the covariance parameters.
#' @param ... Further arguments passed from other methods
#' @return A numeric value. If both `fixed` and `covariance` are FALSE then it returns NA.
#' @method logLik mcml
#' @export
logLik.mcml <- function(object, fixed = TRUE, covariance = TRUE, ...){
  ll <- 0
  if(fixed) ll <- ll + object$logl
  if(covariance) ll <- ll + object$logl_theta
  if(!fixed & !covariance) ll <- NA
  return(NA)
}

#' Extracts the fixed effect estimates
#' 
#' Extracts the fixed effect estimates from an mcml object returned from call of `MCML` or `LA` in the \link[glmmrBase]{Model} class.
#' @param object An `mcml` model fit.
#' @param ... Further arguments passed from other methods
#' @return A named, numeric vector of fixed-effects estimates.
#' @method ranef mcml
#' @export
fixef.mcml <- function(object,...){
  fixed <- object$coefficients$est[1:object$P]
  names(fixed) <- object$coefficients$par[1:object$P]
  return(fixed)
}

#' Extracts the random effect estimates
#' 
#' Extracts the random effect estimates or samples from an mcml object returned from call of `MCML` or `LA` in the \link[glmmrBase]{Model} class.
#' @param object An `mcml` model fit.
#' @param ... Further arguments passed from other methods
#' @return A matrix of dimension (number of fixed effects ) x (number of MCMC samples). For Laplace approximation, the number of "samples" equals one.
#' @method ranef mcml
#' @export
ranef.mcml <- function(object,...){
  return(object$re.samps)
}
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
  digits <- 2
  cat(ifelse(x$method%in%c("mcem","mcnr"),
             "Markov chain Monte Carlo Maximum Likelihood Estimation\nAlgorithm: ",
             "Maximum Likelihood Estimation with Laplace Approximation\nAlgorithm: "),
      ifelse(x$method%in%c("nloptim","nr"),ifelse(x$method=="nr","Newton-Raphson","BOBYQA"),
             ifelse(x$method=="mcem","Markov Chain Expectation Maximisation",
                                        "Markov Chain Newton-Raphson")),
      ifelse(x$sim_step," with simulated likelihood step\n","\n"))
  
  cat("\nFixed effects formula :",x$mean_form)
  cat("\nCovariance function formula: ",x$cov_form)
  cat("\nFamily: ",x$family,", Link function:",x$link,"\n")
  if(x$method%in%c("mcem","mcnr"))cat("\nNumber of Monte Carlo simulations per iteration: ",x$m," with tolerance ",x$tol,"\n")
  # semethod <- ifelse(x$permutation,"permutation test",
  #                    ifelse(x$robust,"robust",ifelse(x$hessian,"hessian","approx")))
  # cat("P-value and confidence interval method: ",semethod,"\n\n")
  dim1 <- dim(x$re.samps)[1]
  pars <- x$coefficients[1:(length(x$coefficients$par)-dim1),c('est','SE','lower','upper')]
  z <- pars$est/pars$SE
  pars <- cbind(pars[,1:2],z=z,p=2*(1-stats::pnorm(abs(z))),pars[,3:4])
  colnames(pars) <- c("Estimate","Std. Err.","z value","p value","2.5% CI","97.5% CI")
  rnames <- x$coefficients$par[1:(length(x$coefficients$par)-dim1)]
  if(any(duplicated(rnames))){
    did <- unique(rnames[duplicated(rnames)])
    for(i in unique(did)){
      rnames[rnames==i] <- paste0(rnames[rnames==i],".",1:length(rnames[rnames==i]))
    }
  }
  rownames(pars) <- rnames
  pars <- apply(pars,2,round,digits = digits)
  print(pars)
  
  cat("\ncAIC: ",round(x$aic,digits))
  cat("\nApproximate R-squared: Conditional: ",round(x$Rsq[1],digits)," Marginal: ",round(x$Rsq[2],digits))
  
#   #messages
#   if(x$permutation)message("Permutation test used for one parameter, other SEs are not reported. SEs and Z values
# are approximate based on the p-value, and assume normality.")
  #if(!x$hessian&!x$permutation)warning("Hessian was not positive definite, standard errors are approximate")
  if(!x$converged)warning("Algorithm did not converge")
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
#' TBC!!
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




#' Map lme4 formula to glmmrBase formula
#' 
#' Returns a formula that can be used for glmmrBase Models from an lme4 input.
#' 
#' @details
#' The package lme4 uses a syntax to specify random effects as `(1|x)` where `x` is the grouping variable.
#' This function will modify such a formula, including those with nesting and crossing operators `/` and `:` into
#' the glmmrBase syntax using the `gr()` function. Not typically required by the user as it is used internally 
#' in the `mcml_lmer` and `mcml_glmer` functions.
#' @param formula A lme4 style formula
#' @param cnames The column names of the data to be used. These are used to check if the specified clustering variables are in 
#' the data.
#' @return A formula.
#' @examples 
#' df <- data.frame(cl = 1:3, t = 4:6)
#' f1 <- lme4_to_glmmr(y ~ x + (1|cl/t),colnames(df))
#' @export
lme4_to_glmmr <- function(formula,cnames){
  re1 <- re0 <- re_names(as.character(formula)[3])
  int1 <- unlist(lapply(regmatches(re1, gregexpr("\\|.*?\\)", re1)), function(x)gsub("[\\|\\)]","",x)))
  for(i in 1:length(int1)){
    if(int1[[i]]%in%cnames){
      re1[[i]] <- gsub(int1[[i]],paste0("gr(",int1[[i]],")"),re1[[i]])
    } else {
      int2 <- unlist(strsplit(int1[[i]],"/"))
      if(all(int2%in%cnames)){
        re1[[i]] <- gsub(int1[[i]],paste0("gr(",int2[1],")"),re1[[i]])
        re1[[i]] <- paste0(re1[[i]],"+(1|gr(",int2[1],",",int2[2],"))")
      } else {
        int3 <- unlist(strsplit(int1[[i]],":"))
        if(all(int3%in%cnames)){
          re1[[i]] <- gsub(int1[[i]],paste0("gr(",int2[1],",",int2[2],")"),re1[[i]])
        } 
      }
    }
  }
  f2 <- as.character(formula)[3]
  f2 <- gsub(" ","",f2)
  for(i in 1:length(re0)){
    re0[i] <- gsub("\\(","\\\\(",re0[i])
    re0[i] <- gsub("\\|","\\\\|",re0[i])
    re0[i] <- gsub("\\)","\\\\)",re0[i])
    re0[i] <- gsub("\\/","\\\\/",re0[i])
    re0[i] <- gsub("\\:","\\\\:",re0[i])
    re1[i] <- gsub("\\(","\\\\(",re1[i])
    re1[i] <- gsub("\\|","\\\\|",re1[i])
    re1[i] <- gsub("\\)","\\\\)",re1[i])
  }
  for(i in 1:length(re1))f2 <- gsub(re0[i],re1[i],f2)
  formula_out <- formula(paste0(as.character(formula)[2],"~",f2))
  return(formula_out)
}

#' lme4 style linear mixed model 
#' 
#' A wrapper for Model stochastic maximum likelihood model fitting replicating lme4's syntax
#' 
#' @details
#' This function aims to replicate the syntax of lme4's `lmer` command. The specified formula can be 
#' the standard lme4 syntax, or alternatively a glmmrBase style formula can also be used to allow for the 
#' wider range of covariance function specifications. For example both `y~x+(1|cl/t)` and `y~x+(1|gr(cl))+(1|gr(cl)*ar1(t))`
#' would be valid formulae.
#' @param formula A two-sided linear formula object including both the fixed and random effects specifications, see Details.
#' @param data A data frame containing the variables named in `formula`.
#' @param start Optional. A vector of starting values for the fixed effects.
#' @param offset Optional. A vector of offset values.
#' @param weights Optional. A vector of observation level weights to apply to the model fit.
#' @param iter.warmup The number of warmup iterations for the MCMC sampling step of each iteration.
#' @param iter.sampling The number of sampling iterations for the MCMC sampling step of each iteration.
#' @param verbose Integer, controls the level of detail printed to the console, either 0 (no output), 
#' 1 (main output), or 2 (detailed output)
#' @param ... additional arguments passed to `Model$MCML()`
#' @return A `mcml` model fit object.
#' @examples 
#' #create a data frame describing a cross-sectional parallel cluster
#' #randomised trial
#' df <- nelder(~(cl(10)*t(5)) > ind(10))
#' df$int <- 0
#' df[df$cl > 5, 'int'] <- 1
#' # simulate data using the Model class
#' df$y <- Model$new(
#'   formula = ~ factor(t) + int - 1 + (1|gr(cl)) + (1|gr(cl,t)),
#'   data = df,
#'   family = stats::gaussian()
#' )$sim_data()
#' fit <- mcml_lmer(y ~ factor(t) + int - 1 + (1|cl/t), data = df)
#' @export
mcml_lmer <- function(formula, data, start = NULL, offset = NULL, verbose = 1L,
                      iter.warmup = 100, iter.sampling = 50, weights = NULL,...){
  # parse the formula from lme4's syntax to glmmrBase
  model <- Model$new(formula = lme4_to_glmmr(formula,colnames(data)),
                   data = data,
                   family = gaussian(),
                   offset = offset,
                   weights = weights)
  model$mcmc_options$warmup <- iter.warmup
  model$mcmc_options$samps <- iter.sampling
  model$set_trace(verbose)
  if(!is.null(start))model$update_parameters(mean.pars = start)
  fit <- model$MCML(...)
  return(fit)
}

#' lme4 style generlized linear mixed model 
#' 
#' A wrapper for Model stochastic maximum likelihood model fitting replicating lme4's syntax
#' 
#' @details
#' This function aims to replicate the syntax of lme4's `lmer` command. The specified formula can be 
#' the standard lme4 syntax, or alternatively a glmmrBase style formula can also be used to allow for the 
#' wider range of covariance function specifications. For example both `y~x+(1|cl/t)` and `y~x+(1|gr(cl))+(1|gr(cl)*ar1(t))`
#' would be valid formulae.
#' @param formula A two-sided linear formula object including both the fixed and random effects specifications, see Details.
#' @param data A data frame containing the variables named in `formula`.
#' @param family A family object expressing the distribution and link function of the model, see \link[stats]{family}.
#' @param start Optional. A vector of starting values for the fixed effects.
#' @param offset Optional. A vector of offset values.
#' @param weights Optional. A vector of observation level weights to apply to the model fit.
#' @param iter.warmup The number of warmup iterations for the MCMC sampling step of each iteration.
#' @param iter.sampling The number of sampling iterations for the MCMC sampling step of each iteration.
#' @param verbose Integer, controls the level of detail printed to the console, either 0 (no output), 
#' 1 (main output), or 2 (detailed output)
#' @param ... additional arguments passed to `Model$MCML()`
#' @return A `mcml` model fit object.
#' @examples 
#' #create a data frame describing a cross-sectional parallel cluster
#' #randomised trial
#' df <- nelder(~(cl(10)*t(5)) > ind(10))
#' df$int <- 0
#' df[df$cl > 5, 'int'] <- 1
#' # simulate data using the Model class
#' df$y <- Model$new(
#'   formula = ~ factor(t) + int - 1 + (1|gr(cl)) + (1|gr(cl,t)),
#'   data = df,
#'   family = stats::binomial()
#' )$sim_data()
#' fit <- mcml_glmer(y ~ factor(t) + int - 1 + (1|cl/t), data = df, family = binomial())
#' @export
mcml_glmer <- function(formula, data, family, start = NULL, offset = NULL, verbose = 1L,
                       iter.warmup = 100, iter.sampling = 50, weights = NULL,...){
  
  model <- Model$new(formula = lme4_to_glmmr(formula,colnames(data)),
                     data = data,
                     family = family,
                     offset = offset,
                     weights = weights)
  model$mcmc_options$warmup <- iter.warmup
  model$mcmc_options$samps <- iter.sampling
  model$set_trace(verbose)
  if(!is.null(start))model$update_parameters(mean.pars = start, cov.pars = runif(length(model$covariance$parameters),0,0.1))
  fit <- model$MCML(...)
  
  return(fit)
}



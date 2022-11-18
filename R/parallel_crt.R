#' Generate a parallel cluster design
#' 
#' Generate a parallel cluster randomised trial design in glmmr
#' 
#' The complete parallel cluster randomised trial design has J clusters
#' observed over T time periods. A proportion (`ratio`) of the clusters are assigned to
#' treatment condition for the duration of the trial and the rest are control.
#' 
#' @param J Integer indicating the number of sequences such that there are J+1 time periods
#' @param M Integer. The number of individual observations per cluster-period, assumed equal across all clusters
#' @param t Integer. The number of time periods.
#' @param ratio Numeric value indicating the proportion of clusters assigned to treatment. Default is 0.5.
#' @param beta Vector of beta parameters to initialise the design, defaults to all zeros.
#' @param icc Intraclass correlation coefficient. User may specify
#' more than one value, see details.
#' @param cac Cluster autocorrelation coefficient, optional and user may specify more than one value, see details
#' @param iac Individual autocorrelation coefficient, optional and user may specify more than one value, see details
#' @param var Assumed overall variance of the model, used to calculate the other covariance, see details
#' @param family a \link[stats]{family} object 
#' @details 
#' The assumed generalised linear mixed model for the parallel cluster trial is, for 
#' individual i, in cluster j, at time t:
#' 
#' \deqn{y_{ijt} \sim F(\mu_{ijt},\sigma)}
#' \deqn{\mu_{ijt} = h^-1(x_{ijt}\beta + \alpha_{1j} + \alpha_{2jt} + \alpha_{3i})}
#' \deqn{\alpha_{p.} \sim N(0,\sigma^2_p), p = 1,2,3}
#' 
#' Defining \eqn{\tau} as the total model variance, then the intraclass correlation 
#' coefficient (ICC) is
#' \deqn{ICC = \frac{\sigma_1 + \sigma_2}{\tau}}
#' the cluster autocorrelation coefficient (CAC) is :
#' \deqn{CAC = \frac{\sigma_1}{\sigma_1 + \sigma_2}}
#' and the individual autocorrelation coefficient as:
#' \deqn{IAC = \frac{\sigma_3}{\tau(1-ICC)}}
#' 
#' When CAC and/or IAC are not specified in the call, then the respective random effects
#' terms are assumed to be zero. For example, if IAC is not specified then \eqn{\alpha_{3i}}
#' does not appear in the model, and we have a cross-sectional sampling design; if IAC
#' were specified then we would have a cohort.
#' 
#' For non-linear models, such as Poisson or Binomial models, there is no single obvious choice
#' for `var_par` (\eqn{\tau} in the above formulae), as the models are heteroskedastic. Choices 
#' might include the variance at the mean values of the parameters or a reasonable choice based
#' on the variance of the respective distribution.
#' 
#' If the user specifies more than one value for icc, cac, or iac, then a ModelSpace is returned
#' with Models with every combination of parameters. This can be used in particular to generate
#' a design space for optimal design analyses.
#' @examples 
#' #generate a simple design with only cluster random effects and 6 clusters in 3 time periods
#' # with 10 individuals in each cluster-period
#' des <- parallel_crt(J=6,M=10,t=3,icc=0.05)
#' # same design but with a cohort of individuals
#' des <- parallel_crt(J=6,M=10,t=3,icc=0.05, iac = 0.1)
#' # same design, but with two clusters per sequence and specifying the initial parameters
#' des <- parallel_crt(J=6,M=10,t=3,beta = c(rnorm(3,0,0.1),-0.1),icc=0.05, iac = 0.1)
#' # specifying multiple values of the variance parameters will return a design space 
#' # with all designs with all the combinations of the variance parameter
#' des <- parallel_crt(J=6,M=10,t=3,icc=c(0.01,0.05), cac = c(0.5,0.7,0.9), iac = 0.1)
#' @return A Model object with MeanFunction and Covariance objects, or
#' a ModelSpace holding several such Model objects.
#' @seealso \link[glmmrBase]{Model}
#' @importFrom methods is 
#' @importFrom stats model.matrix family formula
#' @export
parallel_crt <-  function(J,
                          M,
                          t,
                          ratio = 0.5,
                          beta=c(rep(0,t),0),
                          icc,
                          cac = NULL,
                          iac = NULL,
                          var = 1,
                          family = stats::gaussian()){
  if(missing(icc))stop("icc must be set as a minimum")
  
  ndesigns <- length(icc) * ifelse(!is.null(cac[1]),length(cac),1) *
    ifelse(!is.null(iac[1]),length(iac),1)
  
  if(!is.null(cac[1]) && !is.na(cac[1])){
    wp_var <- icc[1]*var*(1-cac[1])
    bp_var <- icc[1]*var[1]*cac[1]
  } else {
    bp_var <- icc[1]*var
  }
  if(!is.null(iac[1]) && !is.na(iac[1])){
    ind_var <- var*(1-icc[1])*iac[1]
    sigma <- var*(1-icc[1])*(1-iac[1])
  } else {
    sigma <- var*(1-icc[1])
  }
  
  if(!is.null(iac[1]) && !is.na(iac[1])){
    df <- nelder(formula(paste0("~ (J(",J,") > ind(",M,")) * t(",t,")")))
  } else {
    df <- nelder(formula(paste0("~ (J(",J,") * t(",t,")) > ind(",M,")")))
  }
  
  ## assign treatment
  df$int <- 0
  df[df$J <= round(J*ratio,0),'int'] <- 1
  
  if(is.null(cac[1]) || is.na(cac[1])){
    if(is.null(iac[1]) || is.na(iac[1])){
      f1 <- "~(1|gr(J))"
      pars <- c(sqrt(bp_var))
    } else {
      f1 <- "~(1|gr(J)) + (1|gr(ind))"
      pars <- c(sqrt(bp_var),sqrt(ind_var))
    }
  } else {
    if(is.null(iac[1]) || is.na(iac[1])){
      f1 <- "~ (1|gr(J)) + (1|gr(J*t))"
      pars <- c(sqrt(bp_var),sqrt(wp_var))
    } else {
      f1 <- "~ (1|gr(J)) + (1|gr(J*t)) + (1|gr(ind))"
      pars <- c(sqrt(bp_var),sqrt(wp_var),sqrt(ind_var))
    }
  }
  
  d1 <- Model$new(
    covariance = list(
      data=df,
      formula = f1,
      parameters = pars
    ),
    mean = list(
      formula = "~ factor(t) + int - 1",
      data = df,
      family = family,
      parameters = beta
      
    ),
    var_par = sigma
  )
  
  # if(ndesigns>1&requireNamespace(glmmrOptim)){
  #   ds1 <- ModelSpace$new(d1)
  #   if(is.null(cac))cac <- NA
  #   if(is.null(iac))iac <- NA
  #   dsvalues <- expand.grid(icc=icc,cac=cac,iac=iac)
  #   
  #   for(i in 1:(ndesigns-1)){
  #     
  #     if(!is.null(dsvalues$cac[i+1]) && !is.na(dsvalues$cac[i+1])){
  #       wp_var <- dsvalues$icc[i+1]*var*(1-dsvalues$cac[i+1])
  #       bp_var <- dsvalues$icc[i+1]*var*dsvalues$cac[i+1]
  #     } else {
  #       bp_var <- dsvalues$icc[i+1]*var
  #     }
  #     if(!is.null(dsvalues$iac[i+1]) && !is.na(dsvalues$iac[i+1])){
  #       ind_var <- var*(1-dsvalues$icc[i+1])*dsvalues$iac[i+1]
  #       sigma <- var*(1-dsvalues$icc[i+1])*(1-dsvalues$iac[i+1])
  #     } else {
  #       sigma <- var*(1-dsvalues$icc[i+1])
  #     }
  #     
  #     if(is.null(dsvalues$cac[i+1]) || is.na(dsvalues$cac[i+1])){
  #       if(is.null(dsvalues$iac[i+1]) || is.na(dsvalues$iac[i+1])){
  #         f1 <- "~(1|gr(J))"
  #         pars <- c(sqrt(bp_var))
  #       } else {
  #         f1 <- "~(1|gr(J)) + (1|gr(ind))"
  #         pars <- c(sqrt(bp_var),sqrt(ind_var))
  #       }
  #     } else {
  #       if(is.null(dsvalues$iac[i+1]) || is.na(dsvalues$iac[i+1])){
  #         f1 <- "~ (1|gr(J)) + (1|gr(J*t))"
  #         pars <- c(sqrt(bp_var),sqrt(wp_var))
  #       } else {
  #         f1 <- "~ (1|gr(J)) + (1|gr(J*t)) + (1|gr(ind))"
  #         pars <- c(sqrt(bp_var),sqrt(wp_var),sqrt(ind_var))
  #       }
  #     }
  #     
  #     
  #     ds1$add(
  #       Model$new(
  #         covariance = Covariance$new(
  #           data=df,
  #           formula = f1,
  #           parameters = pars
  #         ),
  #         mean.function = d1$mean_function$clone(),
  #         var_par = 1
  #       )
  #     )
  #     
  #     # $parameters <- pars
  #     # ds1$.__enclos_env__$private$designs[[i+1]]$covariance$formula <- f1
  #     
  #   }
  #   return(ds1)
  # } else {
  #   return(d1)
  # }
  return(invisible(d1))
}

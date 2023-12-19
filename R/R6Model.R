#' A GLMM Model
#'
#' A generalised linear mixed model and a range of associated functions
#' @details
#' A detailed vingette for this package is available online<doi:10.48550/arXiv.2303.12657>. Briefly, for the generalised linear mixed model
#'
#' \deqn{Y \sim F(\mu,\sigma)}
#' \deqn{\mu = h^-1(X\beta + Zu)}
#' \deqn{u \sim MVN(0,D)}
#'
#' where h is the link function. The class provides access to all of the matrices above and associated calculations and functions including model fitting, power analysis,
#' and various relevant decompositions. The object is an R6 class and so can serve as a parent class for extended functionality.
#'
#' Many calculations use the covariance matrix of the observations, such as the information matrix, which is used in power calculations and
#' other functions. For non-Gaussian models, the class uses the first-order approximation proposed by Breslow and Clayton (1993) based on the
#' marginal quasilikelihood:
#'
#' \deqn{\Sigma = W^{-1} + ZDZ^T}
#'
#' where _W_ is a diagonal matrix with the GLM iterated weights for each observation equal
#' to, for individual _i_ \eqn{\left( \frac{(\partial h^{-1}(\eta_i))}{\partial \eta_i}\right) ^2 Var(y|u)}
#' (see Table 2.1 in McCullagh and Nelder (1989)). The modification proposed by Zegers et al to the linear predictor to
#' improve the accuracy of approximations based on the marginal quasilikelihood is also available, see `use_attenuation()`.
#'
#' See \href{https://github.com/samuel-watson/glmmrBase/blob/master/README.md}{glmmrBase} for a
#' detailed guide on model specification.
#' 
#' The class also includes model fitting with Markov Chain Monte Carlo Maximum Likelihood implementing the algorithms described by McCulloch (1997), 
#' and fast model fitting using Laplace approximation. Functions for returning related values such as the log gradient, log probability, and other 
#' matrices are also available.
#' @references
#' Breslow, N. E., Clayton, D. G. (1993). Approximate Inference in Generalized Linear Mixed Models.
#' Journal of the American Statistical Association<, 88(421), 9–25. <doi:10.1080/01621459.1993.10594284>
#' 
#' McCullagh P, Nelder JA (1989). Generalized linear models, 2nd Edition. Routledge.
#' 
#' McCulloch CE (1997). “Maximum Likelihood Algorithms for Generalized Linear Mixed Models.” 
#' Journal of the American statistical Association, 92(437), 162–170.<doi:10.2307/2291460>
#' 
#' Zeger, S. L., Liang, K.-Y., Albert, P. S. (1988). Models for Longitudinal Data: A Generalized Estimating Equation Approach.
#' Biometrics, 44(4), 1049.<doi:10.2307/2531734>
#' @importFrom Matrix Matrix
#' @export
Model <- R6::R6Class("Model",
                     public = list(
                       #' @field covariance A \link[glmmrBase]{Covariance} object defining the random effects covariance.
                       covariance = NULL,
                       #' @field mean A \link[glmmrBase]{MeanFunction} object, defining the mean function for the model, including the data and covariate design matrix X.
                       mean = NULL,
                       #' @field family One of the family function used in R's glm functions. See \link[stats]{family} for details
                       family = NULL,
                       #' @field weights A vector indicting the weights for the observations.
                       weights = NULL,
                       #' @field trials For binomial family models, the number of trials for each observation. The default is 1 (bernoulli).
                       trials = NULL,
                       #' @field formula The formula for the model. May be empty if separate formulae are specified for the mean and covariance components.
                       formula = NULL,
                       #' @field var_par Scale parameter required for some distributions (Gaussian, Gamma, Beta).
                       var_par = NULL,
                       #' @description
                       #' Sets the model to use or not use "attenuation" when calculating the first-order approximation to
                       #' the covariance matrix.
                       #' @details
                       #' **Attenuation**
                       #' For calculations such as the information matrix, the first-order approximation to the covariance matrix
                       #' proposed by Breslow and Clayton (1993), described above, is used. The approximation is based on the
                       #' marginal quasilikelihood. Zegers, Liang, and Albert (1988) suggest that a better approximation to the
                       #' marginal mean is achieved by "attenuating" the linear predictor. Setting `use` equal to TRUE uses this
                       #' adjustment for calculations using the covariance matrix for non-linear models.
                       #' @param use Logical indicating whether to use "attenuation".
                       #' @return None. Used for effects.
                       use_attenuation = function(use){
                         curr_state <- private$attenuate_parameters
                         if(use){
                           private$attenuate_parameters <- TRUE
                           Model__use_attenuation(private$ptr,TRUE,private$model_type())
                         } else {
                           private$attenuate_parameters <- FALSE
                           Model__use_attenuation(private$ptr,FALSE,private$model_type())
                         }
                         if(private$attenuate_parameters != curr_state){
                           private$genW()
                         }
                       },
                       #' @description
                       #' Return fitted values. Does not account for the random effects. For simulated values based
                       #' on resampling random effects, see also `sim_data()`. To predict the values including random effects at a new location see also
                       #' `predict()`.
                       #' @param type One of either "`link`" for values on the scale of the link function, or "`response`"
                       #' for values on the scale of the response
                       #' @param X (Optional) Fixed effects matrix to generate fitted values
                       #' @param u (Optional) Random effects values at which to generate fitted values
                       #' @param sample Logical. If TRUE then the parameters will be re-sampled from their sampling distribution. Currently only works 
                       #' with existing X matrix and not user supplied matrix X and this will also ignore any provided random effects.
                       #' @param sample_n Integer. If sample is TRUE, then this is the number of samples.
                       #' @return A \link[Matrix]{Matrix} class object containing the predicted values
                       fitted = function(type="link", X, u, sample= FALSE, sample_n = 100){
                         if(missing(X)){
                           if(!sample){
                             Xb <- self$mean$linear_predictor()
                           } else {
                             Xb <- matrix(NA,nrow=self$n(),ncol=sample_n)
                             b_curr <- Model__get_beta(private$ptr,private$model_type())
                             M <- self$information_matrix()
                             ML <- chol(solve(M))
                             for(i in 1:sample_n){
                               u <- rnorm(length(b_curr))
                               Model__update_beta(private$ptr,b_curr + ML%*%u,private$model_type())
                               Xb[,i] <- drop(Model__xb(private$ptr,private$model_type()))
                             }
                             Model__update_beta(private$ptr,b_curr,private$model_type())
                           }
                         } else {
                           Xb <- X%*%self$mean$parameters 
                         }
                         if(!missing(u) & !sample){
                           Xb <- Xb + self$covariance$Z%*%u
                         }
                         if(type=="response"){
                           Xb <- self$family$linkinv(Xb)
                         }
                         return(Xb)
                       },
                       #' @description 
                       #' Generate predictions at new values
                       #' 
                       #' Generates predicted values using a new data set to specify covariance 
                       #' values and values for the variables that define the covariance function.
                       #' The function will return a list with the linear predictor, conditional 
                       #' distribution of the new random effects term conditional on the current estimates
                       #' of the random effects, and some simulated values of the random effects if requested.
                       #' @param newdata A data frame specifying the new data at which to generate predictions
                       #' @param m Number of samples of the random effects to draw
                       #' @param offset Optional vector of offset values for the new data
                       #' @return A list with the linear predictor, parameters (mean and covariance matrices) for
                       #' the conditional distribution of the random effects, and any random effect samples.
                       predict = function(newdata,
                                          offset = rep(0,nrow(newdata)),
                                          m=0
                                          ){
                         preddata <- private$model_data(newdata)
                         out <- Model__predict(private$ptr,as.matrix(preddata),offset,m,private$model_type())
                         return(out)
                       },
                       #' @description
                       #' Create a new Model object
                       #' @param formula An optional model formula containing fixed and random effect terms. If not specified, then
                       #' separate formulae need to be provided to the covariance and mean arguments below.
                       #' @param covariance (Optional) Either a \link[glmmrBase]{Covariance} object, an equivalent list of arguments
                       #' that can be passed to `Covariance` to create a new object, or a vector of parameter values. At a minimum the list must specify a formula.
                       #' If parameters are not included then they are initialised to 0.5. 
                       #' @param mean (Optional) Either a \link[glmmrBase]{MeanFunction} object, an equivalent list of arguments
                       #' that can be passed to `MeanFunction` to create a new object, or a vector of parameter values. At a minimum the list must specify a formula.
                       #' If parameters are not included then they are initialised to 0.
                       #' @param data A data frame with the data required for the mean function and covariance objects. This argument
                       #' can be ignored if data are provided to the covariance or mean arguments either via `Covariance` and `MeanFunction`
                       #' object, or as a member of the list of arguments to both `covariance` and `mean`.
                       #' @param family A family object expressing the distribution and link function of the model, see \link[stats]{family}. This
                       #' argument is optional if the family is provided either via a `MeanFunction` or `MeanFunction`
                       #' objects, or as members of the list of arguments to `mean`. Current accepts \link[stats]{binomial},
                       #' \link[stats]{gaussian}, \link[stats]{Gamma}, \link[stats]{poisson}, and \link[glmmrBase]{Beta}.
                       #' @param var_par (Optional) Scale parameter required for some distributions, including Gaussian. Default is NULL.
                       #' @param offset (Optional) A vector of offset values. Optional - could be provided to the argument to mean instead.
                       #' @param trials (Optional) For binomial family models, the number of trials for each observation. If it is not set, then it will
                       #' default to 1 (a bernoulli model).
                       #' @param weights (Optional) A vector of weights. 
                       #' @return A new Model class object
                       #' @seealso \link[glmmrBase]{nelder}, \link[glmmrBase]{MeanFunction}, \link[glmmrBase]{Covariance}
                       #' @examples
                       #' \dontshow{
                       #' setParallel(FALSE) # for the CRAN check
                       #' }
                       #' #create a data frame describing a cross-sectional parallel cluster
                       #' #randomised trial
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' mod <- Model$new(
                       #'   formula = ~ factor(t) + int - 1 + (1|gr(cl)) + (1|gr(cl,t)),
                       #'   data = df,
                       #'   family = stats::gaussian()
                       #' )
                       #' 
                       #' #here we will specify a cohort study and provide parameter values
                       #' df <- nelder(~ind(20) * t(6))
                       #' df$int <- 0
                       #' df[df$t > 3, 'int'] <- 1
                       #' # the preferred way of specifying with parameter values
                       #' des <- Model$new(
                       #'   formula = ~ int + (1|gr(ind)),
                       #'   covariance = c(0.05),
                       #'   mean = c(1,0.5),
                       #'   data = df,
                       #'   family = stats::poisson()
                       #'   )
                       #' # also works:
                       #' des <- Model$new(
                       #'   covariance = list(
                       #'     formula = ~ (1|gr(ind)),
                       #'     parameters = c(0.05)),
                       #'   mean = list(
                       #'     formula = ~ int,
                       #'     parameters = c(1,0.5)),
                       #'   data = df,
                       #'   family = stats::poisson())
                       #'
                       #' #an example of a spatial grid with two time points
                       #' df <- nelder(~ (x(10)*y(10))*t(2))
                       #' spt_design <- Model$new(covariance = list( formula = ~(1|ar0(t)*fexp(x,y))),
                       #'                          mean = list(formula = ~ 1),
                       #'                          data = df,
                       #'                          family = stats::gaussian())
                       initialize = function(formula,
                                             covariance,
                                             mean,
                                             data = NULL,
                                             family = NULL,
                                             var_par = NULL,
                                             offset = NULL,
                                             weights = NULL,
                                             trials = NULL){

                         if(is.null(family)){
                           stop("No family specified.")
                         } else {
                           self$family <- family
                         }
                         
                         if(!is.null(var_par)){
                           self$var_par <- var_par
                         } else {
                           self$var_par <- 1
                         }

                         if(!missing(formula)){
                           if(!missing(covariance) && is(covariance,"R6"))stop("Do not specify both formula and Covariance class object")
                           if(!missing(mean) && is(mean,"R6"))stop("Do not specify both formula and MeanFunction class object")
                           self$formula <- Reduce(paste,as.character(formula))
                           if(is.null(data)){
                            stop("Data must be specified with a formula")
                           } else {
                             if(missing(covariance) || (!all(is(covariance,"numeric")) & !"parameters"%in%names(covariance))){
                               self$covariance <- Covariance$new(
                                 formula = formula,
                                 data = data
                               )
                             } else {
                               if("parameters"%in%names(covariance)){
                                 self$covariance <- Covariance$new(
                                   formula = formula,
                                   data = data,
                                   parameters = covariance$parameters
                                 )
                               } else if(all(is(covariance,"numeric"))){
                                 self$covariance <- Covariance$new(
                                   formula = formula,
                                   data = data,
                                   parameters = covariance
                                 )
                               } else {
                                 stop("Cannot interpret covariance argument")
                               }
                             }
                             if(missing(mean) || (!"parameters"%in%names(mean) & !all(is(mean,"numeric")))){
                               self$mean <- MeanFunction$new(
                                 formula = formula,
                                 data = data
                               )
                             } else {
                               if("parameters"%in%names(mean)){
                                 self$mean <- MeanFunction$new(
                                   formula = formula,
                                   data = data,
                                   parameters = mean$parameters
                                 )
                               } else if(all(is(mean,"numeric"))){
                                 self$mean <- MeanFunction$new(
                                   formula = formula,
                                   data = data,
                                   parameters = mean
                                 )
                               } else {
                                 stop("Cannot interpret mean argument")
                               }
                             }
                           }
                         } else {
                           if(is(covariance,"R6")){
                             if(is(covariance,"Covariance")){
                               self$covariance <- covariance
                               if(is.null(covariance$data)){
                                 if(is.null(data)){
                                   stop("No data specified in covariance object or call to function.")
                                 } else {
                                   self$covariance$data <- data
                                 }
                               }
                             } else {
                               stop("covariance should be Covariance class or list of appropriate arguments")
                             }
                           } else if(is(covariance,"list")){
                             if(is.null(covariance$formula))stop("A formula must be specified for the covariance")
                             if(is.null(covariance$data) & is.null(data))stop("No data specified in covariance list or call to function.")
                             self$covariance <- Covariance$new(
                               formula= covariance$formula
                             )
                             if(is.null(covariance$data)){
                               self$covariance$data <- data
                             } else {
                               self$covariance$data <- covariance$data
                             }
                             if(!is.null(covariance$parameters))self$covariance$update_parameters(covariance$parameters)
                             if(!is.null(covariance$eff_range))self$covariance$eff_range <- covariance$eff_range
                           }
                           if(is(mean,"R6")){
                             if(is(mean,"MeanFunction")){
                               self$mean <- mean
                             } else {
                               stop("mean should be MeanFunction class or list of appropriate arguments")
                             }
                           } else if(is(mean,"list")){
                             if(is.null(mean$formula))stop("A formula must be specified for the mean function.")
                             if(is.null(mean$data) & is.null(data))stop("No data specified in mean list or call to function.")
                             if(is.null(mean$data)){
                               self$mean <- MeanFunction$new(
                                 formula = mean$formula,
                                 data = data
                               )
                             } else {
                               self$mean <- MeanFunction$new(
                                 formula = mean$formula,
                                 data = mean$data
                               )
                             }
                             if(!is.null(mean$parameters))self$mean$update_parameters(mean$parameters)
                           }
                         }
                         if(is.null(offset)){
                           self$mean$offset <- rep(0,nrow(self$mean$data))
                         } else {
                           self$mean$offset <- offset
                         }
                         if(is.null(weights)){
                           self$weights <- rep(1,nrow(self$mean$data))
                         } else {
                           self$weights <- weights
                         }
                         if(self$family[[1]]=="binomial"){
                           if(is.null(trials) || all(trials == 1)){
                             self$trials <- rep(1,nrow(self$mean$data))
                             self$family[[1]] <- "bernoulli"
                           } else {
                             self$trials <- trials
                           }
                         }
                         private$update_ptr()
                       },
                       #' @description
                       #' Print method for `Model` class
                       #' @details
                       #' Calls the respective print methods of the linked covariance and mean function objects.
                       #' @param ... ignored
                       print = function(){
                         cat("\U2BC8 GLMM Model")
                         cat("\n   \U2BA1 Family :",self$family[[1]])
                         cat("\n   \U2BA1 Link :",self$family[[2]])
                         cat("\n   \U2BA1 Linear predictor")
                         cat("\n   \U2223     \U2BA1 Formula: ~",self$mean$formula)
                         cat("\n   \U2223     \U2BA1 Parameters: ",self$mean$parameters)
                         cat("\n   \U2BA1 Covariance")
                         cat("\n   \U2223     \U2BA1 Terms: ",re_names(self$covariance$formula))
                         if(private$model_type() == 1)cat(" (NNGP)")
                         if(private$model_type() == 2)cat(" (HSGP)")
                         cat("\n   \U2223     \U2BA1 Parameters: ",self$covariance$parameters)
                         cat("\n   \U2223     \U2BA1 N random effects: ",ncol(self$covariance$Z))
                         cat("\n   \U2BA1 N:",self$n())
                       },
                       #' @description
                       #' Returns the number of observations in the model
                       #' @details
                       #' The matrices X and Z both have n rows, where n is the number of observations in the model/design.
                       #' @param ... ignored
                       n = function(...){
                         self$mean$n()
                       },
                       #' @description
                       #' Subsets the design keeping specified observations only
                       #'
                       #' Given a vector of row indices, the corresponding rows will be kept and the
                       #' other rows will be removed from the mean function and covariance
                       #' @param index Integer or vector integers listing the rows to keep
                       #' @return The function updates the object and nothing is returned.
                       subset_rows = function(index){
                         self$mean$subset_rows(index)
                         self$covariance$subset(index)
                         private$update_ptr(TRUE)
                       },
                       #'@description
                       #'Generates a realisation of the design
                       #'
                       #'Generates a single vector of outcome data based upon the
                       #'specified GLMM design. 
                       #'@param type Either 'y' to return just the outcome data, 'data'
                       #' to return a data frame with the simulated outcome data alongside the model data,
                       #' or 'all', which will return a list with simulated outcomes y, matrices X and Z,
                       #' parameters beta, and the values of the simulated random effects.
                       #' @return Either a vector, a data frame, or a list
                       #' @examples
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' \dontshow{
                       #' setParallel(FALSE) # for the CRAN check
                       #' }
                       #' des <- Model$new(
                       #'   covariance = list(
                       #'     formula = ~ (1|gr(cl)*ar0(t)),
                       #'     parameters = c(0.05,0.8)),
                       #'   mean = list(
                       #'     formula = ~ factor(t) + int - 1,
                       #'     parameters = c(rep(0,5),0.6)),
                       #'   data = df,
                       #'   family = stats::binomial()
                       #' )
                       #' ysim <- des$sim_data()
                       sim_data = function(type = "y"){
                         re <- self$covariance$simulate_re()
                         xb <- self$mean$linear_predictor()
                         mu <- xb + drop(as.matrix(self$covariance$Z)%*%re)

                         f <- self$family
                         if(f[1]=="poisson"){
                           if(f[2]=="log"){
                             y <- rpois(self$n(),exp(mu))
                           }
                           if(f[2]=="identity"){
                             y <- rpois(self$n(),mu)
                           }
                         }

                         if(f[1]%in%c("binomial","bernoulli")){
                           if(f[2]=="logit"){
                             y <- rbinom(self$n(),self$trials,exp(mu)/(1+exp(mu)))
                           }
                           if(f[2]=="log"){
                             y <- rbinom(self$n(),self$trials,exp(mu))
                           }
                           if(f[2]=="identity"){
                             y <- rbinom(self$n(),self$trials,mu)
                           }
                           if(f[2]=="probit"){
                             y <- rbinom(self$n(),self$trials,pnorm(mu))
                           }
                         }

                         if(f[1]=="gaussian"){
                           if(f[2]=="identity"){
                             if(is.null(self$var_par))stop("For gaussian(link='identity') provide var_par")
                             y <- rnorm(self$n(),mu,self$var_par/self$weights)
                           }
                           if(f[2]=="log"){
                             if(is.null(self$var_par))stop("For gaussian(link='log') provide var_par")
                             #CHECK THIS IS RIGHT
                             y <- rnorm(self$n(),exp(mu),self$var_par/self$weights)
                           }
                         }


                         if(f[1]=="Gamma"){
                           if(f[2]=="inverse"){
                             if(is.null(self$var_par))stop("For gamma(link='inverse') provide var_par")
                             y <- rgamma(self$n(),shape = self$var_par,rate = self$var_par*mu)
                           }
                           if(f[2]=="log"){
                             if(is.null(self$var_par))stop("For gamma(link='log') provide var_par")
                             y <- rgamma(self$n(),shape = self$var_par,rate = self$var_par/exp(mu))
                           }
                           if(f[2]=="identity"){
                             if(is.null(self$var_par))stop("For gamma(link='identity') provide var_par")
                             y <- rgamma(self$n(),shape = self$var_par,rate = self$var_par/mu)
                           }
                         }

                         if(f[1]=="beta"){
                           if(f[2]=="logit"){
                             if(is.null(self$var_par))stop("For beta(link='logit') provide var_par")
                             logitxb <- exp(mu)/(1+exp(mu))
                             y <- rbeta(self$n(),logitxb*self$var_par,(1-logitxb)*self$var_par)
                           }
                         }

                         if(type=="data.frame"|type=="data")y <- cbind(y,self$mean$data)
                         if(type=="all")y <- list(y = y, X = self$mean$X, beta = self$mean$parameters,
                                                  Z = self$covariance$Z, u = re)
                         return(y)

                       },
                       #' @description
                       #' Updates the parameters of the mean function and/or the covariance function
                       #'
                       #' @details
                       #' Using `update_parameters()` is the preferred way of updating the parameters of the
                       #' mean or covariance objects as opposed to direct assignment, e.g. `self$covariance$parameters <- c(...)`.
                       #' The function calls check functions to automatically update linked matrices with the new parameters.
                       #'
                       #' @param mean.pars (Optional) Vector of new mean function parameters
                       #' @param cov.pars (Optional) Vector of new covariance function(s) parameters
                       #' @param var.par (Optional) A scalar value for var_par
                       #' @examples
                       #' \dontshow{
                       #' setParallel(FALSE) # for the CRAN check
                       #' }
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' des <- Model$new(
                       #'   covariance = list(
                       #'     formula = ~ (1|gr(cl)*ar0(t))),
                       #'   mean = list(
                       #'     formula = ~ factor(t) + int - 1),
                       #'   data = df,
                       #'   family = stats::binomial()
                       #' )
                       #' des$update_parameters(cov.pars = c(0.1,0.9))
                       update_parameters = function(mean.pars = NULL,
                                                    cov.pars = NULL,
                                                    var.par = NULL){
                         if(!is.null(mean.pars)){
                           self$mean$update_parameters(mean.pars)
                           if(!is.null(private$ptr)){
                             Model__update_beta(private$ptr,mean.pars,private$model_type())
                           }
                         }
                         if(!is.null(cov.pars)){
                           #self$covariance$update_parameters(cov.pars)
                           if(!is.null(private$ptr)){
                             Model__update_theta(private$ptr,cov.pars,private$model_type())
                           }
                         }
                         if(!is.null(var.par)){
                           self$var_par <- var.par
                           if(!is.null(private$ptr)){
                             Model__set_var_par(private$ptr,var.par,private$model_type())
                           }
                         }
                       },
                       #' @description
                       #' Generates the information matrix of the GLS estimator
                       #' @param include.re logical indicating whether to return the information matrix including the random effects components (TRUE), 
                       #' or the GLS information matrix for beta only.
                       #' @return A PxP matrix
                       information_matrix = function(include.re = FALSE){
                         if(is.null(private$ptr)){
                           private$update_ptr()
                         }
                         if(include.re & !private$model_type()>0){
                           return(Model__obs_information_matrix(private$ptr,private$model_type()))
                         } else {
                           if(private$model_type()==0){
                             return(Model__information_matrix(private$ptr,private$model_type()))
                           } else {
                             return(Model__information_matrix_crude(private$ptr,private$model_type()))
                           }
                         }
                       },
                       #' @description 
                       #' Returns the robust sandwich variance-covariance matrix for the fixed effect parameters
                       #' @return A PxP matrix
                       sandwich = function(type){
                         if(is.null(private$ptr))private$update_ptr()
                         return(Model__sandwich(private$ptr,private$model_type()))
                       },
                       #' @description 
                       #' Returns a small sample correction. The option "KR" returns the Kenward-Roger bias-corrected variance-covariance matrix 
                       #' for the fixed effect parameters and degrees of freedom. Option "KR2"  returns an improved correction given 
                       #' in Kenward & Roger (2009) <doi:j.csda.2008.12.013>. Note, that the corrected/improved version is invariant 
                       #' under reparameterisation of the covariance, and it will also make no difference if the covariance is linear 
                       #' in parameters. Exchangeable covariance structures in this package (i.e. `gr()`) are parameterised in terms of 
                       #' the variance rather than standard deviation, so the results will be unaffected. Option "sat" returns the "Satterthwaite"
                       #' correction, which only includes corrected degrees of freedom, along with the GLS standard errors.
                       #' @param type Either "KR", "KR2", or "sat", see description.
                       #' @return A PxP matrix
                       small_sample_correction = function(type){
                         if(is.null(private$ptr)){
                           private$update_ptr()
                         }
                         if(!type %in% c("KR","KR2","sat"))stop("type must be either KR, KR2, or sat")
                         ss_type <- ifelse(type == "KR",1,ifelse(type == "KR2",4,5))
                         return(Model__small_sample_correction(private$ptr,ss_type,private$model_type()))
                       },
                       #' @description 
                       #' Returns the inferential statistics (F-stat, p-value) for a modified Box correction <doi:10.1002/sim.4072> for
                       #' Gaussian-identity models.
                       #' @param y Optional. If provided, will update the vector of outcome data. Otherwise it will use the data from 
                       #' the previous model fit.
                       #' @return A data frame.
                       box = function(y){
                         if(!(self$family[[1]]=="gaussian"&self$family[[2]]=="identity"))stop("Box only available for linear models")
                         if(is.null(private$ptr)){
                           private$update_ptr()
                         }
                         if(!missing(y))private$set_y(y)
                         results <- Model__box(private$ptr,private$model_type())
                         results_out <- data.frame(parameter = Model__beta_parameter_names(private$ptr,private$model_type()),
                                                   "F value" = results$test_stat,
                                                   DoF = results$dof,
                                                   scale = results$scale,
                                                   "p value" = results$p_value)
                         return(results_out)
                       },
                       #' @description
                       #' Estimates the power of the design described by the model using the square root
                       #' of the relevant element of the GLS variance matrix:
                       #'
                       #'  \deqn{(X^T\Sigma^{-1}X)^{-1}}
                       #'
                       #' Note that this is equivalent to using the "design effect" for many
                       #' models.
                       #' @param alpha Numeric between zero and one indicating the type I error rate.
                       #' Default of 0.05.
                       #' @param two.sided Logical indicating whether to use a two sided test
                       #' @param alternative For a one-sided test whether the alternative hypothesis is that the
                       #' parameter is positive "pos" or negative "neg"
                       #' @return A data frame describing the parameters, their values, expected standard
                       #' errors and estimated power.
                       #' @examples
                       #' \dontshow{
                       #' setParallel(FALSE) # for the CRAN check
                       #' }
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' des <- Model$new(
                       #'   covariance = list(
                       #'     formula = ~ (1|gr(cl)) + (1|gr(cl,t)),
                       #'     parameters = c(0.05,0.1)),
                       #'   mean = list(
                       #'     formula = ~ factor(t) + int - 1,
                       #'     parameters = c(rep(0,5),0.6)),
                       #'   data = df,
                       #'   family = stats::gaussian(),
                       #'   var_par = 1
                       #' )
                       #' des$power() #power of 0.90 for the int parameter
                       power = function(alpha=0.05,two.sided=TRUE,alternative = "pos"){
                         M <- self$information_matrix()
                         v0 <- solve(M)
                         v0 <- as.vector(sqrt(diag(v0)))
                         if(two.sided){
                           pwr <- pnorm(abs(self$mean$parameters/v0) - qnorm(1-alpha/2))
                         } else {
                           if(alternative == "pos"){
                             pwr <- pnorm(self$mean$parameters/v0 - qnorm(1-alpha/2))
                           } else {
                             pwr <- pnorm(-self$mean$parameters/v0 - qnorm(1-alpha/2))
                           }
                         }

                         res <- data.frame(Value = self$mean$parameters,
                                           SE = v0,
                                           Power = pwr)
                         return(res)
                       },
                       #' @description
                       #' Returns the diagonal of the matrix W used to calculate the covariance matrix approximation
                       #' @return A vector with values of the glm iterated weights
                       w_matrix = function(){
                         if(is.null(private$ptr)){
                           private$update_ptr()
                         }
                         private$genW()
                         return(private$W)
                       },
                       #' @description
                       #' Returns the derivative of the link function with respect to the linear preditor
                       #' @return A vector
                       dh_deta = function(){
                         Q = dlinkdeta(self$fitted(),self$family[[2]])
                         return(Q)
                       },
                       #' @description
                       #' Returns the (approximate) covariance matrix of y
                       #'
                       #' Returns the covariance matrix Sigma. For non-linear models this is an approximation. See Details.
                       #' @param inverse Logical indicating whether to provide the covariance matrix or its inverse
                       #' @return A matrix.
                       Sigma = function(inverse = FALSE){
                         if(is.null(private$ptr)){
                           private$update_ptr()
                         }
                         return(Model__Sigma(private$ptr,inverse,private$model_type()))
                       },
                       #'@description
                       #'Markov Chain Monte Carlo Maximum Likelihood  model fitting
                       #'
                       #'@details
                       #'**MCMCML**
                       #'Fits generalised linear mixed models using one of three algorithms: Markov Chain Newton
                       #'Raphson (MCNR), Markov Chain Expectation Maximisation (MCEM), or Maximum simulated
                       #'likelihood (MSL). All the algorithms are described by McCullagh (1997). For each iteration
                       #'of the algorithm the unobserved random effect terms (\eqn{\gamma}) are simulated
                       #'using Markov Chain Monte Carlo (MCMC) methods (we use Hamiltonian Monte Carlo through Stan),
                       #'and then these values are conditioned on in the subsequent steps to estimate the covariance
                       #'parameters and the mean function parameters (\eqn{\beta}). For all the algorithms,
                       #'the covariance parameter estimates are updated using an expectation maximisation step.
                       #'For the mean function parameters you can either use a Newton Raphson step (MCNR) or
                       #'an expectation maximisation step (MCEM). A simulated likelihood step can be added at the
                       #'end of either MCNR or MCEM, which uses an importance sampling technique to refine the
                       #'parameter estimates.
                       #'
                       #'The accuracy of the algorithm depends on the user specified tolerance. For higher levels of
                       #'tolerance, larger numbers of MCMC samples are likely need to sufficiently reduce Monte Carlo error.
                       #'
                       #' Options for the MCMC sampler are set by changing the values in `self$mcmc_options`. The information printed to the console
                       #' during model fitting can be controlled with the `self$set_trace()` function.
                       #' 
                       #' To provide weights for the model fitting, store them in self$weights. To set the number of 
                       #' trials for binomial models, set self$trials.
                       #' 
                       #'@param y A numeric vector of outcome data
                       #'@param method The MCML algorithm to use, either `mcem` or `mcnr`, see Details. Default is `mcem`.
                       #'@param sim.lik.step Logical. Either TRUE (conduct a simulated likelihood step at the end of the algorithm), or FALSE (does
                       #'not do this step), defaults to FALSE.
                       #'@param tol Numeric value, tolerance of the MCML algorithm, the maximum difference in parameter estimates
                       #'between iterations at which to stop the algorithm.
                       #'@param max.iter Integer. The maximum number of iterations of the MCML algorithm.
                       #'@param se String. Type of standard error and/or inferential statistics to return. Options are "gls" for GLS standard errors (the default),
                       #' "robust" for robust standard errors, "kr" for original Kenward-Roger bias corrected standard errors, 
                       #' "kr2" for the improved Kenward-Roger correction, "sat" for Satterthwaite degrees of freedom correction (this is the same 
                       #' degrees of freedom correction as Kenward-Roger, but with GLS standard errors), "box" to use a modified Box correction (does not return confidence intervals),
                       #' "bw" to use GLS standard errors with a between-within correction to the degrees of freedom, "bwrobust" to use robust 
                       #' standard errors with between-within correction to the degrees of freedom.
                       #'@param usestan Logical whether to use Stan (through the package `cmdstanr`) for the MCMC sampling. If FALSE then
                       #'the internal Hamiltonian Monte Carlo sampler will be used instead. We recommend Stan over the internal sampler as
                       #'it generally produces a larger number of effective samplers per unit time, especially for more complex
                       #'covariance functions.
                       #'@param se.theta Logical. Whether to calculate the standard errors for the covariance parameters. This step is a slow part
                       #' of the calculation, so can be disabled if required in larger models. Has no effect for Kenward-Roger standard errors.
                       #'@param lower.bound Optional. Vector of lower bounds for the fixed effect parameters. To apply bounds use MCEM.
                       #'@param upper.bound Optional. Vector of upper bounds for the fixed effect parameters. To apply bounds use MCEM.
                       #'@param lower.bound.theta Optional. Vector of lower bounds for the covariance parameters. 
                       #'@param upper.bound.theta Optional. Vector of upper bounds for the covariance parameters. 
                       #'@return A `mcml` object
                       #'@seealso \link[glmmrBase]{Model}, \link[glmmrBase]{Covariance}, \link[glmmrBase]{MeanFunction}
                       #'@examples
                       #'\dontrun{
                       #' #create example data with six clusters, five time periods, and five people per cluster-period
                       #' df <- nelder(~(cl(6)*t(5)) > ind(5))
                       #' # parallel trial design intervention indicator
                       #' df$int <- 0
                       #' df[df$cl > 3, 'int'] <- 1 
                       #' # specify parameter values in the call for the data simulation below
                       #' des <- Model$new(
                       #'   formula= ~ factor(t) + int - 1 +(1|gr(cl)*ar0(t)),
                       #'   covariance = list(parameters = c(0.05,0.7)),
                       #'   mean = list(parameters = c(rep(0,5),0.2)),
                       #'   data = df,
                       #'   family = gaussian(),
                       #'   var_par = 1
                       #' )
                       #' ysim <- des$sim_data() # simulate some data from the model
                       #' fit1 <- des$MCML(y = ysim,method="mcnr",usestan=FALSE) # don't use Stan
                       #' #fits the models using Stan
                       #' fit2 <- des$MCML(y = ysim, method="mcnr")
                       #'  #adds a simulated likelihood step after the MCEM algorithm
                       #' fit3 <- des$MCML(y = ysim, sim.lik.step = TRUE)
                       #'
                       #'  # we could use LA to find better starting values
                       #' fit4 <- des$LA(y=ysim)
                       #' # the fit parameter values are stored in the internal model class object
                       #' fit5 <- des$MCML(y = ysim, method="mcnr") # it should converge much more quickly
                       #'}
                       #'@md
                       MCML = function(y,
                                       method = "mcnr",
                                       sim.lik.step = FALSE,
                                       tol = 1e-2,
                                       max.iter = 30,
                                       se = "gls",
                                       usestan = TRUE,
                                       se.theta = TRUE,
                                       lower.bound = NULL,
                                       upper.bound = NULL,
                                       lower.bound.theta = NULL,
                                       upper.bound.theta = NULL){
                         private$verify_data(y)
                         private$set_y(y)
                         Model__use_attenuation(private$ptr,private$attenuate_parameters,private$model_type())
                         if(!se %in% c("gls","kr","kr2","bw","sat","bwrobust","box"))stop("Option se not recognised")
                         if(self$family[[1]]%in%c("Gamma","beta") & se %in% c("kr","kr2","sat"))stop("KR standard errors are not currently available with gamma or beta families")
                         if(se != "gls" & private$model_type() != 0)stop("Only GLS standard errors supported for GP approximations.")
                         if(se == "box" & !(self$family[[1]]=="gaussian"&self$family[[2]]=="identity"))stop("Box only available for linear models")
                         if(!usestan){
                           Model__mcmc_set_lambda(private$ptr,self$mcmc_options$lambda,private$model_type())
                           Model__mcmc_set_max_steps(private$ptr,self$mcmc_options$maxsteps,private$model_type())
                           Model__mcmc_set_refresh(private$ptr,self$mcmc_options$refresh,private$model_type())
                         }
                         if(!is.null(lower.bound)){
                           Model__set_bound(private$ptr,lower.bound,TRUE,TRUE,private$model_type())
                         }
                         if(!is.null(upper.bound)){
                           Model__set_bound(private$ptr,upper.bound,TRUE,FALSE,private$model_type())
                         }
                         if(!is.null(lower.bound.theta)){
                           if(any(lower.bound.theta < 0))stop("Theta lower bound cannot be negative")
                           Model__set_bound(private$ptr,lower.bound.theta,FALSE,TRUE,private$model_type())
                         }
                         if(!is.null(upper.bound.theta)){
                           Model__set_bound(private$ptr,upper.bound.theta,FALSE,FALSE,private$model_type())
                         }
                         beta <- self$mean$parameters
                         theta <- self$covariance$parameters
                         var_par <- self$var_par
                         var_par_family <- I(self$family[[1]]%in%c("gaussian","Gamma","beta"))
                         ncovpar <- ifelse(var_par_family,length(theta)+1,length(theta))
                         all_pars <- c(beta,theta)
                         if(var_par_family)all_pars <- c(all_pars,var_par)
                         all_pars_new <- rep(1,length(all_pars))
                         var_par_new <- var_par
                         if(private$trace >= 1)message(paste0("using method: ",method))
                         if(private$trace >= 1)cat("\nStart: ",all_pars,"\n")
                         niter <- self$mcmc_options$samps
                         invfunc <- self$family$linkinv
                         L <- Matrix::Matrix(Model__L(private$ptr,private$model_type()))
                         #parse family
                         file_type <- mcnr_family(self$family)
                         ## set up sampler
                         if(usestan){
                           if(!requireNamespace("cmdstanr")){
                             stop("cmdstanr is required to use Stan for sampling. See https://mc-stan.org/cmdstanr/ for details on how to install.
                                    Set option usestan=FALSE to use the in-built MCMC sampler.")
                           } else {
                             if(private$trace >= 1)message("If this is the first time running this model, it will be compiled by cmdstan.")
                             model_file <- system.file("stan",
                                                       file_type$file,
                                                       package = "glmmrBase",
                                                       mustWork = TRUE)
                             mod <- suppressMessages(cmdstanr::cmdstan_model(model_file))
                           }
                         }
                         data <- list(
                           N = self$n(),
                           Q = Model__Q(private$ptr,private$model_type()),
                           Xb = Model__xb(private$ptr,private$model_type()),
                           Z = Model__ZL(private$ptr,private$model_type()),
                           y = y,
                           type=as.numeric(file_type$type)
                         )
                         if(self$family[[1]]=="gaussian")data <- append(data,list(sigma = self$var_par/self$weights))
                         if(self$family[[1]]=="binomial")data <- append(data,list(n = self$trials))
                         if(self$family[[1]]%in%c("beta","Gamma"))data <- append(data,list(var_par = self$var_par))
                         iter <- 0
                         while(any(abs(all_pars-all_pars_new)>tol)&iter < max.iter){
                           all_pars <- all_pars_new
                           iter <- iter + 1
                           if(private$trace >= 1)cat("\nIter: ",iter,"\n",Reduce(paste0,rep("-",40)))
                           if(private$trace == 2)t1 <- Sys.time()
                           if(usestan){
                             data$Xb <-  Model__xb(private$ptr,private$model_type())
                             data$Z <- Model__ZL(private$ptr,private$model_type())
                             if(self$family[[1]]=="gaussian")data$sigma = var_par_new/self$weights
                             if(self$family[[1]]%in%c("beta","Gamma"))data$var_par = var_par_new
                              capture.output(fit <- mod$sample(data = data,
                                                    chains = 1,
                                                    iter_warmup = self$mcmc_options$warmup,
                                                    iter_sampling = self$mcmc_options$samps,
                                                    refresh = 0),
                                  file=tempfile())
                             dsamps <- fit$draws("gamma",format = "matrix")
                             class(dsamps) <- "matrix"
                             Model__update_u(private$ptr,as.matrix(t(dsamps)),private$model_type())
                           } else {
                             Model__mcmc_sample(private$ptr,
                                                 self$mcmc_options$warmup,
                                                 self$mcmc_options$samps,
                                                 self$mcmc_options$adapt,private$model_type())
                           }
                           if(private$trace==2)t2 <- Sys.time()
                           if(private$trace==2)cat("\nMCMC sampling took: ",t2-t1,"s")
                           ## ADD IN RSTAN FUNCTIONALITY ONCE PARALLEL METHODS AVAILABLE IN RSTAN
                           
                           if(method=="mcem"){
                             Model__ml_beta(private$ptr,private$model_type())
                           } else {
                             Model__nr_beta(private$ptr,private$model_type())
                           }
                           Model__ml_theta(private$ptr,private$model_type())
                           beta_new <- Model__get_beta(private$ptr,private$model_type())
                           theta_new <- Model__get_theta(private$ptr,private$model_type())
                           var_par_new <- Model__get_var_par(private$ptr,private$model_type())
                           all_pars_new <- c(beta_new,theta_new)
                           if(var_par_family)all_pars_new <- c(all_pars_new,var_par_new)
                           if(private$trace==2)t3 <- Sys.time()
                           if(private$trace==2)cat("\nModel fitting took: ",t3-t2,"s")
                           if(private$trace >= 1){
                             cat("\nBeta: ", beta_new)
                             cat("\nTheta: ", theta_new)
                             if(var_par_family)cat("\nSigma: ",var_par_new)
                             cat("\nMax. diff: ", round(max(abs(all_pars-all_pars_new)),5))
                             cat("\n",Reduce(paste0,rep("-",40)))
                           }
                         }
                         not_conv <- iter > max.iter|any(abs(all_pars-all_pars_new)>tol)
                         if(not_conv)message(paste0("algorithm not converged. Max. difference between iterations :",round(max(abs(all_pars-all_pars_new)),4)))
                         if(sim.lik.step){
                           if(private$trace >= 1)cat("\n\n")
                           if(private$trace >= 1)message("Optimising simulated likelihood")
                           Model__ml_all(private$ptr,private$model_type())
                           beta_new <- Model__get_beta(private$ptr,private$model_type())
                           theta_new <- Model__get_theta(private$ptr,private$model_type())
                           var_par_new <- Model__get_var_par(private$ptr,private$model_type())
                         }
                         self$update_parameters(mean.pars = beta_new,
                                                cov.pars = theta_new)
                         if(private$trace >= 1)cat("\n\nCalculating standard errors...\n")
                         self$var_par <- var_par_new
                         u <- Model__u(private$ptr, TRUE,private$model_type())
                         if(private$model_type()==0){
                           if(se == "gls" || se == "bw" || se == "box"){
                             M <- Matrix::solve(Model__obs_information_matrix(private$ptr,private$model_type()))[1:length(beta),1:length(beta)]
                             if(se.theta){
                               SE_theta <- tryCatch(sqrt(diag(solve(Model__infomat_theta(private$ptr,private$model_type())))), error = rep(NA, ncovpar))
                             } else {
                               SE_theta <- rep(NA, ncovpar)
                             }
                           } else if(se == "robust" || se == "bwrobust"){
                             M <- Model__sandwich(private$ptr,private$model_type())
                             if(se.theta){
                               SE_theta <- tryCatch(sqrt(diag(solve(Model__infomat_theta(private$ptr,private$model_type())))), error = rep(NA, ncovpar))
                             } else {
                               SE_theta <- rep(NA, ncovpar)
                             }
                           } else if(se == "kr" || se == "kr2" || se == "sat"){
                             ss_type <- ifelse(se=="kr",1,ifelse(se=="kr2",4,5))
                             Mout <- Model__small_sample_correction(private$ptr,ss_type,private$model_type())
                             M <- Mout[[1]]
                             SE_theta <- sqrt(diag(Mout[[2]]))
                           } 
                         } else {
                           # crudely calculate the information matrix for GP approximations - this will be integrated into the main
                           # library in future versions, but can cause error/crash with the above methods
                           M <- Model__information_matrix_crude(private$ptr,private$model_type())
                           nB <- nrow(M)
                           M <- tryCatch(solve(M), error = matrix(NA,nrow = nB,ncol=nB))
                           SE_theta <- rep(NA, ncovpar)
                         }
                         SE <- sqrt(diag(M))
                         repar_table <- self$covariance$parameter_table()
                         beta_names <- Model__beta_parameter_names(private$ptr,private$model_type())
                         theta_names <- repar_table$term
                         if(self$family[[1]]%in%c("Gamma","beta")){
                           mf_pars_names <- c(beta_names,theta_names,"sigma")
                           SE <- c(SE,rep(NA,length(theta_new)+1))
                         } else {
                           mf_pars_names <- c(beta_names,theta_names)
                           if(self$family[[1]]=="gaussian") mf_pars_names <- c(mf_pars_names,"sigma")
                           SE <- c(SE,SE_theta)
                         }
                         res <- data.frame(par = c(mf_pars_names,paste0("d",1:nrow(u))),
                                           est = c(all_pars_new,rowMeans(u)),
                                           SE=c(SE,rep(NA,nrow(u))),
                                           t = NA,
                                           p = NA,
                                           lower = NA,
                                           upper = NA)
                         dof <- rep(self$n(),length(beta))
                         if(se == "kr" || se == "kr2" || se == "sat"){
                           for(i in 1:length(beta)){
                             if(!is.na(res$SE[i])){
                               res$t[i] <- (res$est[i]/res$SE[i])#*sqrt(lambda)
                               res$p[i] <- 2*(1-stats::pt(abs(res$t[i]),Mout$dof[i],lower.tail=TRUE))
                               res$lower[i] <- res$est[i] - stats::qt(0.975,Mout$dof[i],lower.tail=TRUE)*res$SE[i]
                               res$upper[i] <- res$est[i] + stats::qt(0.975,Mout$dof[i],lower.tail=TRUE)*res$SE[i]
                             }
                             dof[i] <- Mout$dof[i]
                           }
                         } else if(se=="bw" || se == "bwrobust" ){
                           res$t <- res$est/res$SE
                           bwdof <- sum(repar_table$count) - length(beta)
                           res$p <- 2*(1-stats::pt(abs(res$t),bwdof,lower.tail=TRUE))
                           res$lower <- res$est - qt(1-0.05/2,bwdof,lower.tail=TRUE)*res$SE
                           res$upper <- res$est + qt(1-0.05/2,bwdof,lower.tail=TRUE)*res$SE
                           dof <- rep(bwdof,length(beta))
                         } else if(se == "box"){
                           box_result <- Model__box(private$ptr,private$model_type())
                           res$t <- box_result$test_stat
                           res$p <- box_result$p_value
                           dof <- data.frame(dof = box_result$dof, scale = box_result$scale)
                         } else {
                           res$t <- res$est/res$SE
                           res$p <- 2*(1-stats::pnorm(abs(res$t)))
                           res$lower <- res$est - qnorm(1-0.05/2)*res$SE
                           res$upper <- res$est + qnorm(1-0.05/2)*res$SE
                         }
                         repar_table <- repar_table[!duplicated(repar_table$id),]
                         rownames(u) <- rep(repar_table$term,repar_table$count)
                         aic <- ifelse(private$model_type()==0 , Model__aic(private$ptr,private$model_type()),NA)
                         xb <- Model__xb(private$ptr,private$model_type())
                         zd <- self$covariance$Z %*% rowMeans(u)
                         wdiag <- Matrix::diag(self$w_matrix())
                         total_var <- var(Matrix::drop(xb)) + var(Matrix::drop(zd)) + mean(wdiag)
                         condR2 <- (var(Matrix::drop(xb)) + var(Matrix::drop(zd)))/total_var
                         margR2 <- var(Matrix::drop(xb))/total_var
                         out <- list(coefficients = res,
                                     converged = !not_conv,
                                     method = method,
                                     m = dim(u)[2],
                                     tol = tol,
                                     sim_lik = sim.lik.step,
                                     aic = aic,
                                     se=se,
                                     Rsq = c(cond = condR2,marg=margR2),
                                     logl = Model__log_likelihood(private$ptr,private$model_type()),
                                     mean_form = self$mean$formula,
                                     cov_form = self$covariance$formula,
                                     family = self$family[[1]],
                                     link = self$family[[2]],
                                     re.samps = u,
                                     iter = iter,
                                     dof = dof,
                                     P = length(self$mean$parameters),
                                     Q = length(self$covariance$parameters),
                                     var_par_family = var_par_family,
                                     y=y)
                         class(out) <- "mcml"
                         return(out)
                       },
                       #'@description
                       #'Maximum Likelihood model fitting with Laplace Approximation
                       #'
                       #'@details
                       #'**Laplace approximation**
                       #'Fits generalised linear mixed models using Laplace approximation to the log likelihood. For non-Gaussian models
                       #'the covariance matrix is approximated using the first order approximation based on the marginal
                       #'quasilikelihood proposed by Breslow and Clayton (1993). The marginal mean in this approximation
                       #'can be further adjusted following the proposal of Zeger et al (1988), use the member function `use_attenuated()` in this
                       #'class, see \link[glmmrBase]{Model}. To provide weights for the model fitting, store them in self$weights. To 
                       #'set the number of trials for binomial models, set self$trials. To control the information printed to the console 
                       #' during model fitting use the `self$set_trace()` function.
                       #'
                       #'@param y A numeric vector of outcome data
                       #'@param start Optional. A numeric vector indicating starting values for the model parameters.
                       #'@param method String. Either "nloptim" for non-linear optimisation, or "nr" for Newton-Raphson (default) algorithm
                       #'@param se String. Type of standard error and/or inferential statistics to return. Options are "gls" for GLS standard errors (the default),
                       #' "robust" for robust standard errors, "kr" for original Kenward-Roger bias corrected standard errors, 
                       #' "kr2" for the improved Kenward-Roger correction, "sat" for Satterthwaite degrees of freedom correction (this is the same 
                       #' degrees of freedom correction as Kenward-Roger, but with GLS standard errors)"box" to use a modified Box correction (does not return confidence intervals),
                       #' "bw" to use GLS standard errors with a between-within correction to the degrees of freedom, "bwrobust" to use robust 
                       #' standard errors with between-within correction to the degrees of freedom. 
                       #' Note that Kenward-Roger assumes REML estimates, which are not currently provided by this function.
                       #'@param max.iter Maximum number of algorithm iterations, default 20.
                       #'@param tol Maximum difference between successive iterations at which to terminate the algorithm
                       #'@param se.theta Logical. Whether to calculate the standard errors for the covariance parameters. This step is a slow part
                       #' of the calculation, so can be disabled if required in larger models. Has no effect for Kenward-Roger standard errors.
                       #'@param lower.bound Optional. Vector of lower bounds for the fixed effect parameters. To apply bounds use nloptim.
                       #'@param upper.bound Optional. Vector of upper bounds for the fixed effect parameters. To apply bounds use nloptim.
                       #'@param lower.bound.theta Optional. Vector of lower bounds for the covariance parameters. 
                       #'@param upper.bound.theta Optional. Vector of upper bounds for the covariance parameters. 
                       #'@return A `mcml` object
                       #' @seealso \link[glmmrBase]{Model}, \link[glmmrBase]{Covariance}, \link[glmmrBase]{MeanFunction}
                       #'@examples
                       #' \dontshow{
                       #' setParallel(FALSE) # for the CRAN check
                       #' }
                       #' #create example data with six clusters, five time periods, and five people per cluster-period
                       #' df <- nelder(~(cl(6)*t(5)) > ind(5))
                       #' # parallel trial design intervention indicator
                       #' df$int <- 0
                       #' df[df$cl > 3, 'int'] <- 1 
                       #' # specify parameter values in the call for the data simulation below
                       #' des <- Model$new(
                       #'   formula = ~ factor(t) + int - 1 + (1|gr(cl)*ar0(t)),
                       #'   covariance = list( parameters = c(0.05,0.7)),
                       #'   mean = list(parameters = c(rep(0,5),-0.2)),
                       #'   data = df,
                       #'   family = stats::binomial()
                       #' )
                       #' ysim <- des$sim_data() # simulate some data from the model
                       #' fit1 <- des$LA(y = ysim)
                       #'@md
                       LA = function(y,
                                     start,
                                     method = "nr",
                                     se = "gls",
                                     max.iter = 40,
                                     tol = 1e-4,
                                     se.theta = TRUE,
                                     lower.bound = NULL,
                                     upper.bound = NULL,
                                     lower.bound.theta = NULL,
                                     upper.bound.theta = NULL){
                         private$verify_data(y)
                         private$set_y(y)
                         Model__use_attenuation(private$ptr,private$attenuate_parameters,private$model_type())
                         if(!se %in% c("gls","kr","kr2","bw","sat","bwrobust","box"))stop("Option se not recognised")
                         if(self$family[[1]]%in%c("Gamma","beta") & (se == "kr"||se=="kr2"||se=="sat"))stop("KR standard errors are not currently available with gamma or beta families")
                         if(!method%in%c("nloptim","nr"))stop("method should be either nr or nloptim")
                         if(se == "box" & !(self$family[[1]]=="gaussian"&self$family[[2]]=="identity"))stop("Box only available for linear models")
                         if(!is.null(lower.bound)){
                           Model__set_lower_bound(private$ptr,lower.bound,private$model_type())
                         }
                         if(!is.null(upper.bound)){
                           Model__set_upper_bound(private$ptr,upper.bound,private$model_type())
                         }
                         if(!is.null(lower.bound.theta)){
                           if(any(lower.bound.theta < 0))stop("Theta lower bound cannot be negative")
                           Model__set_bound(private$ptr,lower.bound.theta,FALSE,TRUE,private$model_type())
                         }
                         if(!is.null(upper.bound.theta)){
                           Model__set_bound(private$ptr,upper.bound.theta,FALSE,FALSE,private$model_type())
                         }
                         var_par_family <- I(self$family[[1]]%in%c("gaussian","Gamma","beta"))
                         beta <- self$mean$parameters
                         theta <- self$covariance$parameters
                         ncovpar <- ifelse(var_par_family,length(theta)+1,length(theta))
                         var_par <- self$var_par
                         all_pars <- c(beta,theta)
                         if(var_par_family)all_pars <- c(all_pars,var_par)
                         all_pars_new <- rep(1,length(all_pars))
                         iter <- 0
                         while(any(abs(all_pars-all_pars_new)>tol)&iter < max.iter){
                           all_pars <- all_pars_new
                           iter <- iter + 1
                           if(private$trace >= 1)cat("\nIter: ",iter,"\n",Reduce(paste0,rep("-",40)))
                           if(method=="nr"){
                             Model__laplace_nr_beta_u(private$ptr,private$model_type())
                           } else {
                             Model__laplace_ml_beta_u(private$ptr,private$model_type())
                           }
                           Model__laplace_ml_theta(private$ptr,private$model_type())
                           beta_new <- Model__get_beta(private$ptr,private$model_type())
                           theta_new <- Model__get_theta(private$ptr,private$model_type())
                           var_par_new <- Model__get_var_par(private$ptr,private$model_type())
                           all_pars_new <- c(beta_new,theta_new)
                           if(var_par_family)all_pars_new <- c(all_pars_new,var_par)
                           if(private$trace >= 1){
                             cat("\nBeta: ", beta_new)
                             cat("\nTheta: ", theta_new)
                             if(var_par_family)cat("\nSigma: ",var_par_new)
                             cat("\nMax. diff: ", round(max(abs(all_pars-all_pars_new)),5))
                             cat("\n",Reduce(paste0,rep("-",40)))
                           }
                         }
                         not_conv <- iter > max.iter|any(abs(all_pars-all_pars_new)>tol)
                         if(not_conv)message(paste0("algorithm not converged. Max. difference between iterations :",round(max(abs(all_pars-all_pars_new)),4)))
                         #Model__laplace_ml_beta_theta(private$ptr)
                         beta_new <- Model__get_beta(private$ptr,private$model_type())
                         theta_new <- Model__get_theta(private$ptr,private$model_type())
                         var_par_new <- Model__get_var_par(private$ptr,private$model_type())
                         all_pars_new <- c(beta_new,theta_new)
                         if(var_par_family)all_pars_new <- c(all_pars_new,var_par_new)
                         self$update_parameters(mean.pars = beta_new,
                                                cov.pars = theta_new)
                         self$var_par <- var_par_new
                         u <- Model__u(private$ptr,TRUE,private$model_type())
                         if(private$trace >= 1)cat("\n\nCalculating standard errors...\n")
                         if(se == "gls" || se =="bw" || se == "box"){
                           M <- Matrix::solve(Model__obs_information_matrix(private$ptr,private$model_type()))[1:length(beta),1:length(beta)]
                           if(se.theta){
                             SE_theta <- tryCatch(sqrt(diag(solve(Model__infomat_theta(private$ptr,private$model_type())))), error = rep(NA, ncovpar))
                           } else {
                             SE_theta <- rep(NA, ncovpar)
                           }
                         } else if(se == "robust" || se == "bwrobust" ){
                           M <- Model__sandwich(private$ptr,private$model_type())
                           if(se.theta){
                             SE_theta <- tryCatch(sqrt(diag(solve(Model__infomat_theta(private$ptr,private$model_type())))), error = rep(NA, ncovpar))
                           } else {
                             SE_theta <- rep(NA, ncovpar)
                           }
                         } else if(se == "kr" || se == "kr2" || se == "sat"){
                           krtype <- ifelse(se=="kr",1,ifelse(se=="kr2",4,5))
                           Mout <- Model__small_sample_correction(private$ptr,krtype,private$model_type())
                           M <- Mout[[1]]
                           SE_theta <- sqrt(diag(Mout[[2]]))
                         }
                         SE <- sqrt(Matrix::diag(M))
                         repar_table <- self$covariance$parameter_table()
                         beta_names <- Model__beta_parameter_names(private$ptr,private$model_type())
                         theta_names <- repar_table$term
                         if(self$family[[1]]%in%c("Gamma","beta")){
                           mf_pars_names <- c(beta_names,theta_names,"sigma")
                           SE <- c(SE,rep(NA,length(theta_new)+1))
                         } else {
                           mf_pars_names <- c(beta_names,theta_names)
                           if(self$family[[1]]=="gaussian") mf_pars_names <- c(mf_pars_names,"sigma")
                           SE <- c(SE,SE_theta)
                         }
                         res <- data.frame(par = c(mf_pars_names,paste0("d",1:nrow(u))),
                                           est = c(all_pars_new,rowMeans(u)),
                                           SE=c(SE,rep(NA,nrow(u))),
                                           t = NA,
                                           p = NA,
                                           lower = NA,
                                           upper = NA)
                         
                         dof <- rep(self$n(),length(beta))
                         if(se == "kr" || se == "kr2" || se == "sat"){
                           for(i in 1:length(beta)){
                             if(!is.na(res$SE[i])){
                               res$t[i] <- (res$est[i]/res$SE[i])#*sqrt(lambda)
                               res$p[i] <- 2*(1-stats::pt(abs(res$t[i]),Mout$dof[i],lower.tail=TRUE))
                               res$lower[i] <- res$est[i] - stats::qt(0.975,Mout$dof[i],lower.tail=TRUE)*res$SE[i]
                               res$upper[i] <- res$est[i] + stats::qt(0.975,Mout$dof[i],lower.tail=TRUE)*res$SE[i]
                             }
                             dof[i] <- Mout$dof[i]
                           }
                         } else if(se=="bw" || se == "bwrobust" ){
                           res$t <- res$est/res$SE
                           bwdof <- sum(repar_table$count) - length(beta)
                           res$p <- 2*(1-stats::pt(abs(res$t),bwdof,lower.tail=TRUE))
                           res$lower <- res$est - qt(1-0.05/2,bwdof,lower.tail=TRUE)*res$SE
                           res$upper <- res$est + qt(1-0.05/2,bwdof,lower.tail=TRUE)*res$SE
                           dof <- rep(bwdof,length(beta))
                         } else if (se == "box") {
                           box_result <- Model__box(private$ptr,private$model_type())
                           res$t <- box_result$test_stat
                           res$p <- box_result$p_value
                           dof <- data.frame(dof = box_result$dof, scale = box_result$scale)
                         } else {
                           res$t <- res$est/res$SE
                           res$p <- 2*(1-stats::pnorm(abs(res$t)))
                           res$lower <- res$est - qnorm(1-0.05/2)*res$SE
                           res$upper <- res$est + qnorm(1-0.05/2)*res$SE
                         }
                         repar_table <- repar_table[!duplicated(repar_table$id),]
                         rownames(u) <- rep(repar_table$term,repar_table$count)
                         aic <- ifelse(private$model_type()==0 , Model__aic(private$ptr,private$model_type()),NA)
                         xb <- Model__xb(private$ptr,private$model_type())
                         zd <- self$covariance$Z %*% u
                         wdiag <- Matrix::diag(self$w_matrix())
                         total_var <- var(Matrix::drop(xb)) + var(Matrix::drop(zd)) + mean(wdiag)
                         condR2 <- (var(Matrix::drop(xb)) + var(Matrix::drop(zd)))/total_var
                         margR2 <- var(Matrix::drop(xb))/total_var
                         out <- list(coefficients = res,
                                     converged = !not_conv,
                                     method = method,
                                     m = dim(u)[2],
                                     tol = tol,
                                     sim_lik = FALSE,
                                     aic = aic,
                                     se =se ,
                                     Rsq = c(cond = condR2,marg=margR2),
                                     mean_form = self$mean$formula,
                                     cov_form = self$covariance$formula,
                                     logl = Model__log_likelihood(private$ptr,private$model_type()),
                                     family = self$family[[1]],
                                     link = self$family[[2]],
                                     re.samps = u,
                                     iter = iter,
                                     dof = dof,
                                     P = length(self$mean$parameters),
                                     Q = length(self$covariance$parameters),
                                     var_par_family = var_par_family,
                                     y = y)
                         class(out) <- "mcml"
                         return(out)
                       },
                       #' @description 
                       #' Set whether to use sparse matrix methods for model calculations and fitting.
                       #' By default the model does not use sparse matrix methods.
                       #' @param sparse Logical indicating whether to use sparse matrix methods
                       #' @param amd Logical indicating whether to use and Approximate Minimum Degree algorithm to calculate an efficient permutation matrix so 
                       #' that the Cholesky decomposition of PAP^T is calculated rather than A.
                       #' @return None, called for effects
                       sparse = function(sparse = TRUE, amd = TRUE){
                         if(!is.null(private$ptr)){
                           if(private$model_type() == 1){
                             message("Sparse has no effect with NNGP models")
                           } else {
                             if(sparse){
                               Model__make_sparse(private$ptr,amd,private$model_type())
                             } else {
                               Model__make_dense(private$ptr,private$model_type())
                             }
                             self$covariance$sparse(sparse,amd)
                           }
                           private$useSparse = sparse
                         } 
                       },
                       #' @description 
                       #' Generate an MCMC sample of the random effects
                       #' @param y Numeric vector of outcome data
                       #' @param usestan Logical whether to use Stan (through the package `cmdstanr`) for the MCMC sampling. If FALSE then
                       #'the internal Hamiltonian Monte Carlo sampler will be used instead. We recommend Stan over the internal sampler as
                       #'it generally produces a larger number of effective samplers per unit time, especially for more complex
                       #'covariance functions.
                       #' @return A matrix of samples of the random effects
                       mcmc_sample = function(y,usestan = TRUE){
                         private$verify_data(y)
                         private$set_y(y)
                         if(usestan){
                           file_type <- mcnr_family(self$family)
                           if(!requireNamespace("cmdstanr")){
                             stop("cmdstanr is required to use Stan for sampling. See https://mc-stan.org/cmdstanr/ for details on how to install.\n
                                    Set option usestan=FALSE to use the in-built MCMC sampler.")
                           } else {
                             if(private$trace >= 1)message("If this is the first time running this model, it will be compiled by cmdstan.")
                             model_file <- system.file("stan",
                                                       file_type$file,
                                                       package = "glmmrBase",
                                                       mustWork = TRUE)
                             mod <- suppressMessages(cmdstanr::cmdstan_model(model_file))
                           }
                           data <- list(
                             N = self$n(),
                             Q = Model__Q(private$ptr,private$model_type()),
                             Xb = Model__xb(private$ptr,private$model_type()),
                             Z = Model__ZL(private$ptr,private$model_type()),
                             y = y,
                             type=as.numeric(file_type$type)
                           )
                           if(self$family[[1]]=="gaussian")data <- append(data,list(sigma = self$var_par/self$weights))
                           if(self$family[[1]]=="binomial")data <- append(data,list(n = self$trials))
                           if(self$family[[1]]%in%c("beta","Gamma"))data <- append(data,list(var_par = self$var_par))
                           if(private$trace >= 1){
                             fit <- mod$sample(data = data,
                                               chains = 1,
                                               iter_warmup = self$mcmc_options$warmup,
                                               iter_sampling = self$mcmc_options$samps,
                                               refresh = self$mcmc_options$refresh)
                           } else {
                             capture.output(fit <- mod$sample(data = data,
                                                              chains = 1,
                                                              iter_warmup = self$mcmc_options$warmup,
                                                              iter_sampling = self$mcmc_options$samps,
                                                              refresh = 0),
                                            file=tempfile())
                           }
                           dsamps <- fit$draws("gamma",format = "matrix")
                           class(dsamps) <- "matrix"
                           Model__update_u(private$ptr,as.matrix(t(dsamps)),private$model_type())
                           dsamps <- Matrix::Matrix(Model__L(private$ptr, private$model_type()) %*% Matrix::t(dsamps)) #check this
                         } else {
                           Model__use_attenuation(private$ptr,private$attenuate_parameters, private$model_type())
                           Model__mcmc_set_lambda(private$ptr,self$mcmc_options$lambda, private$model_type())
                           Model__mcmc_set_max_steps(private$ptr,self$mcmc_options$maxsteps, private$model_type())
                           Model__mcmc_set_refresh(private$ptr,self$mcmc_options$refresh, private$model_type())
                           Model__mcmc_sample(private$ptr,self$mcmc_options$warmup,self$mcmc_options$samps,self$mcmc_options$adapt, private$model_type())
                           dsamps <- Model__u(private$ptr,TRUE, private$model_type())
                           dsamps <- Matrix::Matrix(Model__L(private$ptr, private$model_type()) %*% dsamps)
                         }
                         return(invisible(dsamps))
                       },
                       #' @description 
                       #' The gradient of the log-likelihood with respect to either the random effects or
                       #' the model parameters. The random effects are on the N(0,I) scale, i.e. scaled by the
                       #' Cholesky decomposition of the matrix D. To obtain the random effects from the last 
                       #' model fit, see member function `$u`
                       #' @param y Vector of outcome data
                       #' @param u Vector of random effects scaled by the Cholesky decomposition of D
                       #' @param beta Logical. Whether the log gradient for the random effects (FALSE) or for the linear predictor parameters (TRUE)
                       #' @return A vector of the gradient
                       gradient = function(y,u,beta=FALSE){
                         private$verify_data(y)
                         private$set_y(y)
                         grad <- Model__log_gradient(private$ptr,u,beta, private$model_type())
                         return(grad)
                       },
                       #' @description 
                       #' The partial derivatives of the covariance matrix Sigma with respect to the covariance
                       #' parameters. The function returns a list in order: Sigma, first order derivatives, second 
                       #' order derivatives. The second order derivatives are ordered as the lower-triangular matrix
                       #' in column major order. Letting 'd(i)' mean the first-order partial derivative with respect 
                       #' to parameter i, and d2(i,j) mean the second order derivative with respect to parameters i 
                       #' and j, then if there were three covariance parameters the order of the output would be:
                       #' (sigma, d(1), d(2), d(3), d2(1,1), d2(1,2), d2(1,3), d2(2,2), d2(2,3), d2(3,3)).
                       #' @return A list of matrices, see description for contents of the list.
                       partial_sigma = function(){
                         if(is.null(private$ptr))private$update_ptr()
                         out <- Model__cov_deriv(private$ptr, private$model_type())
                         return(out)
                       },
                       #' @description 
                       #' Returns the sample of random effects from the last model fit
                       #' @param scaled Logical indicating whether to return samples on the N(0,I) scale (`scaled=FALSE`) or
                       #' N(0,D) scale (`scaled=TRUE`)
                       #' @return A matrix of random effect samples
                       u = function(scaled = TRUE){
                         if(is.null(private$ptr))stop("Model not set")
                         return(Model__u(private$ptr,scaled, private$model_type()))
                       },
                       #' @description 
                       #' The log likelihood for the GLMM. The random effects can be left 
                       #' unspecified. If no random effects are provided, and there was a previous model fit with the same data `y`
                       #' then the random effects will be taken from that model. If there was no
                       #' previous model fit then the random effects are assumed to be all zero.
                       #' @param y A vector of outcome data
                       #' @param u An optional matrix of random effect samples. This can be a single column.
                       #' @return The log-likelihood of the model parameters
                       log_likelihood = function(y,u){
                         private$verify_data(y)
                         private$set_y(y)
                         if(!missing(u))Model__update_u(private$ptr,u, private$model_type())
                         return(Model__log_likelihood(private$ptr, private$model_type()))
                       },
                       #' @field mcmc_options There are five options for MCMC sampling that are specified in this list:
                       #' * `warmup` The number of warmup iterations. Note that if using the internal HMC
                       #' sampler, this only applies to the first iteration of the MCML algorithm, as the
                       #' values from the previous iteration are carried over.
                       #' * `samps` The number of MCMC samples drawn in the MCML algorithm. For
                       #' smaller tolerance values larger numbers of samples are required. For the internal
                       #' HMC sampler, larger numbers of samples are generally required than if using Stan since
                       #' the samples generally exhibit higher autocorrealtion, especially for more complex
                       #' covariance structures.
                       #' * `lambda` (Only relevant for the internal HMC sampler) Value of the trajectory length of the leapfrog integrator in Hamiltonian Monte Carlo
                       #'  (equal to number of steps times the step length). Larger values result in lower correlation in samples, but
                       #'  require larger numbers of steps and so is slower. Smaller numbers are likely required for non-linear GLMMs.
                       #'  * `refresh` How frequently to print to console MCMC progress if displaying verbose output.
                       #'  * `maxsteps` (Only relevant for the internal HMC sampler) Integer. The maximum number of steps of the leapfrom integrator
                       mcmc_options = list(warmup = 500,
                                           samps = 250,
                                           lambda = 1,
                                           refresh = 500,
                                           maxsteps = 100,
                                           target_accept = 0.95,
                                           adapt = 50),
                       #' @description
                       #' Prints the internal instructions used to calculate the linear predictor and/or
                       #' the log likelihood. Internally the class uses a reverse polish notation to store and 
                       #' calculate different functions, including user-specified non-linear mean functions. This 
                       #' function will print all the steps. Mainly used for debugging and determining how the 
                       #' class has interpreted non-linear model specifications. 
                       #' @param linpred Logical. Whether to print the linear predictor instructions.
                       #' @param loglik Logical. Whether to print the log-likelihood instructions.
                       #' @return None. Called for effects.
                       calculator_instructions = function(linpred = TRUE, loglik = FALSE){
                         Model__print_instructions(private$ptr,linpred,loglik,private$model_type())
                       },
                       #' @description
                       #' Calculates the marginal effect of variable x. There are several options for 
                       #' marginal effect and several types of conditioning or averaging. The type of marginal
                       #' effect can be the derivative of the mean with respect to x (`dydx`), the expected 
                       #' difference E(y|x=a)-E(y|x=b) (`diff`), or the expected log ratio log(E(y|x=a)/E(y|x=b)) (`ratio`).
                       #' Other fixed effect variables can be set at specific values (`at`), set at their mean values
                       #' (`atmeans`), or averaged over (`average`). Averaging over a fixed effects variable here means
                       #' using all observed values of the variable in the relevant calculation. 
                       #' The random effects can similarly be set at their 
                       #' estimated value (`re="estimated"`), set to zero (`re="zero"`), set to a specific value 
                       #' (`re="at"`), or averaged over (`re="average"`). Estimates of the expected values over the random
                       #' effects are generated using MCMC samples. MCMC samples are generated either through 
                       #' MCML model fitting or using `mcmc_sample`. In the absence of samples `average` and `estimated` 
                       #' will produce the same result. The standard errors are calculated using the delta method with one 
                       #' of several options for the variance matrix of the fixed effect parameters.
                       #' Several of the arguments require the names
                       #' of the variables as given to the model object. Most variables are as specified in the formula,
                       #' factor variables are specified as the name of the `variable_value`, e.g. `t_1`. To see the names
                       #' of the stored parameters and data variables see the member function `names()`.
                       #' @param x String. Name of the variable to calculate the marginal effect for.
                       #' @param type String. Either `dydx` for derivative, `diff` for difference, or `ratio` for log ratio. See description.
                       #' @param re String. Either `estimated` to condition on estimated values, `zero` to set to zero, `at` to
                       #' provide specific values, or `average` to average over the random effects.
                       #' @param se String. Type of standard error to use, either `GLS` for the GLS standard errors, `KR` for 
                       #' Kenward-Roger estimated standard errors, `KR2` for the improved Kenward-Roger correction (see `small_sample_correction()`),
                       #'  or `robust` to use a robust sandwich estimator.
                       #' @param at Optional. A vector of strings naming the fixed effects for which a specified value is given.
                       #' @param atmeans Optional. A vector of strings naming the fixed effects that will be set at their mean value.
                       #' @param average Optional. A vector of strings naming the fixed effects which will be averaged over.
                       #' @param xvals. Optional. A vector specifying the values of `a` and `b` for `diff` and `ratio`. The default is (1,0).
                       #' @param atvals Optional. A vector specifying the values of fixed effects specified in `at` (in the same order).
                       #' @param revals Optional. If `re="at"` then this argument provides a vector of values for the random effects.
                       #' @return A named vector with elements `margin` specifying the point estimate and `se` giving the standard error.
                       marginal = function(x,type,re,se,at = c(),atmeans = c(),average=c(),xvals=c(1,0),atvals=c(),revals=c()){
                         margin_types <- c("dydx","diff","ratio")
                         re_types <- c("estimated","at","zero","average")
                         se_types <- c("GLS","KR","Robust","BW","KR2","Sat")
                         if(!type%in%margin_types)stop("type not recognised")
                         if(!re%in%re_types)stop("re not recognised")
                         if(!se%in%se_types)stop("se not recognised")
                         result <- Model__marginal(xp = private$ptr,
                                                   x = x,
                                                   margin = which(margin_types==type)-1,
                                                   re = which(re_types==re)-1, 
                                                   se = which(se_types==se)-1,
                                                   at = at,
                                                   atmeans = atmeans,
                                                   average = average,
                                                   xvals_first = xvals[1],
                                                   xvals_second =xvals[2],
                                                   atvals = atvals,
                                                   revals = revals,
                                                   type = private$model_type())
                         return(c("margin"=unname(result[1]),"SE"=unname(result[2])))
                       },
                       #' @description
                       #' Updates the outcome data y
                       #' 
                       #' Some functions require outcome data, which is by default set to all zero if no model fitting function 
                       #' has been run. This function can update the interval y data.
                       #' @param y Vector of outcome data
                       #' @return None. Called for effects
                       update_y = function(y){
                         private$verify_data(y)
                         private$set_y(y)
                       },
                       #' @description
                       #' Controls the information printed to the console for other functions. 
                       #' @param trace Integer, either 0 = no information, 1 = some information, 2 = all information
                       #' @return None. Called for effects.
                       set_trace = function(trace){
                         if(!trace%in%c(0,1,2))stop("trace must be 0, 1, or 2")
                         private$trace <- trace
                         Model__set_trace(private$ptr,trace, private$model_type())
                       }
                     ),
                     private = list(
                       W = NULL,
                       Xb = NULL,
                       trace = 0,
                       useSparse = TRUE,
                       logit = function(x){
                         exp(x)/(1+exp(x))
                       },
                       genW = function(){
                         Model__update_W(private$ptr, private$model_type())
                         private$W <- Model__get_W(private$ptr, private$model_type())
                       },
                       attenuate_parameters = FALSE,
                       ptr = NULL,
                       set_y = function(y){
                         if(is.null(private$ptr))private$update_ptr()
                         Model__set_y(private$ptr,y, private$model_type())
                       },
                       model_type = function(){
                         type <- self$covariance$.__enclos_env__$private$type
                         return(type)
                       },
                       update_ptr = function(force = FALSE){
                         if(is.null(private$ptr) | force){
                           if(!self$family[[1]]%in%c("poisson","binomial","gaussian","bernoulli","Gamma","beta"))stop("family must be one of Poisson, Binomial, Gaussian, Gamma, Beta")
                           if(gsub(" ","",self$mean$formula) != gsub(" ","",self$covariance$formula)){
                             form <- paste0(self$mean$formula,"+",self$covariance$formula)
                           } else {
                             form <- gsub(" ","",self$mean$formula)
                           }
                           if(grepl("nngp",form)){
                             form <- gsub("nngp_","",form)
                           } else if(grepl("hsgp",form)){
                             form <- gsub("hsgp_","",form)
                           }
                           type <- private$model_type()
                           data <- self$covariance$data
                           if(any(!colnames(self$mean$data)%in%colnames(data))){
                             cnames <- which(!colnames(self$mean$data)%in%colnames(data))
                             data <- cbind(data,self$mean$data[,cnames])
                           }
                           if(self$family[[1]]=="bernoulli" & any(self$trials>1))self$family[[1]] <- "binomial"
                           if(type == 0){
                             private$ptr <- Model__new_w_pars(form,as.matrix(data),colnames(data),
                                                              tolower(self$family[[1]]),
                                                              self$family[[2]],
                                                              self$mean$parameters,
                                                              self$covariance$parameters)
                           } else if(type==1){
                             nngp <- self$covariance$nngp()
                             private$ptr <- Model_nngp__new_w_pars(form,as.matrix(data),
                                                                   colnames(data),
                                                                    tolower(self$family[[1]]),
                                                                    self$family[[2]],
                                                                    self$mean$parameters,
                                                                    self$covariance$parameters,
                                                                    nngp[2])
                           } else if(type==2){
                             private$ptr <- Model_hsgp__new_w_pars(form,as.matrix(data),
                                                                   colnames(data),
                                                                   tolower(self$family[[1]]),
                                                                   self$family[[2]],
                                                                   self$mean$parameters,
                                                                   self$covariance$parameters)
                           }
                           
                           Model__set_offset(private$ptr,self$mean$offset,type)
                           Model__set_weights(private$ptr,self$weights,type)
                           Model__set_var_par(private$ptr,self$var_par,type)
                           if(self$family[[1]] == "binomial")Model__set_trials(private$ptr,self$trials,type)
                           Model__update_beta(private$ptr,self$mean$parameters,type)
                           Model__update_theta(private$ptr,self$covariance$parameters,type)
                           Model__update_u(private$ptr,matrix(rnorm(Model__Q(private$ptr,type)),ncol=1),type) # initialise random effects to random
                           Model__mcmc_set_lambda(private$ptr,self$mcmc_options$lambda,type)
                           Model__mcmc_set_max_steps(private$ptr,self$mcmc_options$maxsteps,type)
                           Model__mcmc_set_refresh(private$ptr,self$mcmc_options$refresh,type)
                           Model__mcmc_set_target_accept(private$ptr,self$mcmc_options$target_accept,type)
                           if(!private$useSparse & type == 1) Model__make_dense(private$ptr,type)
                           # set covariance pointer
                           self$covariance$.__enclos_env__$private$model_ptr <- private$ptr
                           self$covariance$.__enclos_env__$private$ptr <- NULL
                         }
                       },
                       verify_data = function(y){
                         if(any(is.na(y)))stop("NAs in y")
                         if(self$family[[1]]=="binomial"){
                           if(!(all(y>=0 & y%%1 == 0)))stop("y must be integer >= 0")
                           if(any(y > self$trials))stop("Number of successes > number of trials")
                         } else if(self$family[[1]]=="poisson"){
                           if(any(y <0) || any(y%%1 != 0))stop("y must be integer >= 0")
                         } else if(self$family[[1]]=="beta"){
                           if(any(y<0 || y>1))stop("y must be between 0 and 1")
                         } else if(self$family[[1]]=="Gamma") {
                           if(any(y<=0))stop("y must be positive")
                         } else if(self$family[[1]]=="gaussian" & self$family[[2]]=="log"){
                           if(any(y<=0))stop("y must be positive")
                         } else if(self$family[[1]]=="bernoulli"){
                           if(any(y > self$trials))stop("Number of successes > number of trials")
                           if(any(y > 1))self$family[[1]] <- "binomial"
                           if(!(all(y>=0 & y%%1 == 0)))stop("y must be 0 or 1")
                         }
                       },
                       model_data = function(newdata){
                         cnames <- colnames(self$covariance$data)
                         if(any(!colnames(self$mean$data)%in%cnames)){
                           cnames <- c(cnames1, which(!colnames(self$mean$data)%in%cnames))
                         }
                         if(!isTRUE(all.equal(cnames,colnames(newdata)))){
                           newdat <- newdata[,cnames[cnames%in%colnames(newdata)]]
                           newcnames <- cnames[!cnames%in%colnames(newdata)]
                           for(i in newcnames){
                             if(grepl("factor",i)){
                               id1 <- gregexpr("\\[",i)
                               id2 <- gregexpr("\\]",i)
                               var <- substr(i,id1[[1]][1]+1,id2[[1]][1]-1)
                               if(!var%in%colnames(newdata))stop(paste0("factor ",var," not in data"))
                               val <- substr(i,id2[[1]][1]+1,nchar(i))
                               newcol <- I(newdata[,var]==val)*1
                               newdat <- cbind(newdat,newcol)
                               colnames(newdat)[ncol(newdat)] <- i
                             } else {
                               stop(paste0("Variable ",i," not in data"))
                             }
                           }
                         } else {
                           newdat <- newdata
                         }
                         newdat <- newdat[,cnames]
                         return(newdat)
                       }
                     ))


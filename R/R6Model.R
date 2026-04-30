#' A GLMM Model
#'
#' A generalised linear mixed model 
#' @details
#' See \link[glmmrBase]{glmmrBase-package} for a more in-depth guide.
#'
#' The generalised linear mixed model is:
#'
#' \deqn{Y \sim F(\mu,\sigma)}
#' \deqn{\mu = h^-1(X\beta + Zu)}
#' \deqn{u \sim MVN(0,D)}
#'
#' where F is a distribution with scale parameter \deqn{\sigma}, h is a link function, X are the fixed effects with parameters \deqn{\beta}, Z is the random effect design matrix with multivariate Gaussian distributed effects u. 
#' The class provides access to all of the elements of the model above and associated calculations and functions including model fitting, power analysis,
#' and various relevant matrices, including information matrices and related corrections. The object is an R6 class and so can serve as a parent class for extended functionality.
#'
#' The currently supported families (links) are Gaussian (identity, log), Binomial (logit, log, probit, identity), Poisson (log, identity), Gamma (logit, identity, inverse), and Beta (logit).
#' 
#' This class provides model fitting functionality with a variety of stochastic maximum likelihood algorithms with and without restricted maximum likelihood corrections. A fast Laplace approximation is also included. 
#' Small sample corrections are also provided including Kenward-Roger and Satterthwaite corrections. 
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
#' detailed guide on model specification. A detailed vingette for this package is also available online<doi:10.48550/arXiv.2303.12657>.
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
                         return(invisible(self))
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
                       #' @return Fitted values as either a vector or matrix depending on the number of samples
                       fitted = function(type="link", X, u, sample= FALSE, sample_n = 100){
                         if(missing(X)){
                           if(!sample){
                             Xb <- self$mean$linear_predictor()
                           } else {
                             Xb <- matrix(NA,nrow=self$n(),ncol=sample_n)
                             b_curr <- Model__get_beta(private$ptr,private$model_type())
                             M <- self$information_matrix(average = FALSE)
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
                       #' Generates the residuals for the model
                       #' 
                       #' Generates one of several types of residual for the model. If conditional = TRUE then 
                       #' the residuals include the random effects, otherwise only the fixed effects are included. For type,
                       #' there are raw, pearson, and standardized residuals. For conditional residuals a matrix is returned 
                       #' with each column corresponding to a sample of the random effects.
                       #' @param type Either "standardized", "raw" or "pearson"
                       #' @param conditional Logical indicating whether to condition on the random effects (TRUE) or not (FALSE)
                       #' @return A matrix with either one column is conditional is false, or with number of columns corresponding 
                       #' to the number of MCMC samples.
                       residuals = function(type = "standardized",
                                            conditional = TRUE){
                         if(!private$y_has_been_updated)stop("No data y has been provided")
                         if(!type%in%c("standardized","raw","pearson"))stop("type must be one of standardized, raw, or pearson")
                         rtype_int <- match(type, c("raw","pearson","standardized")) - 1
                         R <- Model__residuals(private$ptr,rtype_int,conditional,private$model_type())
                         return(R)
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
                                          mesh_A = NULL,
                                          m=0
                       ){
                         preddata <- private$model_data(newdata)
                         if(private$model_type() == 4){
                           out <- Model_spde__predict(private$ptr,as.matrix(preddata),offset,mesh_A,m)
                         } else {
                           out <- Model__predict(private$ptr,as.matrix(preddata),offset,m,private$model_type())
                         }
                         
                         return(out)
                       },
                       #' @description
                       #' Create a new Model object. Typically, a model is generated from a formula and data. However, it can also be 
                       #' generated from a previous model fit. 
                       #' @param formula A model formula containing fixed and random effect terms. The formula can be one way (e.g. `~ x + (1|gr(cl))`) or
                       #' two-way (e.g. `y ~ x + (1|gr(cl))`). One way formulae will generate a valid model enabling data simulation, matrix calculation, 
                       #' and power, etc. Outcome data can be passed directly to model fitting functions, or updated later using member function `update_y()`.
                       #' For binomial models, either the syntax `cbind(y, n-y)` can be used for outcomes, or just `y` and the number of trials passed to the argument
                       #' `trials` described below.
                       #' @param covariance (Optional) Either a \link[glmmrBase]{Covariance} object, an equivalent list of arguments
                       #' that can be passed to `Covariance` to create a new object, or a vector of parameter values. At a minimum the list must specify a formula.
                       #' If parameters are not included then they are initialised to 0.5. 
                       #' @param mean (Optional) Either a \link[glmmrBase]{MeanFunction} object, an equivalent list of arguments
                       #' that can be passed to `MeanFunction` to create a new object, or a vector of parameter values. At a minimum the list must specify a formula.
                       #' If parameters are not included then they are initialised to 0.
                       #' @param data A data frame with the data required for the mean function and covariance objects. This argument
                       #' can be ignored if data are provided to the covariance or mean arguments either via `Covariance` and `MeanFunction`
                       #' object, or as a member of the list of arguments to both `covariance` and `mean`.
                       #' @param family A family object expressing the distribution and link function of the model, see \link[stats]{family}. Currently accepts \link[stats]{binomial},
                       #' \link[stats]{gaussian}, \link[stats]{Gamma}, \link[stats]{poisson}, \link[glmmrBase]{Beta}, and \link[glmmrBase]{Quantile}.
                       #' @param var_par (Optional) Scale parameter required for some distributions, including Gaussian. Default is NULL.
                       #' @param offset (Optional) A vector of offset values. Optional - could be provided to the argument to mean instead.
                       #' @param trials (Optional) For binomial family models, the number of trials for each observation. If it is not set, then it will
                       #' default to 1 (a bernoulli model).
                       #' @param weights (Optional) A vector of weights. 
                       #' @param model_fit (optional) A `mcml` model fit resulting from a call to `MCML` or `LA`
                       #' @return A new Model class object
                       #' @seealso \link[glmmrBase]{nelder}, \link[glmmrBase]{MeanFunction}, \link[glmmrBase]{Covariance}
                       #' @examples
                       #' \dontshow{
                       #' setParallel(FALSE) 
                       #' }
                       #' # For more examples, see the examples for MCML.
                       #' 
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
                       #' # We can also include the outcome data in the model initialisation. 
                       #' # For example, simulating data and creating a new object:
                       #' df$y <- mod$sim_data()
                       #'
                       #' mod <- Model$new(
                       #'   formula = y ~ factor(t) + int - 1 + (1|gr(cl)) + (1|gr(cl,t)),
                       #'   data = df,
                       #'   family = stats::gaussian()
                       #' )
                       #'
                       #' # Here we will specify a cohort study
                       #' df <- nelder(~ind(20) * t(6))
                       #' df$int <- 0
                       #' df[df$t > 3, 'int'] <- 1
                       #' 
                       #' des <- Model$new(
                       #'   formula = ~ int + (1|gr(ind)),
                       #'   data = df,
                       #'   family = stats::poisson()
                       #' )
                       #'   
                       #' # or with parameter values specified
                       #'   
                       #' des <- Model$new(
                       #'   formula = ~ int + (1|gr(ind)),
                       #'   covariance = c(0.05),
                       #'   mean = c(1,0.5),
                       #'   data = df,
                       #'   family = stats::poisson()
                       #' )
                       #'
                       #' #an example of a spatial grid with two time points
                       #'
                       #' df <- nelder(~ (x(10)*y(10))*t(2))
                       #' spt_design <- Model$new(formula = ~ 1 + (1|ar0(t)*fexp(x,y)),
                       #'                         data = df,
                       #'                         family = stats::gaussian())
                       initialize = function(formula,
                                             covariance,
                                             mean,
                                             data = NULL,
                                             family = NULL,
                                             var_par = NULL,
                                             offset = NULL,
                                             weights = NULL,
                                             trials = NULL,
                                             model_fit = NULL,
                                             mesh = NULL){
                         if(!is.null(model_fit)){
                           if(!missing(formula) | !missing(covariance) | !missing(mean)) message("Previous model fit has been provided, all other arguments are ignored")
                           self$family <- do.call(model_fit$family, list(link = model_fit$link))
                           if(model_fit$var_par_family){
                             self$var_par <- model_fit$coefficients$est[model_fit$P + model_fit$Q + 1]
                           } else {
                             self$var_par <- 1
                           }
                           self$covariance <- Covariance$new(
                             formula = model_fit$cov_form,
                             parameters = model_fit$coefficients$est[(model_fit$P + 1):(model_fit$P + model_fit$Q)]
                           )
                           self$mean <- MeanFunction$new(
                             formula = model_fit$mean_form,
                             data = model_fit$model_data$data,
                             parameters = model_fit$coefficients$est[1:(model_fit$P)]
                           )
                           self$mean$offset <- model_fit$model_data$offset
                           self$weights <- model_fit$model_data$weights
                           if(self$family[[1]]=="binomial") self$trials <- model_fit$model_data$trials
                         } else {
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
                             form_1 <- private$check_y_formula(formula,data,self$family)
                             processed_data_all <- private$process_data(form_1,data,TRUE,TRUE)
                             self$formula <- Reduce(paste,as.character(processed_data_all$form))
                             if(is.null(data)){
                               stop("Data must be specified with a formula")
                             } else {
                               processed_data_cov <- private$process_data(form_1,data,TRUE,FALSE)
                               if(missing(covariance) || (!all(is(covariance,"numeric")) & !"parameters"%in%names(covariance))){
                                 self$covariance <- Covariance$new(
                                   formula = processed_data_cov$formula
                                 )
                               } else {
                                 if("parameters"%in%names(covariance)){
                                   self$covariance <- Covariance$new(
                                     formula = processed_data_cov$formula,
                                     parameters = covariance$parameters
                                   )
                                 } else if(all(is(covariance,"numeric"))){
                                   self$covariance <- Covariance$new(
                                     formula = processed_data_cov$formula,
                                     parameters = covariance
                                   )
                                 } else {
                                   stop("Cannot interpret covariance argument")
                                 }
                               }
                               if(missing(mean) || (!"parameters"%in%names(mean) & !all(is(mean,"numeric")))){
                                 processed_data <- private$process_data(form_1,data,FALSE,TRUE)
                                 self$mean <- MeanFunction$new(
                                   formula = processed_data$formula,
                                   data = processed_data$data
                                 )
                               } else {
                                 if("parameters"%in%names(mean)){
                                   processed_data <- private$process_data(form_1,data,FALSE,TRUE)
                                   self$mean <- MeanFunction$new(
                                     formula = processed_data$formula,
                                     data = processed_data$data,
                                     parameters = mean$parameters
                                   )
                                 } else if(all(is(mean,"numeric"))){
                                   processed_data <- private$process_data(form_1,data,FALSE,TRUE)
                                   self$mean <- MeanFunction$new(
                                     formula = processed_data$formula,
                                     data = processed_data$data,
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
                                     self$covariance$data <- private$process_data(covariance$formula,data,TRUE,FALSE)$data
                                   }
                                 }
                               } else {
                                 stop("covariance should be Covariance class or parameter vector")
                               }
                             } else {
                               stop("covariance should be Covariance class or parameter vector")
                             } 
                             if(is(mean,"R6")){
                               if(is(mean,"MeanFunction")){
                                 self$mean <- mean
                               } else {
                                 stop("mean should be MeanFunction class or parameter vector")
                               }
                             } else {
                               stop("mean should be MeanFunction class or parameter vector")
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
                             if((is.null(trials) || all(trials == 1)) & is.null(self$trials)){
                               self$trials <- rep(1,nrow(self$mean$data))
                               self$family[[1]] <- "bernoulli"
                             } else {
                               if(is.null(self$trials))self$trials <- trials
                             }
                           }
                         }
                         if(!is.null(mesh)) private$mesh <- mesh
                         private$session_id <- Sys.getpid()
                         
                         if(!is.null(model_fit)){
                           self$covariance$data = model_fit$model_data$data
                         } else {
                           self$covariance$data = private$process_data(form_1,data,TRUE,FALSE)$data
                         }
                         private$update_ptr()
                         
                         if(private$y_in_formula){
                           if(! private$y_name %in% colnames(data)) stop(paste0(private$y_name," not in data"))
                           self$update_y(data[,private$y_name])
                         }
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
                         cat("\n")
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
                         return(invisible(self))
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
                       #'   formula = ~ factor(t) + int - 1 + (1|gr(cl)*ar0(t)),
                       #'   covariance = c(0.05,0.8),
                       #'   mean = c(rep(0,5),0.6),
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
                         if(f[1]%in%c("quantile","quantile_scaled")){
                           message("Quantile based methods are currently EXPERIMENTAL")
                           message("Simulation from quantile family produces random draws from an asymmetric Laplace distribution")
                           if(f[2]=="logit"){
                             mu <- exp(mu)/(1+exp(mu))
                           } else if(f[2]=="log"){
                             mu <- exp(mu)
                           } else if(f[2] == "inverse"){
                             mu <- 1/mu
                           } else if(f[2] == "probit"){
                             mu <- pnorm(mu)
                           }
                           rand_u <- runif(length(mu))
                           y <- rep(NA,length(mu))
                           for(i in 1:length(y)){
                             if(rand_u[i] <= self$family$q){
                               y[i] <- (self$var_par/(1-self$family$q))*log(rand_u[i]/self$family$q) + mu[i]
                             } else {
                               y[i] <- (self$var_par/self$family$q)*log((1-rand_u[i])/(1-self$family$q)) + mu[i]
                             }
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
                       #'   formula = ~ factor(t) + int - 1 + (1|gr(cl)*ar0(t)),
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
                           self$covariance$update_parameters(cov.pars)
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
                         return(invisible(self))
                       },
                       #' @description
                       #' Generates the information matrix of the mixed model GLS estimator (X'S^-1X). The inverse of this matrix is an 
                       #' estimator for the variance-covariance matrix of the fixed effect parameters. For various small sample corrections
                       #' see `small_sample_correction()` and `box()`. For models with non-linear functions of fixed effect parameters,
                       #' a correction to the Hessian matrix is required, which is automatically calculated or optionally returned or disabled. 
                       #' @param include.re logical indicating whether to return the information matrix including the random effects components (TRUE), 
                       #' or the mixed model information matrix for beta only (FALSE).
                       #' @param theta Logical. If TRUE the function will return the variance-coviariance matrix for the covariance parameters and ignore the first argument. Otherwise, the fixed effect
                       #' parameter information matrix is returned.
                       #' @param oim Logical. If TRUE, returns the observed information matrix for both beta and theta, disregarding other arguments to the function.
                       #' @return A matrix
                       information_matrix = function(include.re = FALSE, theta = FALSE, oim = FALSE, average = TRUE){
                         if(oim & !private$y_has_been_updated) stop("No y data has been added")
                         private$update_ptr()
                         nonlin <- Model__any_nonlinear(private$ptr,private$model_type())
                         if(oim){
                           M <- Model__observed_information_matrix(private$ptr,private$model_type())
                         } else {
                           if(theta){
                             M <- Model__infomat_theta(private$ptr,private$model_type())
                           } else {
                             if(include.re & private$model_type() != 1){
                               M <- Model__obs_information_matrix(private$ptr,private$model_type())
                             } else {
                               if(private$model_type()%in%c(0,2,4)){
                                 if(average){
                                   M <- Model__ave_information_matrix(private$ptr,private$model_type())
                                 } else {
                                   M <- Model__information_matrix(private$ptr,private$model_type())
                                 }
                               } else {
                                 M <- Model__information_matrix_crude(private$ptr,private$model_type())
                               }
                             }
                           }
                         }
                         return(M)
                       },
                       #' @description 
                       #' Returns the robust sandwich variance-covariance matrix for the fixed effect parameters
                       #' @return A PxP matrix
                       sandwich = function(){
                         private$update_ptr()
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
                       #' @param oim Logical. If TRUE use the observed information matrix, otherwise use the expected information matrix
                       #' @return A PxP matrix
                       small_sample_correction = function(type, oim = FALSE){
                         if(oim & !private$y_has_been_updated) stop("No y data has been added")
                         private$update_ptr()
                         if(!type %in% c("KR","KR2","sat"))stop("type must be either KR, KR2, or sat")
                         ss_type <- ifelse(type == "KR",1,ifelse(type == "KR2",4,5))
                         return(Model__small_sample_correction(private$ptr,ss_type,oim,private$model_type()))
                       },
                       #' @description 
                       #' Returns the inferential statistics (F-stat, p-value) for a modified Box correction <doi:10.1002/sim.4072> for
                       #' Gaussian-identity models.
                       #' @param y Optional. If provided, will update the vector of outcome data. Otherwise it will use the data from 
                       #' the previous model fit.
                       #' @return A data frame.
                       box = function(y){
                         if(!(self$family[[1]]=="gaussian"&self$family[[2]]=="identity"))stop("Box only available for linear models")
                         private$update_ptr()
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
                       #'   formula = ~ factor(t) + int - 1 + (1|gr(cl)) + (1|gr(cl,t)),
                       #'   covariance = c(0.05,0.1),
                       #'   mean = c(rep(0,5),0.6),
                       #'   data = df,
                       #'   family = stats::gaussian(),
                       #'   var_par = 1
                       #' )
                       #' des$power() #power of 0.90 for the int parameter
                       power = function(alpha=0.05,two.sided=TRUE,alternative = "pos"){
                         M <- self$information_matrix(average = FALSE)
                         v0 <- solve(M)
                         v0 <- as.vector(sqrt(diag(v0)))
                         if(two.sided){
                           pwr <- pnorm(abs(self$mean$parameters/v0) - qnorm(1-alpha/2))
                         } else {
                           if(alternative == "pos"){
                             pwr <- pnorm(self$mean$parameters/v0 - qnorm(1-alpha))
                           } else {
                             pwr <- pnorm(-self$mean$parameters/v0 - qnorm(1-alpha))
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
                         private$update_ptr()
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
                         private$update_ptr()
                         return(Model__Sigma(private$ptr,inverse,private$model_type()))
                       },
                       #'@description
                       #'MCML model fitting with the fastest options
                       #'
                       #' Uses double Newton-Raphson method (with or without REML for Gaussian models). Note that no random effect
                       #' samples are drawn for Gaussian models. It is recommended to use the log 
                       #' version of the covariance functions with this method as the Newton-Raphson steps can lead to negative values otherwise.
                       #' @param niter Integer. Number of samples for the random effects, ignored for Gaussian models, see examples.
                       #' @param max_iter Integer. Maximum number of iterations.
                       #' @param se Either "average" or "point". "Average" (default) estimates the information matrix for beta averaging over MC samples, if `niter` is one (triggering a
                       #' Laplace Approximation) then a final sample is drawn for this estimator. "Point" evaluates the information matrix for beta at the posterior mean of the 
                       #' random effects.
                       #' @param tol Scalar. The tolerance for the convergence criterion. For GLMMs this is the tolerance for 
                       #' the Bayes Factor convergence criterion, for Gaussian linear models the tolerance is the difference 
                       #' in the log-likelihood between successive iterations.
                       #' @param hist Integer. The length of the running mean for the convergence criterion for non-Gaussian models.
                       #' @param k0 Integer. The expected number of iterations until convergence.
                       #' @param reml Bool. For Gaussian models, whether to use REML or not.
                       #' @param start_glm Bool. Start beta from the glm fitted values with random effects set to zero.
                       #' @return A `mcml` model fit object
                       #' @examples
                       #' # Simulated trial data example using REML
                       #' set.seed(123)
                       #'data(SimTrial,package = "glmmrBase")
                       #' fit1 <- Model$new(
                       #'   formula = y ~ int + factor(t) - 1 + (1|grlog(cl)*ar0log(t)),
                       #'   data = SimTrial,
                       #'   family = gaussian()
                       #' )$fit(reml = TRUE)
                       #' 
                       #' # Salamanders data example
                       #' data(Salamanders,package="glmmrBase")
                       #' model <- Model$new(
                       #'   mating~fpop:mpop-1+(1|grlog(mnum))+(1|grlog(fnum)),
                       #'   data = Salamanders,
                       #'   family = binomial()
                       #' )
                       #' 
                       #' fit2 <- model$fit()
                       #' 
                       #' # Example using simulated data
                       #' #create example data with six clusters, five time periods, and five people per cluster-period
                       #' df <- nelder(~(cl(20)*t(10)) > ind(5))
                       #' # parallel trial design intervention indicator
                       #' df$int <- 0
                       #' df[df$cl > 10, 'int'] <- 1 
                       #' # specify parameter values in the call for the data simulation below
                       #' des <- Model$new(
                       #'   formula= ~ factor(t) + int - 1 +(1|grlog(cl)*ar0log(t)),
                       #'   covariance = log(c(0.15,0.7)),
                       #'   mean = c(rep(0,10),0.2),
                       #'   data = df,
                       #'   family = binomial()
                       #' )
                       #' ysim <- des$sim_data() # simulate some data from the model
                       #' des$update_y(ysim)
                       #' set.seed(123)
                       #' fit2 <- des$fit() 
                       #' 
                       #' # use of Gaussian process approximations
                       #' # simulate some data - binomial observation on [-1,1] x [-1,1]
                       #' set.seed(123)
                       #' df <- data.frame(
                       #' x = runif(n, -1, 1),
                       #' y = runif(n, -1, 1))
                       #' df$z <- rnorm(n)
                       #' 
                       #' df$outcome <- Model$new(
                       #'   ~ z + (1|matern1log(x, y)),
                       #'   data = df,
                       #'    family = binomial(),
                       #'   mean = c(1, 0.1),
                       #'   covariance = c(log(2), log(0.3)),
                       #'   trials = rep(10, nrow(df))
                       #' )$sim_data()
                       #' 
                       #' # we can fit the SPDE approximation using a mesh built by fmesher
                       #' df_pred <- expand.grid(x= seq(-1,1,by=0.05), y = seq(-1,1,by=0.05))
                       #' df_pred$z <- 0
                       #' mesh_data <- mesh_helper(unique(df[,1:2]), df_pred[,1:2], c(0.15, 0.75), 0.075, c(0.1,0.3))
                       #' 
                       #' mod <- Model$new(
                       #'   outcome ~ z + (1|spde_matern1log(x, y)),
                       #'   data = df,
                       #'   family = binomial(),
                       #'   trials = rep(10, n),
                       #'   mesh = mesh_data[["data"]],
                       #'   covariance = log(c(0.5, 0.3))
                       #' )
                       #' set.seed(123)
                       #' fit1 <- mod$fit(niter = 50)
                       #' 
                       #' #generate predictions 
                       #' pred1 <- mod$predict(newdata = df_pred,mesh_A = mesh_data[["A_pred"]])
                       #' 
                       #' #' # Non-linear model fitting example using the example provided by nlmer in lme4
                       #' data(Orange, package = "lme4")
                       #' 
                       #' # the lme4 example:
                       #' startvec <- c(Asym = 200, xmid = 725, scal = 350)
                       #' (nm1 <- lme4::nlmer(circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree,
                       #'               Orange, start = startvec))
                       #' 
                       #' Orange <- as.data.frame(Orange)
                       #' Orange$Tree <- as.numeric(Orange$Tree)
                       #' 
                       #' # Here we can specify the model as a function. 
                       #' 
                       #' model <- Model$new(
                       #'   circumference ~ Asym/(1 + exp((xmid - (age))/scal)) - 1 + (Asym|gr(Tree)),
                       #'   data = Orange,
                       #'   family = gaussian(),
                       #'   mean = c(200,725,350),
                       #'   covariance = c(500),
                       #'   var_par = 50
                       #' )
                       #' set.seed(123)
                       #' nfit <- model$fit(niter = 100)
                       #' 
                       #' summary(nfit)
                       #' summary(nm1)
                       #' @md
                       fit = function(niter = 100, max_iter = 30, se = "average",
                                      tol = ifelse(self$family[[1]]=="gaussian"&self$family[[2]]=="identity",1e-6,10), 
                                      hist = 5, k0 = 10, reml = FALSE, start_glm = TRUE){
                         if((self$family[[1]]=="gaussian"&self$family[[2]]=="identity") | private$model_type() == 2)Model__use_reml(private$ptr,reml,private$model_type())
                         if(!se %in% c("average","point"))stop("se argument should be average or point")
                         if(private$model_type() == 2){
                           hsgp_vals <- self$covariance$hsgp()
                           hsgp_dim <- Model_hsgp__dim(private$ptr)
                           if(length(hsgp_vals[["m"]]) != hsgp_dim) hsgp_vals[["m"]] <- rep(hsgp_vals[["m"]][1],hsgp_dim)
                           Model_hsgp__set_approx_pars(private$ptr, hsgp_vals[["m"]], hsgp_vals[["L"]])
                         }
                         Model__fit(private$ptr, niter, max_iter, start_glm, tol, hist, k0, private$model_type())
                         self$update_parameters(mean.pars = Model__get_beta(private$ptr, private$model_type()),
                                                cov.pars = Model__get_theta(private$ptr, private$model_type()))
                         if(private$model_type() == 2) self$covariance$.__enclos_env__$private$genZ()
                         if(se == "average" & niter == 1) Model__posterior_u_sample(private$ptr, 200, FALSE, TRUE, FALSE, private$model_type()) 
                         u <- Model__u(private$ptr,TRUE,private$model_type())
                         M <- self$information_matrix(average = (se == "average")) 
                         M <- solve(M)
                         ncovpars <- Model__n_cov_pars(private$ptr,private$model_type())
                         if(self$family[[1]]=="gaussian")ncovpars <- ncovpars + 1
                         SE_theta <- Model__se_theta(private$ptr,private$model_type())
                         SE <- sqrt(diag(M))
                         repar_table <- self$covariance$parameter_table()
                         beta_names <- Model__beta_parameter_names(private$ptr,private$model_type())
                         theta_names <- repar_table$term
                         all_pars_new <- c(self$mean$parameters, self$covariance$parameters)
                         if(self$family[[1]]%in%c("Gamma","beta","quantile_scaled")){
                           mf_pars_names <- c(beta_names,theta_names,"sigma")
                           SE <- c(SE,rep(NA,length(theta_new)+1))
                         } else {
                           mf_pars_names <- c(beta_names,theta_names)
                           if(self$family[[1]]%in%c("gaussian")){
                             mf_pars_names <- c(mf_pars_names,"sigma")
                             self$update_parameters(var.par = Model__get_var_par(private$ptr,private$model_type()))
                             all_pars_new <- c(all_pars_new, self$var_par)
                           }
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
                         res$t <- res$est/res$SE
                         res$p <- 2*(1-stats::pnorm(abs(res$t)))
                         res$lower <- res$est - qnorm(1-0.05/2)*res$SE
                         res$upper <- res$est + qnorm(1-0.05/2)*res$SE
                         repar_table <- repar_table[!duplicated(repar_table$id),]
                         if(private$model_type()<2){
                           rownames(u) <- rep(repar_table$term,repar_table$count)
                         } else if(private$model_type()==3){
                           rownames(u) <- rep(paste0(repar_table$term[1],".t",1:self$covariance$.__enclos_env__$private$time),each=repar_table$count[1])
                         } else {
                           if(nrow(repar_table)==1) rownames(u) <-  rep(repar_table$term,nrow(u))
                         }
                         aic <- ifelse(private$model_type()==0 , Model__aic(private$ptr,private$model_type()),NA)
                         xb <- Model__xb(private$ptr,private$model_type())
                         zd <- self$covariance$Z %*% rowMeans(u)
                         wdiag <- Matrix::diag(self$w_matrix())
                         total_var <- var(Matrix::drop(xb)) + var(Matrix::drop(zd)) + mean(wdiag)
                         condR2 <- (var(Matrix::drop(xb)) + var(Matrix::drop(zd)))/total_var
                         margR2 <- var(Matrix::drop(xb))/total_var
                         out <- list(coefficients = res,
                                     converged = TRUE,
                                     method = "mcnr",
                                     m = dim(u)[2],
                                     tol = tol,
                                     sim_lik = FALSE,
                                     aic = aic,
                                     se="gls",
                                     vcov = M,
                                     Rsq = c(cond = condR2,marg=margR2),
                                     logl = self$log_likelihood(),
                                     logl_theta = NA,
                                     mean_form = self$mean$formula,
                                     cov_form = self$covariance$formula,
                                     family = self$family[[1]],
                                     link = self$family[[2]],
                                     re.samps = u,
                                     imp.weights = Model__get_importance_weights(private$ptr,private$model_type()),
                                     iter = NA,
                                     dof = dof,
                                     reml = FALSE,
                                     P = length(self$mean$parameters),
                                     Q = length(self$covariance$parameters),
                                     var_par_family =I(self$family[[1]]%in%c("gaussian","Gamma","beta","quantile_scaled")),
                                     model_data = list(
                                       y = Model__y(private$ptr,private$model_type()),
                                       data = private$model_data_frame(),
                                       trials = self$trials,
                                       offset = self$mean$offset,
                                       weights = self$weights
                                     ),
                                     fn_count = 0)
                         class(out) <- "mcml"
                         return(out)
                         
                       },
                       #' @description 
                       #' Set whether to use sparse matrix methods for model calculations and fitting.
                       #' By default the model does not use sparse matrix methods.
                       #' @param sparse Logical indicating whether to use sparse matrix methods
                       #' @return None, called for effects
                       sparse = function(sparse = TRUE){
                         if(!is.null(private$ptr)){
                           if(private$model_type() == 1){
                             message("Sparse has no effect with NNGP models")
                           } else {
                             if(sparse){
                               Model__make_sparse(private$ptr,private$model_type())
                             } else {
                               Model__make_dense(private$ptr,private$model_type())
                             }
                             self$covariance$sparse(sparse)
                           }
                           private$useSparse = sparse
                         }
                         return(invisible(self))
                       },
                       #' @description
                       #' Returns the importance weights for the random effect samples. 
                       #' @return A vector of the weights
                       importance_weights = function(){
                         return(Model__get_importance_weights(private$ptr, private$model_type()))
                       },
                       #' @description 
                       #' The gradient of the log-likelihood with respect to either the random effects or
                       #' the model parameters. The random effects are on the N(0,I) scale, i.e. scaled by the
                       #' Cholesky decomposition of the matrix D. To obtain the random effects from the last 
                       #' model fit, see member function `$u`
                       #' @param y (optional) Vector of outcome data, if not specified then data must have been set in another function.
                       #' @param u (optional) Vector of random effects scaled by the Cholesky decomposition of D
                       #' @param beta Logical. Whether the log gradient for the random effects (FALSE) or for the linear predictor parameters (TRUE)
                       #' @return A vector of the gradient
                       gradient = function(y,u,beta=FALSE){
                         if(!missing(y)){
                           private$verify_data(y)
                           private$set_y(y)
                         } else {
                           if(!private$y_has_been_updated) stop("No y data has been added")
                         }
                         if(missing(u)){
                           u_in <- Model__u(private$ptr, FALSE, private$model_type())
                         } else {
                           u_in <- u
                         }
                         grad <- matrix(NA,ifelse(beta,ncol(self$mean$X),nrow(u_in)),ncol(u_in))
                         for(i in 1:ncol(u)){
                           grad[,i] <- Model__log_gradient(private$ptr,u_in,beta, private$model_type())
                         }
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
                         private$update_ptr()
                         out <- Model__cov_deriv(private$ptr, private$model_type())
                         return(out)
                       },
                       #' @description 
                       #' Returns the sample of random effects from the last model fit, or updates the samples for the model.
                       #' @param scaled Logical indicating whether the samples are on the N(0,I) scale (`scaled=FALSE`) or
                       #' N(0,D) scale (`scaled=TRUE`)
                       #' @param u (optional) Matrix of random effect samples. If provided then the internal samples are replaced with these values. These samples should be N(0,I).
                       #' @return A matrix of random effect samples
                       u = function(scaled = TRUE, u){
                         if(is.null(private$ptr))stop("Model not set")
                         if(missing(u)){
                           return(Model__u(private$ptr,scaled, private$model_type()))
                         } else {
                           Model__update_u(private$ptr,u,FALSE,private$model_type()) 
                         }
                       },
                       #' @description
                        #' Generate importance weighted samples of the random effects. These are generated at the current values
                        #' of the model parameters. Importance weights can be returned by the `self$importance_weights()` function. 
                        #' If not assigned, the samples can be accessed using `self$u()`. 
                        #' @param niter Integer. Number of samples.
                        #' @param scaled Logical. Whether to return the random effects on the data (TRUE) or whitened (FALSE) scale.
                        #' @return Invisibly returns a matrix of random effect samples
                       sample_u = function(niter, scaled = TRUE){
                         Model__posterior_u_sample(private$ptr,scaled,niter,FALSE,TRUE,FALSE, private$model_type())
                         return(invisible(self$u(scaled)))
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
                         if(!missing(y)){
                           private$verify_data(y)
                           private$set_y(y)
                         } else {
                           if(!private$y_has_been_updated) stop("No y data has been added")
                         }
                         if(!missing(u)) Model__update_u(private$ptr,u, private$model_type())
                         return(Model__log_likelihood(private$ptr, private$model_type()))
                       },
                       #' @description
                       #' Prints the internal instructions and data used to calculate the linear predictor. 
                       #' Internally the class uses a reverse polish notation to store and 
                       #' calculate different functions, including user-specified non-linear mean functions. This 
                       #' function will print all the steps. Mainly used for debugging and determining how the 
                       #' class has interpreted non-linear model specifications. 
                       #' @return None. Called for effects.
                       calculator_instructions = function(){
                         Model__print_names(private$ptr,TRUE, TRUE, private$model_type())
                         Model__print_instructions(private$ptr,private$model_type())
                         return(invisible(self))
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
                       #' effects are generated using MC samples. In the absence of samples `average` and `estimated` 
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
                       #' @param xvals Optional. A vector specifying the values of `a` and `b` for `diff` and `ratio`. The default is (1,0).
                       #' @param atvals Optional. A vector specifying the values of fixed effects specified in `at` (in the same order).
                       #' @param revals Optional. If `re="at"` then this argument provides a vector of values for the random effects.
                       #' @param oim Logical. If TRUE use the observed information matrix, otherwise use the expected information matrix for standard error and related calculations.
                       #' @return A named vector with elements `margin` specifying the point estimate and `se` giving the standard error.
                       marginal = function(x,type,re,se,at = c(),atmeans = c(),average=c(),xvals=c(1,0),atvals=c(),revals=c(),oim = FALSE){
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
                                                   oim = oim,
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
                         return(invisible(self))
                       },
                       #' @description
                       #' Controls the information printed to the console for other functions. 
                       #' @param trace Integer, either 0 = no information, 1 = some information, 2 = all information
                       #' @return None. Called for effects.
                       set_trace = function(trace){
                         if(!trace%in%c(0,1,2))stop("trace must be 0, 1, or 2")
                         private$trace <- trace
                         Model__set_trace(private$ptr,trace, private$model_type())
                         return(invisible(self))
                       }
                     ),
                     private = list(
                       W = NULL,
                       Xb = NULL,
                       trace = 1,
                       useSparse = TRUE,
                       session_id = NULL,
                       logit = function(x){
                         exp(x)/(1+exp(x))
                       },
                       genW = function(){
                         Model__update_W(private$ptr, private$model_type())
                         private$W <- Model__get_W(private$ptr, private$model_type())
                       },
                       attenuate_parameters = FALSE,
                       ptr = NULL,
                       y_in_formula = FALSE,
                       y_name = NULL,
                       y_has_been_updated = FALSE,
                       mesh = NULL,
                       set_y = function(y){
                         private$update_ptr()
                         Model__set_y(private$ptr,y, private$model_type())
                         private$y_has_been_updated <- TRUE
                       },
                       model_type = function(){
                         type <- self$covariance$.__enclos_env__$private$type
                         return(type)
                       },
                       update_ptr = function(force = FALSE){
                         if(is.null(private$ptr) | force | private$session_id != Sys.getpid()){
                           if(!self$family[[1]]%in%c("poisson","binomial","gaussian","bernoulli","Gamma","beta","quantile","quantile_scaled"))stop("family must be one of Poisson, Binomial, Gaussian, Gamma, Beta, or quantile")
                           form <- gsub(" ","",self$formula)
                           form <- gsub("~","",self$formula)
                           if(grepl("nngp",form)){
                             self$covariance$.__enclos_env__$private$type <- 1
                             form <- gsub("nngp_","",form)
                           } else if(grepl("hsgp",form)){
                             self$covariance$.__enclos_env__$private$type <- 2
                             form <- gsub("hsgp_","",form)
                           } else if(grepl("spde",form)){
                             self$covariance$.__enclos_env__$private$type <- 4
                             form <- gsub("spde_","",form)
                           } else if(grepl("ar_",form)){
                             self$covariance$.__enclos_env__$private$type <- 3
                             self$covariance$.__enclos_env__$private$time <- as.integer(sub(".*t=(\\d{1,2}).*", "\\1", gsub(" ","",form)))
                             form <- sub(",t=\\d{1,2}", "", sub("ar_", "", gsub(" ","",form)))
                           }
                           type <- private$model_type()
                           if(type %in% c(0,1,2,4)){
                             data <- self$covariance$data
                             if(any(!colnames(self$mean$data)%in%colnames(data))){
                               cnames <- which(!colnames(self$mean$data)%in%colnames(data))
                               data <- cbind(data,self$mean$data[,cnames,drop=FALSE])
                             }
                           } else {
                             data <- self$mean$data
                           }
                           
                           if(self$family[[1]]=="bernoulli" & any(self$trials>1))self$family[[1]] <- "binomial"
                           
                           if(type == 0){
                             private$ptr <- Model__new(form,
                                                       as.matrix(data),
                                                       colnames(data),
                                                       tolower(self$family[[1]]),
                                                       self$family[[2]])
                           } else if(type==1){
                             nngp <- self$covariance$nngp()
                             private$ptr <- Model_nngp__new(form,
                                                            as.matrix(data),
                                                            colnames(data),
                                                            tolower(self$family[[1]]),
                                                            self$family[[2]],
                                                            nngp[2])
                           } else if(type==2){
                             private$ptr <- Model_hsgp__new(form,
                                                            as.matrix(data),
                                                            colnames(data),
                                                            tolower(self$family[[1]]),
                                                            self$family[[2]])
                           } else if(type==3) {
                             n_A <- nrow(self$covariance$data) / self$covariance$.__enclos_env__$private$time
                             self$covariance$data <- self$covariance$data[1:n_A,]
                             private$ptr <- Model_ar__new(form,
                                                          as.matrix(data),
                                                          as.matrix(self$covariance$data),
                                                          colnames(data),
                                                          colnames(self$covariance$data),
                                                          tolower(self$family[[1]]),
                                                          self$family[[2]],
                                                          self$covariance$.__enclos_env__$private$time)
                           } else if(type==4){
                             private$ptr <- Model_spde__new(form,
                                                            as.matrix(data),
                                                            colnames(data),
                                                            tolower(self$family[[1]]),
                                                            self$family[[2]])
                             Model_spde__set_spde_data(xp = private$ptr, private$mesh[["A_loc"]], private$mesh[["C"]], private$mesh[["G"]], 2)
                           }
                           
                           Model__update_beta(private$ptr,self$mean$parameters,type)
                           if(type==2 & !is.null(self$covariance$parameters) & length(self$covariance$parameters) > 2){
                             Model_hsgp__set_anisotropic(private$ptr)
                           }
                           ncovpar <- Model__n_cov_pars(private$ptr,type)
                           if(is.null(self$covariance$parameters)){
                             self$covariance$parameters <- runif(ncovpar,0,0.05)
                           } 
                           Model__update_theta(private$ptr,self$covariance$parameters,type)
                           re <- Model__re_terms(private$ptr,type)
                           paridx <- Model__parameter_fn_index(private$ptr,type)+1
                           names(self$covariance$parameters) <- re[paridx]
                           Model__set_offset(private$ptr,self$mean$offset,type)
                           Model__set_weights(private$ptr,self$weights,type)
                           Model__set_var_par(private$ptr,self$var_par,type)
                           if(self$family[[1]] == "binomial" | self$family[[1]] == "bernoulli")Model__set_trials(private$ptr,self$trials,type)
                           if(self$family[[1]] %in% c("quantile","quantile_scaled")) Model__set_quantile(private$ptr,self$family$q,type)
                           #Model__update_u(private$ptr,matrix(rnorm(Model__Q(private$ptr,type)),ncol=1),type) # initialise random effects to random
                           if(!private$useSparse & type == 1) Model__make_dense(private$ptr,type)
                           # set covariance pointer
                           self$covariance$.__enclos_env__$private$model_ptr <- private$ptr
                           self$covariance$.__enclos_env__$private$ptr <- NULL
                           self$covariance$.__enclos_env__$private$cov_form()
                           # re=initiliase just in case
                           Model__update_beta(private$ptr,self$mean$parameters,type)
                           Model__update_theta(private$ptr,self$covariance$parameters,type)
                           private$session_id <- Sys.getpid()
                         }
                       },
                       verify_data = function(y){
                         if(any(is.na(y)))stop("NAs in y")
                         if(self$family[[1]]=="binomial"){
                           if(!(all(y>=0)))stop("y must be integer >= 0")
                           if(any(y > self$trials))stop("Number of successes > number of trials")
                         } else if(self$family[[1]]=="poisson"){
                           if(any(y <0) )stop("y must be integer >= 0")
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
                           cnames <- c(cnames, colnames(self$mean$data)[which(!colnames(self$mean$data)%in%cnames)])
                         }
                         if(!isTRUE(all.equal(cnames,colnames(newdata)))){
                           newdat <- newdata[,cnames[cnames%in%colnames(newdata)],drop=FALSE]
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
                         newdat <- newdat[,cnames,drop=FALSE]
                         return(newdat)
                       },
                       model_data_frame = function(){
                         cnames <- colnames(self$mean$data)
                         dat <- self$mean$data
                         if(any(!colnames(self$covariance$data)%in%cnames)){
                           dat <- cbind(dat, self$covariance$data[,which(!colnames(self$covariance$data)%in%cnames), drop = FALSE])
                         }
                         return(dat)
                       },
                       check_y_formula = function(form,data,family){
                         # assume formula is a string
                         str1 <- as.character(form)
                         if(length(str1)==3){
                           private$y_in_formula <- TRUE
                           if(grepl("cbind",str1[[2]])){
                             if(!family[[1]]%in%c("binomial"))stop("cbind specification of outcome requires binomial model")
                             yc <- unlist(lapply(regmatches(str1[2], gregexpr("\\(.*?\\)", str1[2])), function(x)gsub("[\\(\\)]","",x)))
                             yc <- unlist(strsplit(unlist(yc),","))
                             yc <- gsub(" ","",yc)
                             if(!yc[1]%in%colnames(data))stop(paste0(yc[1]," not in data"))
                             private$y_name <- yc[1]
                             self$trials <- eval(parse(text = yc[2]),envir = data) + data[,yc[1]]
                           } else {
                             if(!str1[2]%in%colnames(data))stop(paste0(str1[2]," not in data"))
                             private$y_name <- str1[2]
                           }
                           return(as.formula(paste0("~",str1[3])))
                         } else {
                           return(form)
                         }
                       },
                       process_data = function(form,data,is_covariance = FALSE,is_mean = TRUE){
                         s1 <- c()
                         f1 <- as.character(form)[2]
                         f1 <- gsub(" ","",f1[length(f1)])
                         cnames <- colnames(data)
                         result <- list(formula= NA, data = NA)
                         # first check if we can just use R's functions:
                         if(is_mean){
                           r1 <- re_names(f1)
                           f2 <- f1
                           for(i in 1:length(r1)){
                             r1[i] <- gsub("\\(","\\\\(",r1[i])
                             r1[i] <- gsub("\\|","\\\\|",r1[i])
                             r1[i] <- gsub("\\)","\\\\)",r1[i])
                             f2 <- gsub(paste0("\\+",r1[i]),"",f2)
                           }
                           mm_result <- tryCatch(model.matrix(as.formula(paste0("~",f2)),data),
                                                 error = function(e)return(NA))
                           if(!is(mm_result,"logical")){
                             if(any(colnames(mm_result)=="(Intercept)")) mm_result <- mm_result[,-which(colnames(mm_result)=="(Intercept)"),drop=FALSE]
                             for(i in 1:ncol(mm_result)){
                               colnames(mm_result)[i] <- gsub("-","",colnames(mm_result)[i])
                               colnames(mm_result)[i] <- gsub("+","",colnames(mm_result)[i])
                               if(grepl("factor",colnames(mm_result)[i])){
                                 colnames(mm_result)[i] <- gsub("factor","",colnames(mm_result)[i])
                                 colnames(mm_result)[i] <- gsub("\\(","",colnames(mm_result)[i])
                                 colnames(mm_result)[i] <- gsub("\\)","",colnames(mm_result)[i])
                               }
                             }
                             new_formula <- paste0(colnames(mm_result),collapse = "+")
                             if(grepl("-1",f1))new_formula <- paste0(new_formula,"-1")
                             r1 <- re_names(f1)
                             for(i in 1:length(r1)){
                               new_formula <- paste0(new_formula,"+",r1[i])
                             }
                             new_formula <- as.formula(paste0("~ ",new_formula))
                             
                             result <- list(formula = new_formula, data = as.data.frame(mm_result))
                           } 
                         }
                         
                         if(is(result$formula,"logical")){
                           if(is_covariance){
                             s1a <- re_names(f1, FALSE)
                             s1a <- s1a[s1a!="1"]
                             s1a <- Reduce(c,strsplit(s1a,"\\*"))
                             s1alen <- Reduce(c,gregexpr("\\(.*?\\)", s1a))
                             s1akeep <- which(s1alen < 0)
                             s1b <- lapply(regmatches(s1a, gregexpr("\\(.*?\\)", s1a)), function(x)gsub("[\\(\\)]","",x))
                             s1b <- Reduce(c,lapply(s1b,function(x)strsplit(x,",")))
                             s1b[s1akeep] <- s1a[s1akeep]
                             s1 <- c(s1,unique(Reduce(c,s1b)))
                           } 
                           if(is_mean) {
                             s1 <- c(s1,get_variable_names(f1,cnames))
                           }
                           data_idx <- match(s1,cnames)
                           data_idx <- data_idx[!is.na(data_idx)]
                           new_data <- data[,data_idx,drop=FALSE]
                           if(ncol(new_data)>0){
                             for(i in 1:ncol(new_data)){
                               if(is(new_data[,i],"character")|is(new_data[,i],"factor"))new_data[,i] <- as.numeric(as.factor(new_data[,i]))
                             }
                           }
                           return(list(formula = form, data = new_data))
                         } else {
                           return(result)
                         }
                         
                       }
                     ))

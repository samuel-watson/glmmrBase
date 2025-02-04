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
                                          m=0
                       ){
                         preddata <- private$model_data(newdata)
                         out <- Model__predict(private$ptr,as.matrix(preddata),offset,m,private$model_type())
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
                                             model_fit = NULL){
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
                             # if(is(covariance,"list")){
                             #   if(is.null(covariance$formula))stop("A formula must be specified for the covariance")
                             #   if(is.null(covariance$data) & is.null(data))stop("No data specified in covariance list or call to function.")
                             #   self$covariance <- Covariance$new(
                             #     formula= covariance$formula
                             #   )
                             #   if(is.null(covariance$data)){
                             #     self$covariance$data <- private$process_data(self$covariance$formula,data,TRUE,FALSE)$data
                             #   } else {
                             #     self$covariance$data <- covariance$data
                             #   }
                             #   if(!is.null(covariance$parameters))self$covariance$update_parameters(covariance$parameters)
                             # }
                             if(is(mean,"R6")){
                               if(is(mean,"MeanFunction")){
                                 self$mean <- mean
                               } else {
                                 stop("mean should be MeanFunction class or parameter vector")
                               }
                             } else {
                               stop("mean should be MeanFunction class or parameter vector")
                             }
                             
                             #   if(is(mean,"list")){
                             #   if(is.null(mean$formula))stop("A formula must be specified for the mean function.")
                             #   if(is.null(mean$data) & is.null(data))stop("No data specified in mean list or call to function.")
                             #   if(is.null(mean$data)){
                             #     processed_data <- private$process_data(form_1,data,TRUE,FALSE)
                             #     self$mean <- MeanFunction$new(
                             #       formula = processed_data$formula,
                             #       data = processed_data$data
                             #     )
                             #   } else {
                             #     self$mean <- MeanFunction$new(
                             #       formula = mean$formula,
                             #       data = mean$data
                             #     )
                             #   }
                             #   if(!is.null(mean$parameters))self$mean$update_parameters(mean$parameters)
                             # }
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
                       information_matrix = function(include.re = FALSE, theta = FALSE, oim = FALSE){
                         if(oim & !private$y_has_been_updated) stop("No y data has been added")
                         private$update_ptr()
                         nonlin <- Model__any_nonlinear(private$ptr,private$model_type())
                         # if(nonlin){
                         #   if(!private$y_has_been_updated) stop("No y data has been added")
                         #   # if(!hessian.corr %in% c("add","return","none"))stop("hessian.corr must be add, return, or none")
                         # }
                         if(oim){
                           M <- Model__observed_information_matrix(private$ptr,private$model_type())
                         } else {
                           if(theta){
                             M <- Model__infomat_theta(private$ptr,private$model_type())
                           } else {
                             if(include.re & !private$model_type()>0){
                               M <- Model__obs_information_matrix(private$ptr,private$model_type())
                             } else {
                               if(private$model_type()==0){
                                 M <- Model__information_matrix(private$ptr,private$model_type())
                               } else {
                                 M <- Model__information_matrix_crude(private$ptr,private$model_type())
                               }
                             }
                             # if((nonlin & hessian.corr%in%c("add","none")) | !nonlin){
                             #   
                             # }
                             # if(Model__any_nonlinear(private$ptr,private$model_type()) & hessian.corr %in% c("add","return")){
                             #   A <- Model__hessian_correction(private$ptr,private$model_type())
                             #   if(any(eigen(A)$values < 0) & adj.nonspd){
                             #     if(adj.nonspd)message("Hessian correction for non-linear parameters is not positive semi-definite and will be adjusted. To disable this feature set adj.nonspd = FALSE")
                             #     A <- near_semi_pd(A)
                             #   }
                             #   if(hessian.corr == "add"){
                             #     M <- M + A
                             #   } else {
                             #     M <- A
                             #   }
                             # }
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
                       #'Stochastic Maximum Likelihood model fitting
                       #'
                       #'@details
                       #'**Monte Carlo maximum likelihood**
                       #'Fits generalised linear mixed models using one of several algorithms: Markov Chain Newton
                       #'Raphson (MCNR), Markov Chain Expectation Maximisation (MCEM), or stochastic approximation expectation 
                       #'maximisation (SAEM) with or without Polyak-Ruppert averaging. MCNR and MCEM are described by McCulloch (1997)
                       #'<doi:10.1080/01621459.1997.10473613>. For each iteration
                       #'of the algorithms the unobserved random effect terms (\eqn{\gamma}) are simulated
                       #'using Markov Chain Monte Carlo (MCMC) methods,
                       #'and then these values are conditioned on in the subsequent steps to estimate the covariance
                       #'parameters and the mean function parameters (\eqn{\beta}). SAEM uses a Robbins-Munroe approach to approximating 
                       #'the likelihood and requires fewer MCMC samples and may have lower Monte Carlo error, see Jank (2006)<doi:10.1198/106186006X157469>. 
                       #'The option `alpha` determines the rate at which succesive iterations "forget" the past and must be between 0.5 and 1. Higher values
                       #'will result in lower Monte Carlo error but slower convergence. The options `mcem.adapt` and `mcnr.adapt` will modify the number of MCMC samples during each step of model fitting 
                       #'using the suggested values in Caffo, Jank, and Jones (2006)<doi:10.1111/j.1467-9868.2005.00499.x> 
                       #'as the estimates converge. 
                       #'
                       #'The accuracy of the algorithm depends on the user specified tolerance. For higher levels of
                       #'tolerance, larger numbers of MCMC samples are likely need to sufficiently reduce Monte Carlo error. However,
                       #'the SAEM approach does overcome reduce the required samples, especially with R-P averaging. As such a lower number (20-50)
                       #'samples per iteration is normally sufficient to get convergence. 
                       #'
                       #'There are several stopping rules for the algorithm. Either the algorithm will terminate when succesive parameter estimates are
                       #'all within a specified tolerance of each other (`conv.criterion = 1`), or when there is a high probability that the estimated 
                       #'log-likelihood has not been improved. This latter criterion can be applied to either the overall log-likelihood (`conv.criterion = 2`),
                       #'the likelihood just for the fixed effects (`conv.criterion = 3`), or both the likelihoods for the fixed effects and covariance parameters 
                       #'(`conv.criterion = 4`; default).
                       #'
                       #' Options for the MCMC sampler are set by changing the values in `self$mcmc_options`. The information printed to the console
                       #' during model fitting can be controlled with the `self$set_trace()` function.
                       #' 
                       #' To provide weights for the model fitting, store them in self$weights. To set the number of 
                       #' trials for binomial models, set self$trials.
                       #' 
                       #'@param y Optional. A numeric vector of outcome data. If this is not provided then either the outcome must have been specified when 
                       #' initialising the Model object, or the outcome data has been updated using member function `update_y()`
                       #'@param method The MCML algorithm to use, either `mcem` or `mcnr`, or `saem` see Details. Default is `saem`. `mcem.adapt` and `mcnr.adapt` will use adaptive 
                       #'MCMC sample sizes starting small and increasing to the the maximum value specified in `mcmc_options$sampling`, which results in faster convergence. `saem` uses a
                       #'stochastic approximation expectation maximisation algorithm. MCMC samples are kept from all iterations and so a smaller number of samples are needed per iteration. 
                       #'@param tol Numeric value, tolerance of the MCML algorithm, the maximum difference in parameter estimates
                       #'between iterations at which to stop the algorithm. If two values are provided then different tolerances will be 
                       #'applied to the fixed effect and covariance parameters.
                       #'@param max.iter Integer. The maximum number of iterations of the MCML algorithm.
                       #'@param se String. Type of standard error and/or inferential statistics to return. Options are "gls" for GLS standard errors (the default),
                       #' "robust" for robust standard errors, "kr" for original Kenward-Roger bias corrected standard errors, 
                       #' "kr2" for the improved Kenward-Roger correction, "sat" for Satterthwaite degrees of freedom correction (this is the same 
                       #' degrees of freedom correction as Kenward-Roger, but with GLS standard errors), "box" to use a modified Box correction (does not return confidence intervals),
                       #' "bw" to use GLS standard errors with a between-within correction to the degrees of freedom, "bwrobust" to use robust 
                       #' standard errors with between-within correction to the degrees of freedom.
                       #'@param oim Logical. If TRUE use the observed information matrix, otherwise use the expected information matrix for standard error and related calculations.
                       #'@param reml Logical. Whether to use a restricted maximum likelihood correction for fitting the covariance parameters
                       #'@param mcmc.pkg String. Either `cmdstan` for cmdstan (requires the package `cmdstanr`), `rstan` to use rstan sampler, or
                       #'`hmc` to use a cruder Hamiltonian Monte Carlo sampler. cmdstan is recommended as it has by far the best number 
                       #' of effective samples per unit time. cmdstanr will compile the MCMC programs to the library folder the first time they are run, 
                       #' so may not currently be an option for some users.
                       #'@param se.theta Logical. Whether to calculate the standard errors for the covariance parameters. This step is a slow part
                       #' of the calculation, so can be disabled if required in larger models. Has no effect for Kenward-Roger standard errors.
                       #'@param algo Integer. 1 = L-BFGS for beta and BOBYQA for theta, 2 = BOBYQA for both, 3 = L-BFGS for both (default). The L-BFGS algorithm 
                       #'may perform poorly with some covariance structures, in this case select 1 or 2, or apply an upper bound.
                       #'@param lower.bound Optional. Vector of lower bounds for the fixed effect parameters. To apply bounds use MCEM.
                       #'@param upper.bound Optional. Vector of upper bounds for the fixed effect parameters. To apply bounds use MCEM.
                       #'@param lower.bound.theta Optional. Vector of lower bounds for the covariance parameters (default is 0; negative values will cause an error)
                       #'@param upper.bound.theta Optional. Vector of upper bounds for the covariance parameters. 
                       #'@param alpha If using SAEM then this parameter controls the step size. On each iteration i the step size is (1/alpha)^i, default is 0.8. Values around 0.5 
                       #'will result in lower bias but slower convergence, values closer to 1 will result in higher convergence but potentially higher error.
                       #'@param convergence.prob Numeric value in (0,1) indicating the probability of convergence if using convergence criteria 2, 3, or 4.
                       #'@param pr.average Logical indicating whether to use Polyak-Ruppert averaging if using the SAEM algorithm (default is TRUE)
                       #'@param conv.criterion Integer. The convergence criterion for the algorithm. 1 = the maximum difference between parameter estimates between iterations as defined by `tol`,
                       #'2 = The probability of improvement in the overall log-likelihood is less than 1 - `convergence.prob`
                       #'3 = The probability of improvement in the log-likelihood for the fixed effects is less than 1 - `convergence.prob`
                       #'4 = The probabilities of improvement in the log-likelihood the fixed effects and covariance parameters are both less than 1 - `convergence.prob`
                       #'@param skip.theta Logical. If TRUE then the covariance parameter estimation step is skipped. This option is mainly used for testing, but may be useful
                       #'if covariance parameters are known.
                       #'@return A `mcml` object
                       #'@seealso \link[glmmrBase]{Model}, \link[glmmrBase]{Covariance}, \link[glmmrBase]{MeanFunction}
                       #'@examples
                       #'\dontrun{
                       #' # Simulated trial data example
                       #'data(SimTrial,package = "glmmrBase")
                       #' model <- Model$new(
                       #'   formula = y ~ int + factor(t) - 1 + (1|gr(cl)*ar1(t)),
                       #'   data = SimTrial,
                       #'   family = gaussian()
                       #' )
                       #' glm3 <- model$MCML()
                       #' 
                       #' # Salamanders data example
                       #' data(Salamanders,package="glmmrBase")
                       #' model <- Model$new(
                       #'   mating~fpop:mpop-1+(1|gr(mnum))+(1|gr(fnum)),
                       #'   data = Salamanders,
                       #'   family = binomial()
                       #' )
                       #' 
                       #' # we will try MCEM with 500 MCMC iterations
                       #' model$mcmc_options$samps <- 500
                       #' # view the grouping structure 
                       #' glm2 <- model$MCML(method = "mcem")
                       #'
                       #' # Example using simulated data
                       #' #create example data with six clusters, five time periods, and five people per cluster-period
                       #' df <- nelder(~(cl(6)*t(5)) > ind(5))
                       #' # parallel trial design intervention indicator
                       #' df$int <- 0
                       #' df[df$cl > 3, 'int'] <- 1 
                       #' # specify parameter values in the call for the data simulation below
                       #' des <- Model$new(
                       #'   formula= ~ factor(t) + int - 1 +(1|gr(cl)*ar0(t)),
                       #'   covariance = c(0.05,0.7),
                       #'   mean = c(rep(0,5),0.2),
                       #'   data = df,
                       #'   family = gaussian()
                       #' )
                       #' ysim <- des$sim_data() # simulate some data from the model
                       #' fit1 <- des$MCML(y = ysim) # Default model fitting with SAEM-PR
                       #' # use MCEM instead and stop when parameter values are within 1e-2 on successive iterations
                       #' fit2 <- des$MCML(y = ysim, method="mcem",tol=1e-2,conv.criterion = 1)
                       #' 
                       #' # Non-linear model fitting example using the example provided by nlmer in lme4
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
                       #' 
                       #' # for this example, we will use MCEM with adaptive MCMC sample sizes
                       #' 
                       #' model$mcmc_options$samps <- 1000
                       #' nfit <- model$MCML(method = "mcem.adapt")
                       #' 
                       #' summary(nfit)
                       #' summary(nm1)
                       #' 
                       #' 
                       #'}
                       #'@md
                       MCML = function(y = NULL,
                                       method = "saem",
                                       tol = 1e-2,
                                       max.iter = 50,
                                       se = "gls",
                                       oim = FALSE,
                                       reml = TRUE,
                                       mcmc.pkg = "rstan",
                                       se.theta = TRUE,
                                       algo = ifelse(self$mean$any_nonlinear(),2,1),
                                       lower.bound = NULL,
                                       upper.bound = NULL,
                                       lower.bound.theta = NULL,
                                       upper.bound.theta = NULL,
                                       alpha = 0.8,
                                       convergence.prob = 0.95,
                                       pr.average = FALSE,
                                       conv.criterion = 2,
                                       skip.theta = FALSE){
                         # Checks on options and data
                         if(is.null(y)){
                           if(!private$y_has_been_updated)stop("y not specified and not updated in Model object")
                         } else {
                           private$verify_data(y)
                           private$set_y(y)
                         }
                         Model__use_attenuation(private$ptr,private$attenuate_parameters,private$model_type())
                         if(!se %in% c("gls","kr","kr2","bw","sat","bwrobust","box"))stop("Option se not recognised")
                         if(self$family[[1]]%in%c("Gamma","beta") & se %in% c("kr","kr2","sat"))stop("KR standard errors are not currently available with gamma or beta families")
                         if(se != "gls" & private$model_type() != 0)stop("Only GLS standard errors supported for GP approximations.")
                         if(se == "box" & !(self$family[[1]]=="gaussian"&self$family[[2]]=="identity"))stop("Box only available for linear models")
                         if(!mcmc.pkg %in% c("cmdstan","rstan","hmc"))stop("mcmc.pkg must be one of cmdstan, rstan, or hmc")
                         if(!method %in% c("mcem", "mcnr", "saem", "mcem.adapt", "mcnr.adapt"))stop("method must be either mcem, mcnr, saem, mcem.adapt, mcnr.adapt")
                         if(self$family[[1]]%in%c("quantile","quantile_scaled") & method == "mcnr")stop("MCNR with quantile currently disabled, please use SAEM or MCEM with MCML")
                         append_u <- FALSE
                         if(mcmc.pkg == "hmc" & method == "saem")stop("saem and hmc options not currently compatible")
                         adaptive <- method %in% c("mcnr.adapt","mcem.adapt")
                         if(!conv.criterion %in% 1:4)stop("Convergence criterion must be 1, 2, or 3")
                         if(alpha < 0.5 | alpha >= 1)stop("alpha must be in [0.5, 1)")
                         if(convergence.prob <= 0 | convergence.prob >= 1)stop("Convergence probability must be in (0, 1)")
                         if(self$family[[1]]%in%c("quantile","quantile_scaled")){
                           Model__set_quantile(private$ptr,self$family$q,private$model_type())
                           message("Quantile regression is EXPERIMENTAL.")
                         }
                         if(!mcmc.pkg == "hmc"){
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
                         if(length(tol) == 1){
                           algo_tol <- rep(tol,2)
                         } else {
                           algo_tol <- tol[1:2]
                         }
                         Model__use_reml(private$ptr,reml,private$model_type())
                         Model__reset_fn_counter(private$ptr,private$model_type())
                         # set up all the required vectors and data to monitor the algorithm
                         balgo <- ifelse(algo %in% c(1,3) ,2,0) # & !self$mean$any_nonlinear()
                         if(method == "saem") balgo <- 0
                         beta <- self$mean$parameters
                         theta <- self$covariance$parameters
                         var_par <- self$var_par
                         var_par_family <- I(self$family[[1]]%in%c("gaussian","Gamma","beta","quantile_scaled"))
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
                         file_type <- mcnr_family(self$family,mcmc.pkg == "cmdstan")                         
                         ## set up sampler
                         if(mcmc.pkg == "cmdstan"){
                           if(!requireNamespace("cmdstanr")){
                             stop("cmdstanr is required to use Stan for sampling. See https://mc-stan.org/cmdstanr/ for details on how to install.
                                    Set option usestan=FALSE to use the in-built MCMC sampler.")
                           } else {
                             if(private$trace >= 2)message("If this is the first time running this model, it will be compiled by cmdstan.")
                             model_file <- system.file("cmdstan",
                                                       file_type$file,
                                                       package = "glmmrBase",
                                                       mustWork = TRUE)
                             mod <- suppressMessages(cmdstanr::cmdstan_model(model_file))
                           }
                         }
                         # SET UP MCMC DATA STRUCTURE
                         data <- list(
                           Q = Model__Q(private$ptr,private$model_type()),
                           Xb = Model__xb(private$ptr,private$model_type()),
                           Z = Model__ZL(private$ptr,private$model_type()),
                           type=as.numeric(file_type$type)
                         )
                         if(self$family[[1]]%in%c("gaussian","beta","Gamma","quantile","quantile_scaled"))data <- append(data,list(N_cont = self$n(),
                                                                                  N_int = 1,
                                                                                  N_binom = 1,
                                                                                  sigma = rep(self$var_par/self$weights, self$n()),
                                                                                  ycont = Model__y(private$ptr,private$model_type()),
                                                                                  yint = array(0,dim = 1),
                                                                                  q = 0,
                                                                                  n = array(0,dim = 1)))
                         if(self$family[[1]]%in%c("binomial","bernoulli","poisson"))data <- append(data,list(N_int = self$n(),
                                                                                                      N_cont = 1,
                                                                                                      N_binom = 1,
                                                                                                      sigma = array(0,dim = 1),
                                                                                                      yint = Model__y(private$ptr,private$model_type()),
                                                                                                      ycont = array(0,dim = 1),
                                                                                                      q = 0,
                                                                                                      n = array(0,dim = 1)))
                         if(self$family[[1]]=="binomial")data <- append(data,list(N_binom  = self$n(), n = self$trials))
                         if(self$family[[1]]%in%c("beta","Gamma","quantile","quantile_scaled"))data$sigma = rep(self$var_par,self$n())
                         if(self$family[[1]]%in%c("quantile","quantile_scaled"))data$q = self$family$q
                         iter <- 0
                         n_mcmc_sampling <- ifelse(adaptive, 20, self$mcmc_options$samps)
                         beta_diff <- 1
                         theta_diff <- 1
                         converged <- FALSE
                         # run one iteration of fitting beta without re (i.e. glm) to get reasonable starting values
                         # otherwise the algorithm can struggle to converge
                         Model__update_u(private$ptr,matrix(0,nrow = Model__Q(private$ptr,private$model_type()),ncol=1),FALSE,private$model_type())
                         if(private$trace >= 1)cat("\nIter: 0\n")
                         Model__set_sml_parameters(private$ptr, FALSE, self$mcmc_options$samps, alpha, pr.average, private$model_type())
                         Model__ml_beta(private$ptr,0,private$model_type())
                         beta <- Model__get_beta(private$ptr,private$model_type())
                         var_par <- Model__get_var_par(private$ptr,private$model_type())
                         all_pars <- c(beta,theta)
                         if(var_par_family) all_pars <- c(all_pars,var_par)
                         if(private$trace >= 1){
                           cat("\nStarting Beta (GLM): ", round(beta,5))
                           cat("\n",Reduce(paste0,rep("-",40)),"\n")
                         }
                         Model__set_sml_parameters(private$ptr, method == "saem", self$mcmc_options$samps, alpha, pr.average, private$model_type())
                         # START THE MAIN ALGORITHM
                         while(!converged & iter < max.iter){
                           if(iter > 0){
                             all_pars <- all_pars_new
                             beta <- beta_new
                             theta <- theta_new
                           }
                           iter <- iter + 1
                           append_u <- I(method=="saem" & iter > 1)
                           if(private$trace >= 1)cat("\nIter: ",iter,"\n",Reduce(paste0,rep("-",40)),"\n")
                           if(private$trace == 2)t1 <- Sys.time()
                           if(mcmc.pkg == "cmdstan" | mcmc.pkg == "rstan"){
                             data$Xb <-  Model__xb(private$ptr,private$model_type())
                             data$Z <- Model__ZL(private$ptr,private$model_type())
                             if(self$family[[1]]=="gaussian")data$sigma = var_par_new/self$weights
                             if(self$family[[1]]%in%c("beta","Gamma"))data$var_par = var_par_new
                             if(private$trace <= 1){
                               if(mcmc.pkg == "cmdstan"){
                                 capture.output(fit <- mod$sample(data = data,
                                                                  chains = self$mcmc_options$chains,
                                                                  iter_warmup = self$mcmc_options$warmup,
                                                                  iter_sampling = n_mcmc_sampling,
                                                                  refresh = 0),
                                                file=tempfile())
                               } else {
                                 capture.output(suppressWarnings(fit <- rstan::sampling(stanmodels[[file_type$file]],
                                                                                        data=data,
                                                                                        chains=self$mcmc_options$chains,
                                                                                        iter = self$mcmc_options$warmup+n_mcmc_sampling,
                                                                                        warmup = self$mcmc_options$warmup,
                                                                                        refresh = 0)),
                                                file=tempfile())
                               }
                             } else {
                               # warnings have been suppressed below as it warns about R-hat etc, which is not reliable with a single chain.
                               if(mcmc.pkg == "cmdstan"){
                                 suppressWarnings(fit <- mod$sample(data = data,
                                                                    chains = self$mcmc_options$chains,
                                                                    iter_warmup = self$mcmc_options$warmup,
                                                                    iter_sampling = n_mcmc_sampling,
                                                                    refresh = 50))
                               } else {
                                 suppressWarnings(fit <- rstan::sampling(stanmodels[[file_type$file]],
                                                                         data=data,
                                                                         chains=chains,
                                                                         iter = self$mcmc_options$warmup+n_mcmc_sampling,
                                                                         warmup = self$mcmc_options$warmup,
                                                                         refresh = 50))
                               }
                             }
                             if(mcmc.pkg == "cmdstan"){
                               dsamps <- fit$draws("gamma",format = "matrix")
                               class(dsamps) <- "matrix"
                             } else {
                               dsamps <- rstan::extract(fit,pars = "gamma",permuted = FALSE)
                               dsamps <- as.matrix(dsamps[,1,])
                             }
                             Model__update_u(private$ptr,t(dsamps),append_u,private$model_type())
                           } else {
                             Model__mcmc_sample(private$ptr, self$mcmc_options$warmup, n_mcmc_sampling, self$mcmc_options$adapt, private$model_type())
                           }
                           if(private$trace==2)t2 <- Sys.time()
                           if(private$trace==2)cat("\nMCMC sampling took: ",t2-t1,"s")
                           if(method=="mcem" | method=="saem"){
                             Model__ml_beta(private$ptr,balgo,private$model_type())
                           } else {
                             Model__nr_beta(private$ptr,private$model_type())
                           }
                           if(!skip.theta){
                             if(algo == 3){ #& !self$mean$any_nonlinear()
                               tryCatch(Model__ml_theta(private$ptr,2,private$model_type()),
                                        error = function(e) {
                                          if(private$trace >= 1)cat("\nL-BFGS failed for theta, switching to BOBYQA");
                                          Model__ml_theta(private$ptr,0,private$model_type());
                                        })
                             } else {
                               Model__ml_theta(private$ptr,0,private$model_type())
                             }
                           }
                           # set up the vectors needed 
                           beta_new <- Model__get_beta(private$ptr,private$model_type())
                           theta_new <- Model__get_theta(private$ptr,private$model_type())
                           var_par_new <- Model__get_var_par(private$ptr,private$model_type())
                           all_pars_new <- c(beta_new,theta_new)
                           llvals <- Model__get_log_likelihood_values(private$ptr,private$model_type())
                           beta_diff <- max(abs(beta-beta_new))
                           theta_diff <- max(abs(theta-theta_new))
                           fn_counter <- Model__get_fn_counter(private$ptr,private$model_type())
                           if(conv.criterion == 1){
                             converged <- !(beta_diff > algo_tol[1]) & !(theta_diff > algo_tol[2])
                           } 
                           if(iter > 1){
                             udiagnostic <- Model__u_diagnostic(private$ptr,private$model_type())
                             uval <- ifelse(conv.criterion == 2, Reduce(sum,udiagnostic), udiagnostic$first)
                             llvar <- Model__ll_diff_variance(private$ptr, TRUE, conv.criterion==2, private$model_type())
                             if(adaptive) n_mcmc_sampling <- max(n_mcmc_sampling, min(self$mcmc_options$samps, ceiling(llvar * (qnorm(convergence.prob) + qnorm(0.8))^2)/uval^2))
                             if(conv.criterion %in% c(2,3)){
                               conv.criterion.value <- uval + qnorm(convergence.prob)*sqrt(llvar/n_mcmc_sampling)
                               prob.converged <- pnorm(-uval/sqrt(llvar/n_mcmc_sampling))
                               converged <- conv.criterion.value < 0
                             } 
                             if(conv.criterion == 4){
                               llvart <- Model__ll_diff_variance(private$ptr, FALSE, TRUE, private$model_type())
                               conv.criterion.value <- udiagnostic$first + qnorm(convergence.prob)*sqrt(llvar/n_mcmc_sampling)
                               prob.converged <- pnorm(-udiagnostic$first/sqrt(llvar/n_mcmc_sampling))
                               conv.criterion.valuet <- udiagnostic$second + qnorm(convergence.prob)*sqrt(llvart/n_mcmc_sampling)
                               prob.convergedt <- pnorm(-udiagnostic$second/sqrt(llvart/n_mcmc_sampling))
                               converged <- conv.criterion.value < 0 & conv.criterion.valuet < 0
                             }
                           }
                           if(var_par_family)all_pars_new <- c(all_pars_new,var_par_new)
                           if(private$trace==2)t3 <- Sys.time()
                           if(private$trace==2)cat("\nModel fitting took: ",t3-t2,"s")
                           if(private$trace >= 1){
                             cat("\nBeta: ", round(beta_new,5))
                             cat("\nMax beta difference: ", round(beta_diff,5))
                             cat("\nTheta: ", round(theta_new,5))
                             cat("\nMax theta difference: ", round(theta_diff,5))
                             if(var_par_family)cat("\nSigma: ",round(var_par_new,5))
                             cat("\nMax. difference : ", round(max(abs(all_pars-all_pars_new)),5))
                             cat("\nLog-likelihoods: beta ", round(llvals$first,5)," theta ",round(llvals$second,5))
                             cat("\nFn evaluations: beta ",fn_counter$first," theta ",fn_counter$second)
                             if(iter>1){
                               if(adaptive)cat("\nMCMC sample size (adaptive): ",n_mcmc_sampling)
                               cat("\nLog-lik diff values: ", round(udiagnostic$first,5),", ", round(udiagnostic$second,5)," overall: ", round(Reduce(sum,udiagnostic), 5))
                               cat("\nLog-lik variance: ", round(llvar,5))
                               if(conv.criterion >= 2)cat(" convergence criterion:", ifelse(conv.criterion == 4, " (beta) "," "), round(conv.criterion.value,5)," Prob.: ",round(prob.converged,3))
                               if(conv.criterion == 4)cat(" (theta) ", round(conv.criterion.valuet,5)," Prob.: ",round(prob.convergedt,3))
                             }
                             cat("\n",Reduce(paste0,rep("-",40)),"\n")
                           }
                         }
                         if(!converged)message(paste0("Algorithm not converged. Max. difference between iterations :",round(max(abs(all_pars-all_pars_new)),4)))
                         self$update_parameters(mean.pars = beta_new,
                                                cov.pars = theta_new)
                         if(private$trace >= 1)cat("\n\nCalculating standard errors...\n")
                         self$var_par <- var_par_new
                         u <- Model__u(private$ptr,TRUE,private$model_type())
                         if(private$model_type()==0){
                           if(se == "gls" || se == "bw" || se == "box"){
                             M <- self$information_matrix() #Matrix::solve(Model__obs_information_matrix(private$ptr,private$model_type()))[1:length(beta),1:length(beta)]
                             M <- solve(M)
                             if(se.theta){
                               SE_theta <- tryCatch(sqrt(diag(solve(Model__infomat_theta(private$ptr,private$model_type())))), error = rep(NA, ncovpar))
                             } else {
                               SE_theta <- rep(NA, ncovpar)
                             }
                           } else if(se == "robust" || se == "bwrobust"){
                             M <- Model__sandwich(private$ptr,private$model_type())
                             if(se.theta){
                               SE_theta <- tryCatch(sqrt(diag(Model__infomat_theta(private$ptr,private$model_type()))), error = rep(NA, ncovpar))
                             } else {
                               SE_theta <- rep(NA, ncovpar)
                             }
                           } else if(se == "kr" || se == "kr2" || se == "sat"){
                             ss_type <- ifelse(se=="kr",1,ifelse(se=="kr2",4,5))
                             Mout <- Model__small_sample_correction(private$ptr,ss_type,oim,private$model_type())
                             M <- Mout[[1]]
                             SE_theta <- sqrt(diag(Mout[[2]]))
                           } 
                         } else {
                           # crudely calculate the information matrix for GP approximations - this will be integrated into the main
                           # library in future versions, but can cause error/crash with the above methods
                           M <- self$information_matrix()#Model__information_matrix_crude(private$ptr,private$model_type())
                           nB <- nrow(M)
                           M <- tryCatch(solve(M), error = matrix(NA,nrow = nB,ncol=nB))
                           SE_theta <- rep(NA, ncovpar)
                         }
                         SE <- sqrt(diag(M))
                         repar_table <- self$covariance$parameter_table()
                         beta_names <- Model__beta_parameter_names(private$ptr,private$model_type())
                         theta_names <- repar_table$term
                         if(self$family[[1]]%in%c("Gamma","beta","quantile_scaled")){
                           mf_pars_names <- c(beta_names,theta_names,"sigma")
                           SE <- c(SE,rep(NA,length(theta_new)+1))
                         } else {
                           mf_pars_names <- c(beta_names,theta_names)
                           if(self$family[[1]]%in%c("gaussian")) mf_pars_names <- c(mf_pars_names,"sigma")
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
                         if(private$model_type()!=2){
                           rownames(u) <- rep(repar_table$term,repar_table$count)
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
                                     converged = converged,
                                     method = method,
                                     m = dim(u)[2],
                                     tol = tol,
                                     sim_lik = FALSE, #sim.lik.step,
                                     aic = aic,
                                     se=se,
                                     vcov = M,
                                     Rsq = c(cond = condR2,marg=margR2),
                                     logl = llvals$first,
                                     logl_theta = llvals$second,
                                     mean_form = self$mean$formula,
                                     cov_form = self$covariance$formula,
                                     family = self$family[[1]],
                                     link = self$family[[2]],
                                     re.samps = u,
                                     iter = iter,
                                     dof = dof,
                                     reml = reml,
                                     P = length(self$mean$parameters),
                                     Q = length(self$covariance$parameters),
                                     var_par_family = var_par_family,
                                     model_data = list(
                                       y = Model__y(private$ptr,private$model_type()),
                                       data = private$model_data_frame(),
                                       trials = self$trials,
                                       offset = self$mean$offset,
                                       weights = self$weights
                                     ),
                                     fn_count = fn_counter)
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
                       #'@param y Optional. A numeric vector of outcome data. If this is not provided then either the outcome must have been specified when 
                       #' initialising the Model object, or the outcome data has been updated using member function `update_y()`
                       #'@param start Optional. A numeric vector indicating starting values for the model parameters.
                       #'@param method String. Either "nloptim" for non-linear optimisation, or "nr" for Newton-Raphson (default) algorithm
                       #'@param se String. Type of standard error and/or inferential statistics to return. Options are "gls" for GLS standard errors (the default),
                       #' "robust" for robust standard errors, "kr" for original Kenward-Roger bias corrected standard errors, 
                       #' "kr2" for the improved Kenward-Roger correction, "sat" for Satterthwaite degrees of freedom correction (this is the same 
                       #' degrees of freedom correction as Kenward-Roger, but with GLS standard errors)"box" to use a modified Box correction (does not return confidence intervals),
                       #' "bw" to use GLS standard errors with a between-within correction to the degrees of freedom, "bwrobust" to use robust 
                       #' standard errors with between-within correction to the degrees of freedom. 
                       #' Note that Kenward-Roger assumes REML estimates, which are not currently provided by this function.
                       #'@param oim Logical. If TRUE use the observed information matrix, otherwise use the expected information matrix for standard error and related calculations.
                       #'@param reml Logical. Whether to use a restricted maximum likelihood correction for fitting the covariance parameters
                       #'@param max.iter Maximum number of algorithm iterations, default 20.
                       #'@param tol Maximum difference between successive iterations at which to terminate the algorithm
                       #'@param se.theta Logical. Whether to calculate the standard errors for the covariance parameters. This step is a slow part
                       #' of the calculation, so can be disabled if required in larger models. Has no effect for Kenward-Roger standard errors.
                       #'@param algo Integer. 1 = L-BFGS for beta-u and BOBYQA for theta (default), 2 = BOBYQA for both.
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
                       #'   covariance = c(0.05,0.7),
                       #'   mean = c(rep(0,5),-0.2),
                       #'   data = df,
                       #'   family = stats::binomial()
                       #' )
                       #' ysim <- des$sim_data() # simulate some data from the model
                       #' fit1 <- des$LA(y = ysim)
                       #'@md
                       LA = function(y = NULL,
                                     start,
                                     method = "nr",
                                     se = "gls",
                                     oim = FALSE,
                                     reml = TRUE,
                                     max.iter = 40,
                                     tol = 1e-4,
                                     se.theta = TRUE,
                                     algo = 2,
                                     lower.bound = NULL,
                                     upper.bound = NULL,
                                     lower.bound.theta = NULL,
                                     upper.bound.theta = NULL){
                         
                         if(is.null(y)){
                           if(!private$y_has_been_updated)stop("y not specified and not updated in Model object")
                         } else {
                           private$verify_data(y)
                           private$set_y(y)
                         }
                         Model__use_attenuation(private$ptr,private$attenuate_parameters,private$model_type())
                         if(!se %in% c("gls","kr","kr2","bw","sat","bwrobust","box"))stop("Option se not recognised")
                         if(self$family[[1]]%in%c("Gamma","beta") & (se == "kr"||se == "kr2"||se == "sat"))stop("KR standard errors are not currently available with gamma or beta families")
                         if(!method%in%c("nloptim","nr"))stop("method should be either nr or nloptim")
                         if(self$family[[1]]%in%c("quantile","quantile_scaled") & method == "nr")stop("nr with quantile currently disabled, please use nloptim with LA")
                         if(se == "box" & !(self$family[[1]]=="gaussian"&self$family[[2]]=="identity"))stop("Box only available for linear models")
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
                         Model__use_reml(private$ptr,reml,private$model_type())
                         var_par_family <- I(self$family[[1]]%in%c("gaussian","Gamma","beta","quantile_scaled"))
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
                             Model__laplace_beta_u(private$ptr,private$model_type())
                           } else {
                             if(algo %in% c(1,3)){
                               tryCatch(Model__laplace_ml_beta_u(private$ptr,2,private$model_type()),
                                        error = function(e) {
                                          if(private$trace >= 1)cat("\nL-BFGS failed, switching to BOBYQA");
                                          Model__laplace_ml_beta_u(private$ptr,0,private$model_type());
                                        })
                             } else {
                               Model__laplace_ml_beta_u(private$ptr,0,private$model_type())
                             }
                           }
                           if(algo == 3){
                             tryCatch(Model__laplace_ml_theta(private$ptr,2,private$model_type()),
                                      error = function(e) {
                                        if(private$trace >= 1)cat("\nL-BFGS failed, switching to BOBYQA");
                                        Model__laplace_ml_theta(private$ptr,0,private$model_type());
                                      })
                           } else {
                             Model__laplace_ml_theta(private$ptr,0,private$model_type())
                           }
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
                         #Model__laplace_ml_beta_theta(private$ptr,0,private$model_type())
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
                           M <- self$information_matrix()#Matrix::solve(Model__obs_information_matrix(private$ptr,private$model_type()))[1:length(beta),1:length(beta)]
                           M <- solve(M)
                           if(se.theta){
                             SE_theta <- tryCatch(sqrt(diag(solve(Model__infomat_theta(private$ptr,private$model_type())))), error = rep(NA, ncovpar))
                           } else {
                             SE_theta <- rep(NA, ncovpar)
                           }
                         } else if(se == "robust" || se == "bwrobust" ){
                           M <- Model__sandwich(private$ptr,private$model_type())
                           if(se.theta){
                             SE_theta <- tryCatch(sqrt(diag(Model__infomat_theta(private$ptr,private$model_type()))), error = rep(NA, ncovpar))
                           } else {
                             SE_theta <- rep(NA, ncovpar)
                           }
                         } else if(se == "kr" || se == "kr2" || se == "sat"){
                           krtype <- ifelse(se=="kr",1,ifelse(se=="kr2",4,5))
                           Mout <- Model__small_sample_correction(private$ptr,krtype,oim,private$model_type())
                           M <- Mout[[1]]
                           SE_theta <- sqrt(diag(Mout[[2]]))
                         }
                         SE <- sqrt(Matrix::diag(M))
                         repar_table <- self$covariance$parameter_table()
                         beta_names <- Model__beta_parameter_names(private$ptr,private$model_type())
                         theta_names <- repar_table$term
                         if(self$family[[1]]%in%c("Gamma","beta","quantile_scaled")){
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
                                     vcov = M,
                                     Rsq = c(cond = condR2,marg=margR2),
                                     mean_form = self$mean$formula,
                                     cov_form = self$covariance$formula,
                                     logl = Model__log_likelihood(private$ptr,private$model_type()),
                                     family = self$family[[1]],
                                     link = self$family[[2]],
                                     re.samps = u,
                                     iter = iter,
                                     dof = dof,
                                     reml = reml,
                                     P = length(self$mean$parameters),
                                     Q = length(self$covariance$parameters),
                                     var_par_family = var_par_family,
                                     model_data = list(
                                       y = Model__y(private$ptr,private$model_type()),
                                       data = private$model_data_frame(),
                                       trials = self$trials,
                                       offset = self$mean$offset,
                                       weights = self$weights
                                     ))
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
                       #' @param mcmc.pkg String. Either `cmdstan` for cmdstan (requires the package `cmdstanr`), `rstan` to use rstan sampler, or
                       #'`hmc` to use a cruder Hamiltonian Monte Carlo sampler. cmdstan is recommended as it has by far the best number 
                       #' of effective samples per unit time. cmdstanr will compile the MCMC programs to the library folder the first time they are run, 
                       #' so may not currently be an option for some users.
                       #' @return A matrix of samples of the random effects
                       mcmc_sample = function(mcmc.pkg = "rstan"){
                         if(!mcmc.pkg %in% c("cmdstan","rstan","hmc"))stop("mcmc.pkg must be one of cmdstan, rstan, or hmc")
                         if(!private$y_has_been_updated) stop("No y data has been added")
                         if(mcmc.pkg == "cmdstan" | mcmc.pkg == "rstan"){
                           file_type <- mcnr_family(self$family,mcmc.pkg == "cmdstan")
                           if(mcmc.pkg == "cmdstan"){
                             if(!requireNamespace("cmdstanr")){
                               stop("cmdstanr is required to use cmdstan for sampling. See https://mc-stan.org/cmdstanr/ for details on how to install.")
                             } else {
                               if(private$trace>=1)message("If this is the first time running this model, it will be compiled by cmdstan.")
                               model_file <- system.file("cmdstan",
                                                         file_type$file,
                                                         package = "glmmrBase",
                                                         mustWork = TRUE)
                               mod <- suppressMessages(cmdstanr::cmdstan_model(model_file))
                             }
                           }
                           data <- list(
                             Q = Model__Q(private$ptr,private$model_type()),
                             Xb = Model__xb(private$ptr,private$model_type()),
                             Z = Model__ZL(private$ptr,private$model_type()),
                             type=as.numeric(file_type$type)
                           )
                           if(self$family[[1]]%in%c("gaussian","beta","Gamma","quantile","quantile_scaled"))data <- append(data,list(N_cont = self$n(),
                                                                                                                                     N_int = 1,
                                                                                                                                     N_binom = 1,
                                                                                                                                     sigma = rep(self$var_par/self$weights, self$n()),
                                                                                                                                     ycont = Model__y(private$ptr,private$model_type()),
                                                                                                                                     yint = array(0,dim = 1),
                                                                                                                                     q = 0,
                                                                                                                                     n = array(0,dim = 1)))
                           if(self$family[[1]]%in%c("binomial","bernoulli","poisson"))data <- append(data,list(N_int = self$n(),
                                                                                                               N_cont = 1,
                                                                                                               N_binom = 1,
                                                                                                               sigma = array(0,dim = 1),
                                                                                                               yint = Model__y(private$ptr,private$model_type()),
                                                                                                               ycont = array(0,dim = 1),
                                                                                                               q = 0,
                                                                                                               n = array(0,dim = 1)))
                           if(self$family[[1]]=="binomial")data <- append(data,list(N_binom  = self$n(), n = self$trials))
                           if(self$family[[1]]%in%c("beta","Gamma","quantile","quantile_scaled"))data$sigma = rep(self$var_par,self$n())
                           if(self$family[[1]]%in%c("quantile","quantile_scaled"))data$q = self$family$q
                           if(private$trace <= 1){
                             if(private$trace==1)message("Starting MCMC sampling. Set self$trace(2) for detailed output")
                             if(mcmc.pkg == "cmdstan"){
                               capture.output(fit <- mod$sample(data = data,
                                                                chains = 1,
                                                                iter_warmup = self$mcmc_options$warmup,
                                                                iter_sampling = self$mcmc_options$samps,
                                                                refresh = 0),
                                              file=tempfile())
                             } else {
                               capture.output(fit <- rstan::sampling(stanmodels[[file_type$file]],
                                                                     data=data,
                                                                     chains=1,
                                                                     iter = self$mcmc_options$warmup+self$mcmc_options$samps,
                                                                     warmup = self$mcmc_options$warmup,
                                                                     refresh = 0),
                                              file=tempfile())
                             }
                           } else {
                             if(mcmc.pkg == "cmdstan"){
                               fit <- mod$sample(data = data,
                                                 chains = 1,
                                                 iter_warmup = self$mcmc_options$warmup,
                                                 iter_sampling = self$mcmc_options$samps,
                                                 refresh = 50)
                             } else {
                               fit <- rstan::sampling(stanmodels[[file_type$file]],
                                                      data=data,
                                                      chains=1,
                                                      iter = self$mcmc_options$warmup+self$mcmc_options$samps,
                                                      warmup = self$mcmc_options$warmup,
                                                      refresh = 50)
                             }
                           }
                           if(mcmc.pkg == "cmdstan"){
                             dsamps <- fit$draws("gamma",format = "matrix")
                             class(dsamps) <- "matrix"
                           } else {
                             dsamps <- rstan::extract(fit,"gamma",FALSE)
                             dsamps <- as.matrix(dsamps[,1,])
                           }
                           if(private$trace==1)message("Sampling complete, updating model")
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
                       #' @field mcmc_options There are five options for MCMC sampling that are specified in this list:
                       #' * `warmup` The number of warmup iterations. Note that if using the internal HMC
                       #' sampler, this only applies to the first iteration of the MCML algorithm, as the
                       #' values from the previous iteration are carried over.
                       #' * `samps` The number of MCMC samples drawn in the MCML algorithms. For
                       #' smaller tolerance values larger numbers of samples are required. For the internal
                       #' HMC sampler, larger numbers of samples are generally required than if using Stan since
                       #' the samples generally exhibit higher autocorrealtion, especially for more complex
                       #' covariance structures. For SAEM a small number is recommended as all samples are stored and used 
                       #' from every iteration.
                       #' * `lambda` (Only relevant for the internal HMC sampler) Value of the trajectory length of the leapfrog integrator in Hamiltonian Monte Carlo
                       #'  (equal to number of steps times the step length). Larger values result in lower correlation in samples, but
                       #'  require larger numbers of steps and so is slower. Smaller numbers are likely required for non-linear GLMMs.
                       #'  * `refresh` How frequently to print to console MCMC progress if displaying verbose output.
                       #'  * `maxsteps` (Only relevant for the internal HMC sampler) Integer. The maximum number of steps of the leapfrom integrator
                       mcmc_options = list(warmup = 100,
                                           samps = 25,
                                           chains = 1,
                                           lambda = 1,
                                           refresh = 500,
                                           maxsteps = 100,
                                           target_accept = 0.95,
                                           adapt = 50),
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
                           # if(gsub(" ","",self$mean$formula) != gsub(" ","",self$covariance$formula)){
                           #   form <- paste0(self$mean$formula,"+",self$covariance$formula)
                           # } else {
                           #   form <- gsub(" ","",self$mean$formula)
                           # }
                           form <- gsub(" ","",self$formula)
                           form <- gsub("~","",self$formula)
                           if(grepl("nngp",form)){
                             self$covariance$.__enclos_env__$private$type <- 1
                             form <- gsub("nngp_","",form)
                           } else if(grepl("hsgp",form)){
                             self$covariance$.__enclos_env__$private$type <- 2
                             form <- gsub("hsgp_","",form)
                           }
                           type <- private$model_type()
                           data <- self$covariance$data
                           if(any(!colnames(self$mean$data)%in%colnames(data))){
                             cnames <- which(!colnames(self$mean$data)%in%colnames(data))
                             data <- cbind(data,self$mean$data[,cnames,drop=FALSE])
                           }
                           #data <- private$process_data(as.formula(paste0("~",form)),data,TRUE,TRUE)
                           if(self$family[[1]]=="bernoulli" & any(self$trials>1))self$family[[1]] <- "binomial"
                           if(is.null(self$covariance$parameters)){
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
                             }
                             Model__update_beta(private$ptr,self$mean$parameters,type)
                             ncovpar <- Model__n_cov_pars(private$ptr,type)
                             self$covariance$parameters <- runif(ncovpar)
                             re <- Model__re_terms(private$ptr,type)
                             paridx <- Model__parameter_fn_index(private$ptr,type)+1
                             names(self$covariance$parameters) <- re[paridx]
                             Model__update_theta(private$ptr,self$covariance$parameters,type)
                           } else {
                             if(type == 0){
                               private$ptr <- Model__new_w_pars(form,
                                                                as.matrix(data),
                                                                colnames(data),
                                                                tolower(self$family[[1]]),
                                                                self$family[[2]],
                                                                self$mean$parameters,
                                                                self$covariance$parameters)
                             } else if(type==1){
                               nngp <- self$covariance$nngp()
                               private$ptr <- Model_nngp__new_w_pars(form,
                                                                     as.matrix(data),
                                                                     colnames(data),
                                                                     tolower(self$family[[1]]),
                                                                     self$family[[2]],
                                                                     self$mean$parameters,
                                                                     self$covariance$parameters,
                                                                     nngp[2])
                             } else if(type==2){
                               private$ptr <- Model_hsgp__new_w_pars(form,
                                                                     as.matrix(data),
                                                                     colnames(data),
                                                                     tolower(self$family[[1]]),
                                                                     self$family[[2]],
                                                                     self$mean$parameters,
                                                                     self$covariance$parameters)
                             }
                           }
                           
                           Model__set_offset(private$ptr,self$mean$offset,type)
                           Model__set_weights(private$ptr,self$weights,type)
                           Model__set_var_par(private$ptr,self$var_par,type)
                           if(self$family[[1]] == "binomial")Model__set_trials(private$ptr,self$trials,type)
                           if(self$family[[1]] %in% c("quantile","quantile_scaled")) Model__set_quantile(private$ptr,self$family$q,type)
                           # Model__update_beta(private$ptr,self$mean$parameters,type)
                           # Model__update_theta(private$ptr,self$covariance$parameters,type)
                           Model__update_u(private$ptr,matrix(rnorm(Model__Q(private$ptr,type)),ncol=1),type) # initialise random effects to random
                           Model__mcmc_set_lambda(private$ptr,self$mcmc_options$lambda,type)
                           Model__mcmc_set_max_steps(private$ptr,self$mcmc_options$maxsteps,type)
                           Model__mcmc_set_refresh(private$ptr,self$mcmc_options$refresh,type)
                           Model__mcmc_set_target_accept(private$ptr,self$mcmc_options$target_accept,type)
                           if(!private$useSparse & type == 1) Model__make_dense(private$ptr,type)
                           # set covariance pointer
                           self$covariance$.__enclos_env__$private$model_ptr <- private$ptr
                           self$covariance$.__enclos_env__$private$ptr <- NULL
                           self$covariance$.__enclos_env__$private$cov_form()
                           private$session_id <- Sys.getpid()
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
                           # if(!all(s1 %in% cnames)){
                           #   not_in <- which(!s1 %in% cnames)
                           #   stop(paste0("The following variables are not in the data: ",paste0(s1[not_in],collapse = " ")))
                           # } else {
                           #   
                           # }
                         } else {
                           return(result)
                         }
                         
                       }
                     ))

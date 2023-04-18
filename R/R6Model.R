#' A GLMM Model
#'
#' A generalised linear mixed model and a range of associated functions
#' @details
#' For the generalised linear mixed model
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
#' (see Table 2.1 in McCullagh and Nelder (1989) <ISBN:9780412317606>). The modification proposed by Zegers et al to the linear predictor to
#' improve the accuracy of approximations based on the marginal quasilikelihood is also available, see `use_attenuation()`.
#'
#' See \href{https://github.com/samuel-watson/glmmrBase/blob/master/README.md}{glmmrBase} for a
#' detailed guide on model specification.
#' @references
#' Breslow, N. E., Clayton, D. G. (1993). Approximate Inference in Generalized Linear Mixed Models.
#' Journal of the American Statistical Association<, 88(421), 9–25. <doi:10.1080/01621459.1993.10594284>
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
                       #' @field exp_condition A vector indicting the unique experimental conditions for each observation, see Details.
                       exp_condition = NULL,
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
                           if(!is.null(private$ptr)).Model__use_attenuation(private$ptr,TRUE)
                         } else {
                           private$attenuate_parameters <- FALSE
                           if(!is.null(private$ptr)).Model__use_attenuation(private$ptr,FALSE)
                         }
                         if(private$attenuate_parameters != curr_state){
                           private$genW()
                         }
                       },
                       #' @description
                       #' Return fitted values. Does not account for the random effects. For simulated values based
                       #' on resampling random effects, see `sim_data()`. To predict the values at a new location see 
                       #' `predict()`.
                       #' @param type One of either "`link`" for values on the scale of the link function, or "`response`"
                       #' for values on the scale of the response
                       #' @param X (Optional) Fixed effects matrix to generate fitted values
                       #' @param u (Optional) Random effects values at which to generate fitted values
                       #' @return A \link[Matrix]{Matrix} class object containing the predicted values
                       fitted = function(type="link", X, u){
                         if(missing(X)){
                           Xb <- self$mean$linear_predictor()
                         } else {
                           Xb <- X%*%self$mean$parameters 
                         }
                         
                         if(!missing(u)){
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
                         if(is.null(private$ptr))stop("No previous model has been estimated")
                         if(missing(offset)){
                           offs <- rep(0,nrow(newdata))
                         } else {
                           offs <- offset
                         }
                         preddata <- private$model_data(newdata)
                         out <- .Model__predict(private$ptr,as.matrix(preddata),offset,m)
                         return(out)
                       },
                       #' @description
                       #' Create a new Model object
                       #' @param formula An optional model formula containing fixed and random effect terms. If not specified, then
                       #' seaparate formulae need to be provided to the covariance and mean arguments below.
                       #' @param covariance Either a \link[glmmrBase]{Covariance} object, or an equivalent list of arguments
                       #' that can be passed to `Covariance` to create a new object. At a minimum the list must specify a formula.
                       #' If parameters are not included then they are initialised to 0.5.
                       #' @param mean Either a \link[glmmrBase]{MeanFunction} object, or an equivalent list of arguments
                       #' that can be passed to `MeanFunction` to create a new object. At a minimum the list must specify a formula.
                       #' If parameters are not included then they are initialised to 0.
                       #' @param data A data frame with the data required for the mean function and covariance objects. This argument
                       #' can be ignored if data are provided to the covariance or mean arguments either via `Covariance` and `MeanFunction`
                       #' object, or as a member of the list of arguments to both `covariance` and `mean`.
                       #' @param family A family object expressing the distribution and link function of the model, see \link[stats]{family}. This
                       #' argument is optional if the family is provided either via a `MeanFunction` or `MeanFunction`
                       #' objects, or as members of the list of arguments to `mean`. Current accepts \link[stats]{binomial},
                       #' \link[stats]{gaussian}, \link[stats]{Gamma}, \link[stats]{poisson}, and \link[glmmrBase]{Beta}.
                       #' @param var_par Scale parameter required for some distributions, including Gaussian. Default is NULL.
                       #' @param offset A vector of offset values. Optional - could be provided to the argument to mean instead.
                       #' @param verbose Logical indicating whether to provide detailed output
                       #' @return A new Model class object
                       #' @seealso \link[glmmrBase]{nelder}, \link[glmmrBase]{MeanFunction}, \link[glmmrBase]{Covariance}
                       #' @examples
                       #' #create a data frame describing a cross-sectional parallel cluster
                       #' #randomised trial
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' 
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
                       #'
                       #' des <- Model$new(
                       #'   covariance = list(
                       #'     formula = ~ (1|gr(ind)),
                       #'     parameters = c(0.25)),
                       #'   mean = list(
                       #'     formula = ~ int,
                       #'     parameters = c(1,0.5)),
                       #'   data = df,
                       #'   family = stats::poisson())
                       #'
                       #' # or as
                       #' des <- Model$new(
                       #'   formula = ~ int + (1|gr(ind)),
                       #'   covariance = list(parameters = c(0.25)),
                       #'   mean = list(parameters = c(1,0.5)),
                       #'   data = df,
                       #'   family = stats::poisson()
                       #'   )
                       #'
                       #'
                       #' #an example of a spatial grid with two time points
                       #' df <- nelder(~ (x(10)*y(10))*t(2))
                       #' spt_design <- Model$new(covariance = list( formula = ~(1|ar1(t)*fexp(x,y))),
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
                                             verbose=TRUE){

                         if(is.null(family)){
                           stop("No family specified.")
                         } else {
                           self$family <- family
                         }

                         if(!missing(formula)){
                           self$formula <- Reduce(paste,as.character(formula))
                           if(is.null(data)){
                            stop("Data must be specified with a formula")
                           } else {
                             if(missing(covariance) || is.null(covariance$parameters)){
                               self$covariance <- Covariance$new(
                                 formula = formula,
                                 data = data
                               )
                             } else {
                               self$covariance <- Covariance$new(
                                 formula = formula,
                                 data = data,
                                 parameters = covariance$parameters
                               )
                             }
                             if(missing(mean) || is.null(mean$parameters)){
                               self$mean <- MeanFunction$new(
                                 formula = formula,
                                 data = data
                               )
                             } else {
                               self$mean <- MeanFunction$new(
                                 formula = formula,
                                 data = data,
                                 parameters = mean$parameters
                               )
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
                             if(!is.null(covariance$parameters))self$covariance$parameters <- covariance$parameters
                             if(!is.null(covariance$eff_range))self$covariance$eff_range <- covariance$eff_range
                             if(is.null(covariance$data)){
                               self$covariance$data <- data
                             } else {
                               self$covariance$data <- covariance$data
                             }
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
                             if(!is.null(mean$parameters))self$mean$parameters <- mean$parameters
                           }
                           if(is.null(offset)){
                             self$mean$offset <- rep(0,nrow(self$mean$data))
                           } else {
                             self$mean$offset <- offset
                           }
                           self$mean$check(verbose = verbose)
                           self$covariance$check(verbose=verbose)
                         }

                         if(!is.null(var_par)){
                           self$var_par <- var_par
                         } else {
                           self$var_par <- 1
                         }

                         private$hash <- private$hash_do()
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
                         cat("\n   \U2223     \U2BA1 Terms: ",.re_names(self$covariance$formula))
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
                       #' @return The function updates the object and nothing is returned
                       #' @examples
                       #' #generate a stepped wedge design and remove the first sequence
                       #' des <- stepped_wedge(8,10,icc=0.05)
                       #' ids_to_keep <- which(des$mean$data$J!=1)
                       #' des$subset_rows(ids_to_keep)
                       subset_rows = function(index){
                         self$mean$subset_rows(index)
                         self$covariance$subset(index)
                         self$check(verbose=FALSE)
                       },
                       #' @description
                       #' Subsets the columns of the design
                       #'
                       #' Removes the specified columns from the linked mean function object's X matrix.
                       #' @param index Integer or vector of integers specifying the indexes of the columns to keep
                       #' @return The function updates the object and nothing is returned
                       #' @examples
                       #' #generate a stepped wedge design and remove first and last time periods
                       #' des <- stepped_wedge(8,10,icc=0.05)
                       #' des$subset_cols(c(2:8,10))
                       subset_cols = function(index){
                         self$mean$subset_cols(index)
                         self$check(verbose=FALSE)
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
                       #' des <- Model$new(
                       #'   covariance = list(
                       #'     formula = ~ (1|gr(cl)*ar1(t)),
                       #'     parameters = c(0.25,0.8)),
                       #'   mean = list(
                       #'     formula = ~ factor(t) + int - 1,
                       #'     parameters = c(rep(0,5),0.6)),
                       #'   data = df,
                       #'   family = stats::binomial()
                       #' )
                       #' ysim <- des$sim_data()
                       sim_data = function(type = "y"){
                         re <- self$covariance$simulate_re()
                         mu <- drop(as.matrix(self$mean$X)%*%self$mean$parameters) +
                           drop(as.matrix(self$covariance$Z)%*%re) +
                           self$mean$offset

                         f <- self$family
                         if(f[1]=="poisson"){
                           if(f[2]=="log"){
                             y <- rpois(self$n(),exp(mu))
                           }
                           if(f[2]=="identity"){
                             y <- rpois(self$n(),mu)
                           }
                         }

                         if(f[1]=="binomial"){
                           if(f[2]=="logit"){
                             y <- rbinom(self$n(),1,exp(mu)/(1+exp(mu)))
                           }
                           if(f[2]=="log"){
                             y <- rbinom(self$n(),1,exp(mu))
                           }
                           if(f[2]=="identity"){
                             y <- rbinom(self$n(),1,mu)
                           }
                           if(f[2]=="probit"){
                             y <- rbinom(self$n(),1,pnorm(mu))
                           }
                         }

                         if(f[1]=="gaussian"){
                           if(f[2]=="identity"){
                             if(is.null(self$var_par))stop("For gaussian(link='identity') provide var_par")
                             y <- rnorm(self$n(),mu,self$var_par)
                           }
                           if(f[2]=="log"){
                             if(is.null(self$var_par))stop("For gaussian(link='log') provide var_par")
                             #CHECK THIS IS RIGHT
                             y <- rnorm(self$n(),exp(mu),self$var_par)
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
                       #'@description
                       #'Checks for any changes in linked objects and updates.
                       #'
                       #' Checks for any changes in any object and updates all linked objects if
                       #' any are detected. Generally called automatically.
                       #'@param verbose Logical indicating whether to report if any updates are made, defaults to TRUE
                       #'@return Linked objects are updated by nothing is returned
                       #'@examples
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' des <- Model$new(
                       #'   covariance = list(
                       #'     formula = ~ (1|gr(cl)*ar1(t)),
                       #'     parameters = c(0.25,0.8)),
                       #'   mean = list(
                       #'     formula = ~ factor(t) + int - 1,
                       #'     parameters = c(rep(0,5),0.6)),
                       #'   data = df,
                       #'   family = stats::binomial()
                       #' )
                       #' des$check() #does nothing
                       #' des$covariance$parameters <- c(0.1,0.9)
                       #' des$check() #updates
                       #' des$mean$parameters <- c(rnorm(5),0.1)
                       #' des$check() #updates
                       check = function(verbose=TRUE){
                         self$covariance$check(verbose=verbose)
                         self$mean$check(verbose = verbose)
                         if(private$hash != private$hash_do()){
                           private$generate()
                         }
                       },
                       #' @description
                       #' Updates the parameters of the mean function and/or the covariance function
                       #'
                       #' @details
                       #' Using `update_parameters()` is the preferred way of updating the parameters of the
                       #' mean or covariance objects as opposed to direct assignment, e.g. `self$covariance$parameters <- c(...)`.
                       #' The function calls check functions to automatically update linked matrices with the new parameters.
                       #' If using direct assignment, call `self$check()` afterwards.
                       #'
                       #' @param mean.pars (Optional) Vector of new mean function parameters
                       #' @param cov.pars (Optional) Vector of new covariance function(s) parameters
                       #' @param var.par (Optional) A scalar value for var_par
                       #' @param verbose Logical indicating whether to provide more detailed feedback
                       #' @examples
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' des <- Model$new(
                       #'   covariance = list(
                       #'     formula = ~ (1|gr(cl)*ar1(t))),
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
                             .Model__update_beta(private$ptr,mean.pars)
                           }
                         }
                         if(!is.null(cov.pars)){
                           self$covariance$update_parameters(cov.pars)
                           if(!is.null(private$ptr)){
                             .Model__update_theta(private$ptr,cov.pars)
                           }
                         }
                         if(!is.null(var.par)){
                           self$var_par <- var.par
                           if(!is.null(private$ptr)){
                             .Model__set_var_par(private$ptr,var.par)
                           }
                         }
                         
                         self$check(FALSE)
                       },
                       #' @description
                       #' Generates the information matrix of the GLS estimator
                       #' @return A PxP matrix
                       information_matrix = function(){
                         if(is.null(private$ptr)){
                           private$update_ptr(rep(0,nrow(self$mean$data)))
                         }
                         return(.Model__information_matrix(private$ptr))
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
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' des <- Model$new(
                       #'   covariance = list(
                       #'     formula = ~ (1|gr(cl)) + (1|gr(cl,t)),
                       #'     parameters = c(0.25,0.1)),
                       #'   mean = list(
                       #'     formula = ~ factor(t) + int - 1,
                       #'     parameters = c(rep(0,5),0.6)),
                       #'   data = df,
                       #'   family = stats::gaussian(),
                       #'   var_par = 1
                       #' )
                       #' des$power() #power of 0.90 for the int parameter
                       power = function(alpha=0.05,two.sided=TRUE,alternative = "pos"){
                         self$check(verbose=FALSE)
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

                         res <- data.frame(Parameter = colnames(self$mean$X),
                                           Value = self$mean$parameters,
                                           SE = v0,
                                           Power = pwr)
                         return(res)
                       },
                       #' @description
                       #' Returns the diagonal of the matrix W used to calculate the covariance matrix approximation
                       #' @return A vector with values of the glm iterated weights
                       w_matrix = function(){
                         private$genW()
                         return(private$W)
                       },
                       #' @description
                       #' Returns the derivative of the link function with respect to the linear preditor
                       #' @return A vector
                       dh_deta = function(){
                         Q = .dlinkdeta(self$fitted(),self$family[[2]])
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
                           private$update_ptr(rep(0,nrow(self$mean$data)))
                         }
                         return(.Model__Sigma(private$ptr,inverse))
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
                       #' Options for the MCMC sampler are set by changing the values in `self$mcmc_options`
                       #' 
                       #'@param y A numeric vector of outcome data
                       #'@param method The MCML algorithm to use, either `mcem` or `mcnr`, see Details. Default is `mcem`.
                       #'@param sim.lik.step Logical. Either TRUE (conduct a simulated likelihood step at the end of the algorithm), or FALSE (does
                       #'not do this step), defaults to FALSE.
                       #'@param verbose Logical indicating whether to provide detailed output, defaults to TRUE.
                       #'@param tol Numeric value, tolerance of the MCML algorithm, the maximum difference in parameter estimates
                       #'between iterations at which to stop the algorithm.
                       #'@param max.iter Integer. The maximum number of iterations of the MCML algorithm.
                       #'@param sparse Logical indicating whether to use sparse matrix methods
                       #'@param usestan Logical whether to use Stan (through the package `cmdstanr`) for the MCMC sampling. If FALSE then
                       #'the internal Hamiltonian Monte Carlo sampler will be used instead. We recommend Stan over the internal sampler as
                       #'it generally produces a larger number of effective samplers per unit time, especially for more complex
                       #'covariance functions.
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
                       #'   formula= ~ factor(t) + int - 1 +(1|gr(cl)*ar1(t)),
                       #'   covariance = list(parameters = c(0.25,0.7)),
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
                                       verbose=TRUE,
                                       tol = 1e-2,
                                       max.iter = 30,
                                       sparse = FALSE,
                                       usestan = TRUE){
                         private$verify_data(y)
                         private$update_ptr(y)
                         .Model__use_attenuation(private$ptr,private$attenuate_parameters)
                         ### DO SOME BASIC CHECKS ON Y TO MAKE SURE IT DOESN'T CAUSE ERROR!
                         
                         if(!usestan){
                           .Model__mcmc_set_lambda(private$ptr,self$mcmc_options$lambda)
                           .Model__mcmc_set_max_steps(private$ptr,self$mcmc_options$maxsteps)
                           .Model__mcmc_set_refresh(private$ptr,self$mcmc_options$refresh)
                         }
                         
                         trace <- ifelse(verbose,2,0)
                         beta <- self$mean$parameters
                         theta <- self$covariance$parameters
                         var_par <- self$var_par
                         
                         var_par_family <- I(self$family[[1]]%in%c("gaussian","Gamma","beta"))
                         
                         all_pars <- c(beta,theta)
                         if(var_par_family)all_pars <- c(all_pars,var_par)
                         all_pars_new <- rep(1,length(all_pars))
                         var_par_new <- var_par
                         
                         if(verbose)message(paste0("using method: ",method))
                         if(verbose)cat("\nStart: ",all_pars,"\n")
                         
                         niter <- self$mcmc_options$samps
                         invfunc <- self$family$linkinv
                         L <- Matrix::Matrix(.Model__L(private$ptr))
                         
                         #parse family
                         file_type <- mcnr_family(self$family)
                         
                         ## set up sampler
                         if(usestan){
                           if(!requireNamespace("cmdstanr")){
                             stop("cmdstanr is required to use Stan for sampling. See https://mc-stan.org/cmdstanr/ for details on how to install.\n
                                    Set option usestan=FALSE to use the in-built MCMC sampler.")
                           } else {
                             if(verbose)message("If this is the first time running this model, it will be compiled by cmdstan.")
                             model_file <- system.file("stan",
                                                       file_type$file,
                                                       package = "glmmrBase",
                                                       mustWork = TRUE)
                             mod <- suppressMessages(cmdstanr::cmdstan_model(model_file))
                           }
                         }
                         
                         data <- list(
                           N = self$n(),
                           Q = .Model__Q(private$ptr),
                           Xb = .Model__xb(private$ptr),
                           Z = .Model__ZL(private$ptr),
                           y = y,
                           sigma = var_par,
                           type=as.numeric(file_type$type)
                         )
                         iter <- 0
                         
                         while(any(abs(all_pars-all_pars_new)>tol)&iter < max.iter){
                           all_pars <- all_pars_new
                           iter <- iter + 1
                           if(verbose)cat("\nIter: ",iter,"\n",Reduce(paste0,rep("-",40)))
                           if(trace==2)t1 <- Sys.time()
                           
                           if(usestan){
                             data$Xb <-  .Model__xb(private$ptr)
                             data$Z <- .Model__ZL(private$ptr)
                             data$sigma <- var_par_new
                             
                             capture.output(fit <- mod$sample(data = data,
                                                              chains = 1,
                                                              iter_warmup = self$mcmc_options$warmup,
                                                              iter_sampling = self$mcmc_options$samps,
                                                              refresh = 0),
                                            file=tempfile())
                             
                             dsamps <- fit$draws("gamma",format = "matrix")
                             class(dsamps) <- "matrix"
                             #dsamps <- Matrix::Matrix(L %*% Matrix::t(dsamps)) #check this
                             .Model__update_u(private$ptr,as.matrix(t(dsamps)))
                           } else {
                             .Model__mcmc_sample(private$ptr,
                                                 self$mcmc_options$warmup,
                                                 self$mcmc_options$samps,
                                                 self$mcmc_options$adapt)
                           }
                           if(trace==2)t2 <- Sys.time()
                           if(trace==2)cat("\nMCMC sampling took: ",t2-t1,"s")
                           ## ADD IN RSTAN FUNCTIONALITY ONCE PARALLEL METHODS AVAILABLE IN RSTAN
                           
                           if(method=="mcnr"){
                             .Model__ml_beta(private$ptr)
                           } else {
                             .Model__nr_beta(private$ptr)
                           }
                           .Model__ml_theta(private$ptr)
                           #L <- Matrix::Matrix(.Model__L(private$ptr))
                           
                           beta_new <- .Model__get_beta(private$ptr)
                           theta_new <- .Model__get_theta(private$ptr)
                           var_par_new <- .Model__get_var_par(private$ptr)
                           all_pars_new <- c(beta_new,theta_new)
                           if(var_par_family)all_pars_new <- c(all_pars_new,var_par_new)
                           
                           if(trace==2)t3 <- Sys.time()
                           if(trace==2)cat("\nModel fitting took: ",t3-t2,"s")
                           if(verbose){
                             #cat("\ntheta:",theta[all_pars])
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
                           if(verbose)cat("\n\n")
                           if(verbose)message("Optimising simulated likelihood")
                           .Model__ml_all(private$ptr)
                           
                           beta_new <- .Model__get_beta(private$ptr)
                           theta_new <- .Model__get_theta(private$ptr)
                           var_par_new <- .Model__get_var_par(private$ptr)
                         }
                         
                         self$update_parameters(mean.pars = beta_new,
                                                cov.pars = theta_new)
                         
                         self$var_par <- var_par_new
                         u <- .Model__u(private$ptr)
                         invM <- Matrix::solve(self$information_matrix())
                         SE <- sqrt(Matrix::diag(invM))
                         
                         if(verbose)cat("\n\nCalculating standard errors...\n")
                         
                         repar_table <- self$covariance$parameter_table()
                         
                         
                         if(var_par_family){
                           mf_pars_names <- c(colnames(self$mean$X),repar_table$term,"sigma")
                           SE <- c(SE,rep(NA,length(theta_new)+1))
                         } else {
                           mf_pars_names <- c(colnames(self$mean$X),repar_table$term)
                           SE <- c(SE,rep(NA,length(theta_new)))
                         }
                         
                         res <- data.frame(par = c(mf_pars_names,paste0("d",1:nrow(u))),
                                           est = c(all_pars_new,rowMeans(u)),
                                           SE=c(SE,rep(NA,nrow(u))))
                         
                         res$lower <- res$est - qnorm(1-0.05/2)*res$SE
                         res$upper <- res$est + qnorm(1-0.05/2)*res$SE
                         
                         repar_table <- repar_table[!duplicated(repar_table$id),]
                         
                         rownames(u) <- rep(repar_table$term,repar_table$count)
                         
                         aic <- .Model__aic(private$ptr)
                         
                         xb <- .Model__xb(private$ptr)
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
                                     Rsq = c(cond = condR2,marg=margR2),
                                     mean_form = self$mean$formula,
                                     cov_form = self$covariance$formula,
                                     family = self$family[[1]],
                                     link = self$family[[2]],
                                     re.samps = u,
                                     iter = iter,
                                     P = length(self$mean$parameters),
                                     Q = length(self$covariance$parameters),
                                     var_par_family = var_par_family)
                         
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
                       #'class, see \link[glmmrBase]{Model}.
                       #'
                       #'@param y A numeric vector of outcome data
                       #'@param start Optional. A numeric vector indicating starting values for the model parameters.
                       #'@param method String. Either "nloptim" for non-linear optimisation, or "nr" for Newton-Raphson (default) algorithm
                       #'@param verbose logical indicating whether to provide detailed algorithm feedback (default is TRUE).
                       #'@param max.iter Maximum number of algorithm iterations, default 20.
                       #'@param tol Maximum difference between successive iterations at which to terminate the algorithm
                       #'@return A `mcml` object
                       #' @seealso \link[glmmrBase]{Model}, \link[glmmrBase]{Covariance}, \link[glmmrBase]{MeanFunction}
                       #'@examples
                       #' #create example data with six clusters, five time periods, and five people per cluster-period
                       #' df <- nelder(~(cl(6)*t(5)) > ind(5))
                       #' # parallel trial design intervention indicator
                       #' df$int <- 0
                       #' df[df$cl > 3, 'int'] <- 1 
                       #' # specify parameter values in the call for the data simulation below
                       #' des <- Model$new(
                       #'   formula = ~ factor(t) + int - 1 + (1|gr(cl)*ar1(t)),
                       #'   covariance = list( parameters = c(0.25,0.7)),
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
                                     verbose = FALSE,
                                     max.iter = 40,
                                     tol = 1e-2){
                         private$verify_data(y)
                         private$update_ptr(y)
                         .Model__use_attenuation(private$ptr,private$attenuate_parameters)
                         # initialise u to random values as algorithm can fail if all zeros
                         # .Model__update_u(private$ptr,matrix(rnorm(ncol(self$covariance$Z)),nrow=ncol(self$covariance$Z),ncol=1))
                         if(!method%in%c("nloptim","nr"))stop("method should be either nr or nloptim")
                         trace <- ifelse(verbose,1,0)
                         
                         var_par_family <- I(self$family[[1]]%in%c("gaussian","Gamma","beta"))
                         
                         trace <- ifelse(verbose,2,0)
                         beta <- self$mean$parameters
                         theta <- self$covariance$parameters
                         var_par <- self$var_par
                         all_pars <- c(beta,theta)
                         if(var_par_family)all_pars <- c(all_pars,var_par)
                         all_pars_new <- rep(1,length(all_pars))
                         iter <- 0
                         
                         while(any(abs(all_pars-all_pars_new)>tol)&iter < max.iter){
                           all_pars <- all_pars_new
                           
                           iter <- iter + 1
                           if(verbose)cat("\nIter: ",iter,"\n",Reduce(paste0,rep("-",40)))
                           
                           if(method=="nr"){
                             .Model__laplace_nr_beta_u(private$ptr)
                           } else {
                             .Model__laplace_ml_beta_u(private$ptr)
                           }
                           .Model__laplace_ml_theta(private$ptr)
                           
                           beta_new <- .Model__get_beta(private$ptr)
                           theta_new <- .Model__get_theta(private$ptr)
                           var_par_new <- .Model__get_var_par(private$ptr)
                           all_pars_new <- c(beta_new,theta_new)
                           if(var_par_family)all_pars_new <- c(all_pars_new,var_par)
                           
                           if(verbose){
                             cat("\nBeta: ", beta_new)
                             cat("\nTheta: ", theta_new)
                             if(var_par_family)cat("\nSigma: ",var_par_new)
                             cat("\nMax. diff: ", round(max(abs(all_pars-all_pars_new)),5))
                             cat("\n",Reduce(paste0,rep("-",40)))
                           }
                         }
                         not_conv <- iter > max.iter|any(abs(all_pars-all_pars_new)>tol)
                         if(not_conv)message(paste0("algorithm not converged. Max. difference between iterations :",round(max(abs(all_pars-all_pars_new)),4)))
                         
                         .Model__laplace_ml_beta_theta(private$ptr)
                         beta_new <- .Model__get_beta(private$ptr)
                         theta_new <- .Model__get_theta(private$ptr)
                         var_par_new <- .Model__get_var_par(private$ptr)
                         all_pars_new <- c(beta_new,theta_new)
                         if(var_par_family)all_pars_new <- c(all_pars_new,var_par)
                         
                         
                         self$update_parameters(mean.pars = beta_new,
                                                cov.pars = theta_new)
                         
                         self$var_par <- var_par_new
                         u <- .Model__u(private$ptr,TRUE)
                         invM <- Matrix::solve(self$information_matrix())
                         SE <- sqrt(Matrix::diag(invM))
                         
                         if(verbose)cat("\n\nCalculating standard errors...\n")
                         
                         repar_table <- self$covariance$parameter_table()
                         
                         
                         if(var_par_family){
                           mf_pars_names <- c(colnames(self$mean$X),repar_table$term,"sigma")
                           SE <- c(SE,rep(NA,length(theta_new)+1))
                         } else {
                           mf_pars_names <- c(colnames(self$mean$X),repar_table$term)
                           SE <- c(SE,rep(NA,length(theta_new)))
                         }
                         
                         res <- data.frame(par = c(mf_pars_names,paste0("d",1:nrow(u))),
                                           est = c(all_pars_new,drop(u)),
                                           SE=c(SE,rep(NA,nrow(u))))
                         
                         res$lower <- res$est - qnorm(1-0.05/2)*res$SE
                         res$upper <- res$est + qnorm(1-0.05/2)*res$SE
                         
                         repar_table <- repar_table[!duplicated(repar_table$id),]
                         
                         rownames(u) <- rep(repar_table$term,repar_table$count)
                         
                         aic <- .Model__aic(private$ptr)
                         
                         xb <- .Model__xb(private$ptr)
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
                                     Rsq = c(cond = condR2,marg=margR2),
                                     mean_form = self$mean$formula,
                                     cov_form = self$covariance$formula,
                                     family = self$family[[1]],
                                     link = self$family[[2]],
                                     re.samps = u,
                                     iter = iter,
                                     P = length(self$mean$parameters),
                                     Q = length(self$covariance$parameters),
                                     var_par_family = var_par_family)
                         
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
                           if(sparse){
                             .Model__make_sparse(private$ptr)
                           } else {
                             .Model__make_dense(private$ptr)
                           }
                         } 
                         private$useSparse = sparse
                         self$covariance$sparse(sparse)
                       },
                       #' @description 
                       #' Generate an MCMC sample of the random effects
                       #' @param y Numeric vector of outcome data
                       #' @param usestan Logical whether to use Stan (through the package `cmdstanr`) for the MCMC sampling. If FALSE then
                       #'the internal Hamiltonian Monte Carlo sampler will be used instead. We recommend Stan over the internal sampler as
                       #'it generally produces a larger number of effective samplers per unit time, especially for more complex
                       #'covariance functions.
                       #' @param verbose Logical indicating whether to provide detailed output to the console
                       #' @return A matrix of samples of the random effects
                       mcmc_sample = function(y,usestan = TRUE,verbose=TRUE){
                         private$verify_data(y)
                         private$update_ptr(y)
                         if(usestan){
                           file_type <- mcnr_family(self$family)
                           if(!requireNamespace("cmdstanr")){
                             stop("cmdstanr is required to use Stan for sampling. See https://mc-stan.org/cmdstanr/ for details on how to install.\n
                                    Set option usestan=FALSE to use the in-built MCMC sampler.")
                           } else {
                             if(verbose)message("If this is the first time running this model, it will be compiled by cmdstan.")
                             model_file <- system.file("stan",
                                                       file_type$file,
                                                       package = "glmmrBase",
                                                       mustWork = TRUE)
                             mod <- suppressMessages(cmdstanr::cmdstan_model(model_file))
                           }
                           data <- list(
                             N = self$n(),
                             Q = .Model__Q(private$ptr),
                             Xb = .Model__xb(private$ptr),
                             Z = .Model__ZL(private$ptr),
                             y = y,
                             sigma = self$var_par,
                             type=as.numeric(file_type$type)
                           )
                           
                           if(verbose){
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
                           dsamps <- Matrix::Matrix(.Model__L(private$ptr) %*% Matrix::t(dsamps)) #check this
                           
                         } else {
                           
                           if(verbose).Model__set_trace(private$ptr,2)
                           .Model__use_attenuation(private$ptr,private$attenuate_parameters)
                           .Model__mcmc_set_lambda(private$ptr,self$mcmc_options$lambda)
                           .Model__mcmc_set_max_steps(private$ptr,self$mcmc_options$maxsteps)
                           .Model__mcmc_set_refresh(private$ptr,self$mcmc_options$refresh)
                           .Model__mcmc_sample(private$ptr,self$mcmc_options$warmup,self$mcmc_options$samps,self$mcmc_options$adapt)
                           dsamps <- .Model__u(private$ptr,TRUE)
                         }
                         return(dsamps)
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
                       log_gradient = function(y,u,beta=FALSE){
                         private$verify_data(y)
                         private$update_ptr(y)
                         grad <- .Model__log_gradient(private$ptr,u,beta)
                         return(grad)
                       },
                       #' @description 
                       #' Returns the sample of random effects from the last model fit
                       #' @param scaled Logical indicating whether to return samples on the N(0,I) scale (`scaled=FALSE`) or
                       #' N(0,D) scale (`scaled=TRUE`)
                       #' @return A matrix of random effect samples
                       u = function(scaled = TRUE){
                         if(is.null(private$ptr))stop("No previous model fit")
                         return(.Model__u(private$ptr,scaled))
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
                         private$update_ptr(y)
                         if(!missing(u)).Model__update_u(private$ptr,u)
                         return(.Model__log_likelihood(private$ptr))
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
                                           adapt = 50)
                     ),
                     private = list(
                       W = NULL,
                       Xb = NULL,
                       useSparse = TRUE,
                       logit = function(x){
                         exp(x)/(1+exp(x))
                       },
                       generate = function(){
                         private$genW()
                       },
                       genW = function(){
                         # assume random effects value is at zero
                         if(!self$family[[1]]%in%c("poisson","binomial","gaussian","Gamma","beta"))stop("family must be one of Poisson, Binomial, Gaussian, Gamma, Beta")
                         xb <- self$mean$linear_predictor()
                         if(private$attenuate_parameters){
                           xb <- .attenuate_xb(xb = xb,
                                              Z = as.matrix(self$covariance$Z),
                                              D = as.matrix(self$covariance$D),
                                              link = self$family[[2]])
                         }
                         wdiag <- .gen_dhdmu(xb = xb,
                                            family=self$family[[1]],
                                            link = self$family[[2]])

                         if(self$family[[1]] == "gaussian"){
                           wdiag <- self$var_par * self$var_par * wdiag
                         } else if(self$family[[1]] == "Gamma"){
                           wdiag <- wdiag/self$var_par
                         } else if(self$family[[1]] == "beta"){
                           wdiag <- wdiag*(1+self$var_par)
                         } else if(self$family[[1]] == "binomial"){
                           wdiag <- wdiag/self$var_par
                         }

                         W <- diag(drop(wdiag))
                         private$W <- Matrix::Matrix(W)
                       },
                       genS = function(update=TRUE){
                         S = .gen_sigma_approx(xb=matrix(self$mean$X%*%self$mean$parameters,ncol=1),
                                              Z = as.matrix(self$covariance$Z),
                                              D = as.matrix(self$covariance$D),
                                              family=self$family[[1]],
                                              link = self$family[[2]],
                                              var_par = self$var_par,
                                              attenuate = private$attenuate_parameters)
                         return(S)
                       },
                       attenuate_parameters = FALSE,
                       hash = NULL,
                       hash_do = function(){
                         digest::digest(c(self$covariance$.__enclos_env__$private$hash,
                                          self$mean$.__enclos_env__$private$hash))
                       },
                       ptr = NULL,
                       update_ptr = function(y){
                         if(length(y)!=nrow(self$mean$X))stop("y not correct size")
                         if(gsub(" ","",self$mean$formula) != gsub(" ","",self$covariance$formula)){
                           form <- paste0(self$mean$formula,"+",self$covariance$formula)
                         } else {
                           form <- gsub(" ","",self$mean$formula)
                         }
                         data <- self$covariance$data
                         if(any(!colnames(self$mean$data)%in%colnames(data))){
                           cnames <- which(!colnames(self$mean$data)%in%colnames(data))
                           data <- cbind(data,self$mean$data[,cnames])
                         }
                         private$ptr <- .Model__new(y,form,as.matrix(data),colnames(data),
                                                    self$family[[1]],self$family[[2]])
                         .Model__set_offset(private$ptr,self$mean$offset)
                         .Model__update_beta(private$ptr,self$mean$parameters)
                         .Model__update_theta(private$ptr,self$covariance$parameters)
                         .Model__set_var_par(private$ptr,self$var_par)
                         .Model__mcmc_set_lambda(private$ptr,self$mcmc_options$lambda)
                         .Model__mcmc_set_max_steps(private$ptr,self$mcmc_options$maxsteps)
                         .Model__mcmc_set_refresh(private$ptr,self$mcmc_options$refresh)
                         .Model__mcmc_set_target_accept(private$ptr,self$mcmc_options$target_accept)
                         if(!private$useSparse){
                           .Model__make_dense(private$ptr)
                         } 
                       },
                       verify_data = function(y){
                         if(self$family[[1]]=="binomial"){
                           if(!all(y==0 | y==1))stop("y must be 0 or 1")
                         } else if(self$family[[1]]=="poisson"){
                           if(any(y <0) || any(y%%1 != 0))stop("y must be integer >= 0")
                         } else if(self$family[[1]]=="beta"){
                           if(any(y<0 || y>1))stop("y must be between 0 and 1")
                         } else if(self$family[[1]]=="Gamma") {
                           if(any(y<=0))stop("y must be positive")
                         } else if(self$family[[1]]=="gaussian" & self$family[[2]]=="log"){
                           if(any(y<=0))stop("y must be positive")
                         }
                       },
                       model_data = function(newdata){
                         cnames1 <- colnames(self$covariance$data)
                         cnames2 <- colnames(self$mean$data)
                         cnames2 <- cnames2[!cnames2%in%cnames1]
                         cnames <- c(cnames1,cnames2)
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
                           newdat = newdata
                         }
                         return(newdat)
                       }
                     ))


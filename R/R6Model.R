#' A GLMM Model 
#' 
#' An R6 class representing a GLMM model
#' @details
#' For the generalised linear mixed model 
#' 
#' \deqn{Y \sim F(\mu,\sigma)}
#' \deqn{\mu = h^-1(X\beta + Zu)}
#' \deqn{u \sim MVN(0,D)}
#' 
#' where h is the link function. A Model in comprised of a \link[glmmrBase]{MeanFunction} object, which defines the family F, 
#' link function h, and fixed effects design matrix X, and a \link[glmmrBase]{Covariance} object, which defines Z and D. The class provides
#' methods for analysis and simulation with these models.
#' 
#' This class provides methods for generating the matrices described above and data simulation, and serves as a base for extended functionality 
#' in related packages.
#' 
#' Many calculations use the covariance matrix of the observations, such as the information matrix, which is used in power calculations and 
#' other functions. For non-Gaussian models, the class uses the approximation proposed by Breslow and Clayton (1993) based on the 
#' marginal quasilikelihood:
#' 
#' \deqn{\Sigma = W^{-1} + ZDZ^T}
#' 
#' where _W_ is a diagonal matrix with the GLM iterated weights for each observation equal
#' to, for individual _i_ \eqn{\left( \frac{(\partial h^{-1}(\eta_i))}{\partial \eta_i}\right) ^2 Var(y|u)} 
#' (see Table 2.1 in McCullagh and Nelder (1989) <ISBN:9780412317606>). For very large designs, this can be disabled as
#' the memory requirements can be prohibitive (use option `skip_sigma`). 
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
                       #' @field mean_function A \link[glmmrBase]{MeanFunction} object, defining the mean function for the model, including the data and covariate design matrix X.
                       mean_function = NULL,
                       #' @field family One of the family function used in R's glm functions. See \link[stats]{family} for details
                       family = NULL,
                       #' @field exp_condition A vector indicting the unique experimental conditions for each observation, see Details.
                       exp_condition = NULL,
                       #' @field Sigma The overall covariance matrix for the observations. Calculated and updated automatically as \eqn{W^{-1} + ZDZ^T} where W is an n x n 
                       #' diagonal matrix with elements on the diagonal equal to the GLM iterated weights. See Details.
                       Sigma = NULL,
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
                         } else {
                           private$attenuate_parameters <- FALSE
                         }
                         if(private$attenuate_parameters != curr_state){
                           private$genW()
                           private$genS()
                         }
                       },
                       #' @description 
                       #' Return predicted values based on the currently stored parameter values in `mean_function`
                       #' @param type One of either "`link`" for values on the scale of the link function, or "`response`" 
                       #' for values on the scale of the response
                       #' @return A \link[Matrix]{Matrix} class object containing the predicted values
                       fitted = function(type="link"){
                         Xb <- Matrix::drop(self$mean_function$X %*% self$mean_function$parameters)
                         if(type=="response"){
                           Xb <- self$family$linkinv(Xb)
                         }
                         return(Xb)
                       },
                       #' @description 
                       #' Create a new Model object
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
                       #' @param verbose Logical indicating whether to provide detailed output
                       #' @param skip.sigma Logical indicating whether to skip the creating of the covariance matrix Sigma. For 
                       #' very large designs with thousands of observations or more, the covariance matrix will be too big to 
                       #' fit in memory, so this option will prevent sigma being created.
                       #' @return A new Model class object
                       #' @seealso \link[glmmrBase]{nelder}, \link[glmmrBase]{MeanFunction}, \link[glmmrBase]{Covariance}
                       #' @examples 
                       #' #create a data frame describing a cross-sectional parallel cluster
                       #' #randomised trial
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' 
                       #' mf1 <- MeanFunction$new(
                       #'   formula = ~ factor(t) + int - 1,
                       #'   data=df
                       #' )
                       #' cov1 <- Covariance$new(
                       #'   data = df,
                       #'   formula = ~ (1|gr(cl)) + (1|gr(cl*t))
                       #' )
                       #' des <- Model$new(
                       #'   covariance = cov1,
                       #'   mean = mf1,
                       #'   family = stats::gaussian(),
                       #'   var_par = 1
                       #' )
                       #' 
                       #' #alternatively we can pass the data directly to Model
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
                       #' #an example of a spatial grid with two time points
                       #' df <- nelder(~ (x(10)*y(10))*t(2))
                       #' spt_design <- Model$new(covariance = list( formula = ~(1|fexp(x,y)*ar1(t))),
                       #'                          mean = list(formula = ~ 1),
                       #'                          data = df,
                       #'                          family = stats::gaussian()) 
                       initialize = function(covariance,
                                             mean,
                                             data = NULL,
                                             family = NULL,
                                             var_par = NULL,
                                             verbose=TRUE,
                                             skip.sigma = FALSE){
                         
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
                         
                         if(is.null(family)){
                           stop("No family specified.")
                         } else {
                           self$family <- family
                         }
                         
                         if(is(mean,"R6")){
                           if(is(mean,"MeanFunction")){
                             self$mean_function <- mean
                             if(is.null(mean$data)){
                               if(is.null(data)){
                                 stop("No data specified in MeanFunction object or call to function.")
                               } else {
                                 self$mean_function$data <- data
                               }
                             }
                             
                           } else {
                             stop("mean should be MeanFunction class or list of appropriate arguments")
                           }
                         } else if(is(mean,"list")){
                           if(is.null(mean$formula))stop("A formula must be specified for the mean function.")
                           if(is.null(mean$data) & is.null(data))stop("No data specified in mean list or call to function.")
                           self$mean_function <- MeanFunction$new(
                             formula = mean$formula
                           )
                           
                           if(!is.null(mean$parameters))self$mean_function$parameters <- mean$parameters
                           if(is.null(mean$data)){
                             self$mean_function$data <- data 
                           } else {
                             self$mean_function$data <- mean$data
                           }
                         }
                         
                         if(!is.null(var_par)){
                           self$var_par <- var_par
                         } else {
                           self$var_par <- 1
                         }
                         
                         self$covariance$check(verbose=verbose)
                         self$mean_function$check(verbose = verbose)
                         if(!skip.sigma)private$generate()
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
                         cat("\n   \U2223     \U2BA1 Formula: ~",as.character(self$mean_function$formula)[2])
                         cat("\n   \U2223     \U2BA1 Parameters: ",self$mean_function$parameters)
                         cat("\n   \U2BA1 Covariance")
                         cat("\n   \U2223     \U2BA1 Formula: ~",as.character(self$covariance$formula)[2])
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
                         self$mean_function$n()
                       },
                       #' @description
                       #' Returns the number of clusters at each level
                       #' @details
                       #' **Number of clusters**
                       #' Returns a data frame describing the number of independent clusters or groups at each level in the design. For example,
                       #' if there were cluster-periods nested in clusters, then the top level would be clusters, and the second level would be 
                       #' cluster periods.
                       #' @param ... ignored
                       #' @return A data frame with the level, number of clusters, and variables describing each level.
                       #' @examples 
                       #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                       #' df$int <- 0
                       #' df[df$cl > 5, 'int'] <- 1
                       #' 
                       #' des <- Model$new(
                       #'   covariance = list(formula = ~ (1|gr(cl)) + (1|gr(cl*t))),
                       #'   mean = list(formula = ~ factor(t) + int - 1),
                       #'   data = df, 
                       #'   family = stats::gaussian(),
                       #'   var_par = 1
                       #' )
                       #' des$n_cluster() ## returns two levels of 10 and 50
                       n_cluster = function(){
                         # gr_var <- apply(self$covariance$.__enclos_env__$private$D_data$func_def,1,
                         #                 function(x)any(x==1))
                         n_blocks <- max(self$covariance$.__enclos_env__$private$D_data$cov[,1])
                         gr_var <- sapply(0:n_blocks,function(i) any(self$covariance$.__enclos_env__$private$D_data$cov[self$covariance$.__enclos_env__$private$D_data$cov[,1]==i,3] == 1))
                         gr_count <- self$covariance$.__enclos_env__$private$D_data$cov[!duplicated(self$covariance$.__enclos_env__$private$D_data$cov[,1]),2]
                         flist <- rev(self$covariance$.__enclos_env__$private$flistvars)
                         gr_cov_var <- lapply(flist,function(x)x$rhs)
                         if(any(gr_var)){
                           dfncl <- data.frame(Level = 1:sum(gr_var),"N.clusters"=sort(gr_count[gr_var]),"Variables"=unlist(lapply(gr_cov_var,paste0,collapse=" "))[gr_var][order(gr_count[gr_var])])
                         } else {
                           dfncl <- data.frame(Level = 1,"N.clusters"=1,"Variables"=paste0(unlist(gr_cov_var)[!duplicated(unlist(gr_cov_var))],collapse=" "))
                         }
                         return(dfncl)
                         
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
                       #' ids_to_keep <- which(des$mean_function$data$J!=1)
                       #' des$subset_rows(ids_to_keep)
                       subset_rows = function(index){
                         self$mean_function$subset_rows(index)
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
                         self$mean_function$subset_cols(index)
                         self$check(verbose=FALSE)
                       },
                       #'@description 
                       #'Generates a realisation of the design
                       #'
                       #'Generates a single vector of outcome data based upon the 
                       #'specified GLMM design
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
                         re <- do.call(sample_re,append(self$covariance$get_D_data(),
                                                        list(eff_range = self$covariance$eff_range,
                                                             gamma = self$covariance$parameters)))
                         mu <- c(drop(as.matrix(self$mean_function$X)%*%self$mean_function$parameters)) + c(as.matrix(self$covariance$Z)%*%re)
                         
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
                         
                         if(type=="data.frame"|type=="data")y <- cbind(y,self$mean_function$data)
                         if(type=="all")y <- list(y = y, X = self$mean_function$X, beta = self$mean_function$parameters,
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
                       #' des$mean_function$parameters <- c(rnorm(5),0.1)
                       #' des$check() #updates 
                       check = function(verbose=TRUE){
                         self$covariance$check(verbose=verbose)
                         self$mean_function$check(verbose = verbose)
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
                                                    verbose = FALSE){
                         if(!is.null(mean.pars))self$mean_function$update_parameters(mean.pars,verbose)
                         if(!is.null(cov.pars))self$covariance$update_parameters(cov.pars,verbose)
                         self$check(verbose)
                       },
                       #' @description 
                       #' Generates the information matrix
                       #' @return A PxP matrix
                       information_matrix = function(){
                         Matrix::crossprod(self$mean_function$X,solve(self$Sigma))%*%self$mean_function$X
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
                       #'     formula = ~ (1|gr(cl)) + (1|gr(cl*t)),
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
                           pwr <- pnorm(abs(self$mean_function$parameters/v0) - qnorm(1-alpha/2))
                         } else {
                           if(alternative == "pos"){
                             pwr <- pnorm(self$mean_function$parameters/v0 - qnorm(1-alpha/2))
                           } else {
                             pwr <- pnorm(-self$mean_function$parameters/v0 - qnorm(1-alpha/2))
                           }
                         }
                         
                         res <- data.frame(Parameter = colnames(self$mean_function$X),
                                           Value = self$mean_function$parameters,
                                           SE = v0,
                                           Power = pwr)
                         print(res)
                         return(invisible(res))
                       }
                     ),
                     private = list(
                       W = NULL,
                       Xb = NULL,
                       logit = function(x){
                         exp(x)/(1+exp(x))
                       },
                       generate = function(){
                         private$genW()
                         private$genS()
                       },
                       genW = function(){
                         # assume random effects value is at zero
                         if(!self$family[[1]]%in%c("poisson","binomial","gaussian","Gamma","beta"))stop("family must be one of Poisson, Binomial, Gaussian, Gamma, Beta")
                         xb <- c(self$mean_function$.__enclos_env__$private$Xb)
                         if(private$attenuate_parameters){
                           xb <- attenuate_xb(xb = xb,
                                              Z = as.matrix(self$covariance$Z),
                                              D = as.matrix(self$covariance$D),
                                              link = self$family[[2]])
                         }
                         wdiag <- gen_dhdmu(xb = xb,
                                            family=self$family[[1]],
                                            link = self$family[[2]])
                         
                         if(self$family[[1]] == "gaussian"){
                           wdiag <- self$var_par * self$var_par * wdiag
                         } else if(self$family[[1]] == "Gamma"){
                           wdiag <- wdiag/self$var_par
                         } else if(self$family[[1]] == "beta"){
                           wdiag <- wdiag*(1+self$var_par)
                         }
                         
                         W <- diag(drop(wdiag))
                         private$W <- Matrix::Matrix(W)
                       },
                       genS = function(update=TRUE){
                         S = gen_sigma_approx(xb=matrix(self$mean_function$X%*%self$mean_function$parameters,ncol=1),
                                              Z = as.matrix(self$covariance$Z),
                                              D = as.matrix(self$covariance$D),
                                              family=self$family[[1]],
                                              link = self$family[[2]],
                                              var_par = self$var_par,
                                              attenuate = private$attenuate_parameters)
                         if(update){
                           self$Sigma <- Matrix::Matrix(S)
                           private$hash <- private$hash_do()
                         } else {
                           return(S)
                         }
                       },
                       attenuate_parameters = TRUE,
                       hash = NULL,
                       hash_do = function(){
                         digest::digest(c(self$covariance$.__enclos_env__$private$hash,
                                          self$mean_function$.__enclos_env__$private$hash))
                       }
                     ))


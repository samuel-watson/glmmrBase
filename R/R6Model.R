#' A GLMM Model 
#' 
#' An R6 class representing a GLMM model
#' @details
#' For the generalised linear mixed model 
#' 
#' \deqn{Y \sim F(\mu,\sigma)}
#' \deqn{\mu = h^-1(X\beta + Z\gamma)}
#' \deqn{\gamma \sim MVN(0,D)}
#' 
#' where h is the link function. A Model in comprised of a \link[glmmrBase]{MeanFunction} object, which defines the family F, 
#' link function h, and fixed effects design matrix X, and a \link[glmmrBase]{Covariance} object, which defines Z and D. The class provides
#' methods for analysis and simulation with these models.
#' 
#' This class provides methods for generating the matrices described above and data simulation, and serves as a base for extended functionality 
#' in related packages.
#' 
#' The class by default calculates the covariance matrix of the observations as:
#' 
#' \deqn{\Sigma = W^{-1} + ZDZ^T}
#' 
#' where _W_ is a diagonal matrix with the WLS iterated weights for each observation equal
#' to, for individual _i_ \eqn{\phi a_i v(\mu_i)[h'(\mu_i)]^2} (see Table 2.1 in McCullagh 
#' and Nelder (1989) <ISBN:9780412317606>). For very large designs, this can be disabled as
#' the memory requirements can be prohibitive. 
#' 
#' See \href{https://github.com/samuel-watson/glmmrBase/blob/master/README.md}{glmmrBase} for a 
#' detailed guide on model specification.
#' @importFrom Matrix Matrix
#' @export 
Model <- R6::R6Class("Model",
                  public = list(
                    #' @field covariance A \link[glmmrBase]{Covariance} object defining the random effects covariance.
                    covariance = NULL,
                    #' @field mean_function A \link[glmmrBase]{MeanFunction} object, defining the mean function for the model, including the data and covariate design matrix X.
                    mean_function = NULL,
                    #' @field exp_condition A vector indicting the unique experimental conditions for each observation, see Details.
                    exp_condition = NULL,
                    #' @field Sigma The overall covariance matrix for the observations. Calculated and updated automatically as \eqn{W^{-1} + ZDZ^T} where W is an n x n 
                    #' diagonal matrix with elements on the diagonal equal to the GLM iterated weights. See Details.
                    Sigma = NULL,
                    #' @field var_par Scale parameter required for some distributions (Gaussian, Gamma, Beta).
                    var_par = NULL,
                    #' @description 
                    #' Return predicted values based on the currently stored parameter values in `mean_function`
                    #' @param type One of either "`link`" for values on the scale of the link function, or "`response`" 
                    #' for values on the scale of the response
                    #' @return A \link[Matrix]{Matrix} class object containing the predicted values
                    fitted = function(type="link"){
                      Xb <- Matrix::drop(self$mean_function$X %*% self$mean_function$parameters)
                      if(type=="response"){
                        Xb <- self$mean_function$family$linkinv(Xb)
                      }
                      return(Xb)
                    },
                    #' @description 
                    #' Create a new Model object
                    #' @param covariance Either a \link[glmmrBase]{Covariance} object, or an equivalent list of arguments
                    #' that can be passed to `Covariance` to create a new object.
                    #' @param mean.function Either a \link[glmmrBase]{MeanFunction} object, or an equivalent list of arguments
                    #' that can be passed to `MeanFunction` to create a new object.
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
                    #'   data=df,
                    #'   parameters = c(rep(0,5),0.6),
                    #'   family = stats::gaussian()
                    #' )
                    #' cov1 <- Covariance$new(
                    #'   data = df,
                    #'   formula = ~ (1|gr(cl)) + (1|gr(cl*t)),
                    #'   parameters = c(0.25,0.1)
                    #' )
                    #' des <- Model$new(
                    #'   covariance = cov1,
                    #'   mean.function = mf1,
                    #'   var_par = 1
                    #' )
                    #' 
                    #' #alternatively we can pass the data directly to Model
                    #' #here we will specify a cohort study
                    #' df <- nelder(~ind(20) * t(6))
                    #' df$int <- 0
                    #' df[df$t > 3, 'int'] <- 1
                    #' 
                    #' des <- Model$new(
                    #' covariance = list(
                    #'   data=df,
                    #'   formula = ~ (1|gr(ind)*ar1(t)),
                    #'   parameters = c(0.25,0.8)),
                    #' mean.function = list(
                    #'   formula = ~factor(t) + int - 1,
                    #'   data=df,
                    #'   parameters = rep(0,7),
                    #'   family = stats::poisson()))
                    #'                   
                    #' #an example of a spatial grid with two time points
                    #' df <- nelder(~ (x(10)*y(10))*t(2))
                    #' spt_design <- Model$new(covariance = list(data=df,
                    #'                                            formula = ~(1|fexp(x,y)*ar1(t)),
                    #'                                            parameters =c(0.2,0.1,0.8)),
                    #'                          mean.function = list(data=df,
                    #'                                               formula = ~ 1,
                    #'                                               parameters = c(0.5),
                    #'                                               family=stats::poisson())) 
                    initialize = function(covariance,
                                          mean.function,
                                          var_par = NULL,
                                          verbose=TRUE,
                                          skip.sigma = FALSE){
                      if(is(covariance,"R6")){
                        if(is(covariance,"Covariance")){
                          self$covariance <- covariance
                        } else {
                          stop("covariance should be Covariance class or list of appropriate arguments")
                        }
                      } else if(is(covariance,"list")){
                        if(is.null(covariance$eff_range))covariance$eff_range = NULL
                        self$covariance <- Covariance$new(
                          formula= covariance$formula,
                          data = covariance$data,
                          parameters = covariance$parameters,
                          eff_range = covariance$eff_range,
                          verbose = verbose
                        )
                        
                      }

                      if(is(mean.function,"R6")){
                        if(is(mean.function,"MeanFunction")){
                          self$mean_function <- mean.function
                        } else {
                          stop("mean.function should be MeanFunction class or list of appropriate arguments")
                        }
                      } else if(is(mean.function,"list")){
                        if("random_function"%in%names(mean.function)){
                          rfunc <- mean.function$random_function
                          tpar <- mean.function$treat_var
                        } else {
                          rfunc <- NULL
                          tpar <- NULL
                        }
                        self$mean_function <- MeanFunction$new(
                          formula = mean.function$formula,
                          data = mean.function$data,
                          family = mean.function$family,
                          parameters = mean.function$parameters,
                          random_function = rfunc,
                          treat_var = tpar,
                          verbose = verbose
                        )
                      }

                      self$var_par <- var_par

                      if(!skip.sigma)private$generate()
                      private$hash <- private$hash_do()
                    },
                    #' @description 
                    #' Print method for `Model` class
                    #' @details 
                    #' Calls the respective print methods of the linked covariance and mean function objects.
                    #' @param ... ignored
                    print = function(){
                      cat("\n----------------------------------------\n")
                      print(self$mean_function)
                      cat("\n----------------------------------------\n")
                      print(self$covariance)
                      cat("\n----------------------------------------\n")
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
                    #' mf1 <- MeanFunction$new(
                    #'   formula = ~ factor(t) + int - 1,
                    #'   data=df,
                    #'   parameters = c(rep(0,5),0.6),
                    #'   family = stats::gaussian()
                    #' )
                    #' cov1 <- Covariance$new(
                    #'   data = df,
                    #'   formula = ~ (1|gr(cl)) + (1|gr(cl*t)),
                    #'   parameters = c(0.25,0.1)
                    #' )
                    #' des <- Model$new(
                    #'   covariance = cov1,
                    #'   mean.function = mf1,
                    #'   var_par = 1
                    #' )
                    #' des$n_cluster() ## returns two levels of 10 and 50
                    n_cluster = function(){
                      gr_var <- apply(self$covariance$.__enclos_env__$private$D_data$func_def,1,
                                      function(x)any(x==1))
                      gr_count <- self$covariance$.__enclos_env__$private$D_data$N_dim
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
                    #'@param type Either 'y' to return just the outcome data, or 'data'
                    #' to return a data frame with the simulated outcome data alongside the model data 
                    #' @return Either a vector or a data frame
                    #' @examples
                    #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                    #' df$int <- 0
                    #' df[df$cl > 5, 'int'] <- 1
                    #' des <- Model$new(
                    #'   covariance = list(
                    #'     data = df,
                    #'     formula = ~ (1|gr(cl)*ar1(t)),
                    #'     parameters = c(0.25,0.8)),
                    #'   mean.function = list(
                    #'     formula = ~ factor(t) + int - 1,
                    #'     data=df,
                    #'     parameters = c(rep(0,5),0.6),
                    #'     family = stats::binomial())
                    #' )
                    #' ysim <- des$sim_data()
                    sim_data = function(type = "y"){
                      z <- stats::rnorm(nrow(self$covariance$D))
                      L <- blockMat(self$covariance$get_chol_D())
                      re <- L%*%matrix(z,ncol=1)
                      mu <- c(drop(as.matrix(self$mean_function$X)%*%self$mean_function$parameters)) + as.matrix(self$covariance$Z%*%re)
                      
                      f <- self$mean_function$family
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

                      if(f[1]=="gamma"){
                        if(f[2]=="inverse"){
                          if(is.null(self$var_par))stop("For gamma(link='inverse') provide var_par")
                          #CHECK THIS IS RIGHT
                          y <- rgamma(self$n(),shape = 1/(mu*self$var_par),rate = 1/self$var_par)
                        }
                      }
                     
                      if(type=="data.frame"|type=="data")y <- cbind(y,self$mean_function$data)
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
                    #'     data = df,
                    #'     formula = ~ (1|gr(cl)*ar1(t)),
                    #'     parameters = c(0.25,0.8)),
                    #'   mean.function = list(
                    #'     formula = ~ factor(t) + int - 1,
                    #'     data=df,
                    #'     parameters = c(rep(0,5),0.6),
                    #'     family = stats::binomial())
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
                    #' Generates the information matrix
                    #' @return A PxP matrix
                    information_matrix = function(){
                      Matrix::crossprod(self$mean_function$X,solve(self$Sigma))%*%self$mean_function$X
                    }
                  ),
                  private = list(
                    W = NULL,
                    Xb = NULL,
                    logit = function(x){
                      exp(x)/(1+exp(x))
                    },
                    generate = function(){
                      # add check for var par with gaussian family
                      
                      private$genW(family = self$mean_function$family,
                                   Xb = self$mean_function$.__enclos_env__$private$Xb,
                                   var_par = self$var_par)
                      private$genS(D = self$covariance$D,
                                   Z = self$covariance$Z,
                                   W = private$W)
                    },
                    genW = function(family,
                                    Xb,
                                    var_par=NULL){
                      # assume random effects value is at zero
                      if(!family[[1]]%in%c("poisson","binomial","gaussian","gamma"))stop("family must be one of Poisson, Binomial, Gaussian, Gamma")
                      
                      wdiag <- gen_dhdmu(c(Xb),
                                         family=family[[1]],
                                         link = family[[2]])
                      
                      if(family[[1]]%in%c("gaussian","gamma")){
                        wdiag <- var_par * wdiag
                      }
                      W <- diag(drop(wdiag))
                      private$W <- Matrix::Matrix(W)
                    },
                    genS = function(D,Z,W,update=TRUE){
                      if(is(D,"numeric")){
                        S <- W + D * Matrix::tcrossprod(Z)
                      } else {
                        S <- W + Z %*% Matrix::tcrossprod(D,Z)
                      }
                      if(update){
                        self$Sigma <- Matrix::Matrix(S)
                        private$hash <- private$hash_do()
                      } else {
                        return(S)
                      }

                    },
                    hash = NULL,
                    hash_do = function(){
                      digest::digest(c(self$covariance$.__enclos_env__$private$hash,
                                       self$mean_function$.__enclos_env__$private$hash))
                    }
                  ))



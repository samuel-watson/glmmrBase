#' R6 Class representing a mean function/linear predictor
#' 
#' For the generalised linear mixed model 
#' 
#' \deqn{Y \sim F(\mu,\sigma)}
#' \deqn{\mu = h^-1(X\beta + Z\gamma)}
#' \deqn{\gamma \sim MVN(0,D)}
#' 
#' this class defines the fixed effects design matrix X. 
#' The mean function is defined by a model formula, data, and parameters.
#' A new instance can be generated with $new(). The class will generate the 
#' relevant matrix X automatically. See \href{https://github.com/samuel-watson/glmmrBase/blob/master/README.md}{glmmrBase} for a 
#' detailed guide on model specification.
#' @export
MeanFunction <- R6::R6Class("MeanFunction",
                        public = list(
                          #' @field formula model formula for the fixed effects
                          formula = NULL,
                          #' @field data Data frame with data required to build X
                          data = NULL,
                          #' @field parameters A vector of parameter values for \eqn{\beta} used for simulating data and calculating
                          #' covariance matrix of observations for non-linear models.
                          parameters = NULL,
                          #' @field offset An optional vector specifying the offset values
                          offset = NULL,
                          #' @field X the fixed effects design matrix
                          X = NULL,
                          #' @description 
                          #' Returns the number of observations
                          #' 
                          #' @param ... ignored
                          #' @return The number of observations in the model
                          #' @examples
                          #' \dontshow{
                          #' setParallel(FALSE) # for the CRAN check
                          #' }
                          #' df <- nelder(~(cl(4)*t(5)) > ind(5))
                          #' df$int <- 0
                          #' df[df$cl <= 2, 'int'] <- 1
                          #' mf1 <- MeanFunction$new(formula = ~ int ,
                          #'                         data=df,
                          #'                         parameters = c(-1,1)
                          #'                         )
                          #' mf1$n()
                          n=function(){
                            nrow(self$data)
                          },
                          #' @description 
                          #' Create a new MeanFunction object
                          #' 
                          #' @details 
                          #' Specification of the mean function follows standard model formulae in R. 
                          #' For example for a stepped-wedge cluster trial model, a typical mean model is 
                          #' \eqn{E(y_{ijt}|\delta)=\beta_0 + \tau_t + \beta_1 d_{jt} + z_{ijt}\delta} where \eqn{\tau_t} 
                          #' are fixed effects for each time period. The formula specification for this would be `~ factor(t) + int` 
                          #' where `int` is the name of the variable indicating the treatment.
                          #' 
                          #' One can also include non-linear functions of variables in the mean function, and name the parameters. 
                          #' The resulting X matrix is then a matrix of first-order partial derivatives. For example, one can
                          #' specify `~ int + b_1*exp(b_2*x)`.
                          #' 
                          #' @param formula A \link[stats]{formula} object that describes the mean function, see Details
                          #' @param data (Optional) A data frame containing the covariates in the model, named in the model formula
                          #' @param parameters (Optional) A vector with the values of the parameters \eqn{\beta} to use in data simulation and covariance calculations.
                          #' If the parameters are not specified then they are initialised to 0.
                          #' @param offset A vector of offset values (optional)
                          #' @param verbose Logical indicating whether to report detailed output
                          #' @return A MeanFunction object
                          #' @examples 
                          #' \dontshow{
                          #' setParallel(FALSE) # for the CRAN check
                          #' }
                          #' df <- nelder(~(cl(4)*t(5)) > ind(5))
                          #' df$int <- 0
                          #' df[df$cl <= 2, 'int'] <- 1
                          #' mf1 <- MeanFunction$new(formula = ~ int ,
                          #'                         data=df,
                          #'                         parameters = c(-1,1),
                          #'                         )
                          initialize = function(formula,
                                                data,
                                                parameters = NULL ,
                                                offset = NULL,
                                                verbose = FALSE){

                            self$formula <- Reduce(paste,as.character(formula))
                            if(!is(data,"data.frame"))stop("data must be data frame")
                            self$data <- data
                            
                            private$original_formula <- self$formula
                            
                            if(is.null(offset) & !is.null(data)){
                              self$offset <- rep(0,nrow(self$data))
                            } 
                            
                            private$generate(verbose=verbose)
                            if(!is.null(parameters)){
                              self$update_parameters(parameters)
                            } else {
                              self$update_parameters(runif(ncol(self$X),-2,2))
                            }
                            
                            if(verbose)self$print()
                            
                            },
                          #' @description 
                          #' Prints details about the object
                          #' 
                          #' @param ... ignored
                          print = function(){
                            cat("\U2BC8 Linear Predictor")
                            cat("\n     \U2BA1 Formula: ~",self$formula)
                            cat("\n     \U2BA1 Parameters: ",self$parameters)
                            cat("\n")
                          },
                          #' @description 
                          #' Updates the model parameters
                          #' 
                          #' @details 
                          #' Using `update_parameters()` is the preferred way of updating the parameters of the 
                          #' mean or covariance objects as opposed to direct assignment, e.g. `self$parameters <- c(...)`. 
                          #' The function calls check functions to automatically update linked matrices with the new parameters.
                          #' 
                          #' @param parameters A vector of parameters for the mean function.
                          #' @param verbose Logical indicating whether to provide more detailed feedback
                          update_parameters = function(parameters){
                            self$parameters <- parameters
                            Linpred__update_pars(private$ptr,self$parameters)
                            if(Linpred__any_nonlinear(private$ptr))self$X <- Linpred__x(private$ptr)
                            names(self$parameters) <- Linpred__beta_names(private$ptr)
                          },
                          #' @description 
                          #' Returns or replaces the column names of the data in the object
                          #' 
                          #' @param names If NULL then the function prints the column names, if a vector of names, then it attempts to 
                          #' replace the current column names of the data
                          #' @examples 
                          #' \dontshow{
                          #' setParallel(FALSE) # for the CRAN check
                          #' }
                          #' df <- nelder(~(cl(4)*t(5)) > ind(5))
                          #' df$int <- 0
                          #' df[df$cl <= 5, 'int'] <- 1
                          #' mf1 <- MeanFunction$new(formula = ~ int ,
                          #'                         data=df,
                          #'                         parameters = c(-1,1)
                          #'                         )
                          #' mf1$colnames(c("cluster","time","individual","treatment"))
                          #' mf1$colnames()
                          colnames = function(names = NULL){
                            if(is.null(names)){
                              print(colnames(self$data))
                            } else {
                              colnames(self$data) <- names
                            }
                          },
                          #' @description 
                          #' Keeps a subset of the data and removes the rest
                          #' 
                          #' All indices not in the provided vector of row numbers will be removed from both the data and fixed effects 
                          #' design matrix X.
                          #' 
                          #' @param index Rows of the data to keep
                          #' @return NULL
                          #' @examples 
                          #' \dontshow{
                          #' setParallel(FALSE) # for the CRAN check
                          #' }
                          #' df <- nelder(~(cl(4)*t(5)) > ind(5))
                          #' df$int <- 0
                          #' df[df$cl <= 5, 'int'] <- 1
                          #' mf1 <- MeanFunction$new(formula = ~ int ,
                          #'                         data=df,
                          #'                         parameters = c(-1,1)
                          #'                         )
                          #' mf1$subset_rows(1:20) 
                          subset_rows = function(index){
                            self$X <- self$X[index,]
                            self$data <- self$data[index,]
                            self$offset <- self$offset[index]
                            private$update_ptr()
                            private$genX()
                          },
                          #' @description 
                          #' Returns the linear predictor 
                          #' 
                          #' Returns the linear predictor, X * beta
                          #' @return A vector
                          linear_predictor = function(){
                            xb <- Linpred__xb(private$ptr) + self$offset #self$X %*% self$parameters + self$offset
                            if(is(xb,"matrix"))xb <- drop(xb)
                            if(is(xb,"Matrix"))xb <- Matrix::drop(xb)
                            return(xb)
                          },
                          #' @description
                          #' Returns a logical indicating whether the mean function contains non-linear functions of model parameters.
                          #' Mainly used internally.
                          #' @return None. Called for effects
                          any_nonlinear = function(){
                            return(Linpred__any_nonlinear(private$ptr))
                          }
                        ),
                        private = list(
                          mod_string = NULL,
                          form = NULL,
                          original_formula = NULL,
                          ptr = NULL,
                          update_ptr = function(){
                            private$ptr <- Linpred__new(self$formula,
                                                  as.matrix(self$data),
                                                  colnames(self$data))
                          },
                          generate = function(verbose = FALSE){
                            self$formula <- private$original_formula
                            if(grepl("~",self$formula) && length(as.formula(self$formula))==3)stop("formula should not have dependent variable.")
                            if(grepl("~",self$formula))self$formula <- gsub("~","",self$formula)
                            self$formula <- gsub(" ","",self$formula)
                            self$formula <- gsub("\\+\\([^ \\+]\\|.*\\)","",self$formula,perl = T)
                            private$update_ptr()
                            private$genX()
                          },
                          genX = function(){
                            self$X <- Linpred__x(private$ptr)
                            if(!is.null(self$parameters)&ncol(self$X)!=length(self$parameters))stop("wrong length parameter vector")
                            
                          }
                          
                        ))




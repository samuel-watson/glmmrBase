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
                          #' @field X the fixed effects design matrix
                          X = NULL,
                          #' @description 
                          #' Returns the number of observations
                          #' 
                          #' @param ... ignored
                          #' @return The number of observations in the model
                          #' @examples
                          #' df <- nelder(~(cl(4)*t(5)) > ind(5))
                          #' df$int <- 0
                          #' df[df$cl <= 5, 'int'] <- 1
                          #' mf1 <- MeanFunction$new(formula = ~ int ,
                          #'                         data=df,
                          #'                         parameters = c(-1,1)
                          #'                         )
                          #' mf1$n()
                          n=function(){
                            nrow(self$data)
                          },
                          #' @description 
                          #' Checks if any changes have been made and updates
                          #' 
                          #' Checks if any changes have been made and updates, usually called automatically.
                          #' @param verbose Logical whether to report if any changes detected.
                          #' @return NULL
                          #' @examples
                          #' df <- nelder(~(cl(4)*t(5)) > ind(5))
                          #' df$int <- 0
                          #' df[df$cl <= 5, 'int'] <- 1
                          #' mf1 <- MeanFunction$new(formula = ~ int ,
                          #'                         data=df,
                          #'                         parameters = c(-1,1)
                          #'                         )
                          #' mf1$parameters <- c(0,0)
                          #' mf1$check()
                          check = function(verbose=TRUE){
                            if(private$hash != private$hash_do()){
                              if(verbose)message("Updating model")
                              private$generate()
                            }},
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
                          #' One can also include non-linear functions of variables in the mean function. These are handled in the analyses 
                          #' by first-order approximation. 
                          #' 
                          #' If not all of `formula`, `data`, and `parameters` are not specified then the linked matrices 
                          #' are not calculated. These options can be later specified, or updated via a \link[glmmrBase]{Model} object.
                          #' If these arguments are updated or changed then call `self$check()` to update linked matrices. Updating of 
                          #' parameters is automatic if using the `update_parameters()` member function.
                          #' @param formula A \link[stats]{formula} object that describes the mean function, see Details
                          #' @param data (Optional) A data frame containing the covariates in the model, named in the model formula
                          #' @param parameters (Optional) A vector with the values of the parameters \eqn{\beta} to use in data simulation and covariance calculations.
                          #' If the parameters are not specified then they are initialised to 0.
                          #' @param verbose Logical indicating whether to report detailed output
                          #' @return A MeanFunction object
                          #' @examples 
                          #' df <- nelder(~(cl(4)*t(5)) > ind(5))
                          #' df$int <- 0
                          #' df[df$cl <= 5, 'int'] <- 1
                          #' mf1 <- MeanFunction$new(formula = ~ int ,
                          #'                         data=df,
                          #'                         parameters = c(-1,1),
                          #'                         )
                          initialize = function(formula,
                                                data = NULL,
                                                parameters = NULL ,
                                                verbose = FALSE
                          ){

                            allset <- TRUE
                            self$formula <- as.formula(formula, env=.GlobalEnv)
                            
                            if(!is.null(data)){
                              if(!is(data,"data.frame"))stop("data must be data frame")
                              self$data <- data
                            } else {
                              allset <- FALSE
                            }
                            
                            if(!is.null(parameters)){
                              self$parameters <- parameters
                            } 
                            
                            if(allset){
                              private$generate(verbose=verbose)
                            } else {
                              private$hash <- digest::digest(1)
                            }
                            if(verbose & allset)self$print()
                            
                            },
                          #' @description 
                          #' Prints details about the object
                          #' 
                          #' @param ... ignored
                          print = function(){
                            cat("\U2BC8 Linear Predictor")
                            cat("\n     \U2BA1 Formula: ~",as.character(self$formula)[2])
                            cat("\n     \U2BA1 Parameters: ",self$parameters)
                          },
                          #' @description 
                          #' Updates the model parameters
                          #' 
                          #' @details 
                          #' Using `update_parameters()` is the preferred way of updating the parameters of the 
                          #' mean or covariance objects as opposed to direct assignment, e.g. `self$parameters <- c(...)`. 
                          #' The function calls check functions to automatically update linked matrices with the new parameters.
                          #' If using direct assignment, call `self$check()` afterwards. 
                          #' 
                          #' @param parameters A vector of parameters for the mean function.
                          #' @param verbose Logical indicating whether to provide more detailed feedback
                          update_parameters = function(parameters,
                                                       verbose = FALSE){
                            self$parameters <- parameters
                            self$check(verbose)
                          },
                          #' @description 
                          #' Returns or replaces the column names of the data in the object
                          #' 
                          #' @param names If NULL then the function prints the column names, if a vector of names, then it attemps to 
                          #' replace the current column names of the data
                          #' @examples 
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
                          },
                          #' @description 
                          #' Keeps a subset of the columns of X 
                          #' 
                          #' All indices not in the provided vector of column numbers will be removed from the fixed effects design
                          #' matrix X.
                          #' 
                          #' @param index Columns of X to keep
                          #' @return NULL
                          #' @examples 
                          #' df <- nelder(~(cl(4)*t(5)) > ind(5))
                          #' df$int <- 0
                          #' df[df$cl <= 5, 'int'] <- 1
                          #' mf1 <- MeanFunction$new(formula = ~ int ,
                          #'                         data=df,
                          #'                         parameters = c(-1,1)
                          #'                         )
                          #' mf1$subset_cols(1:2) 
                          subset_cols = function(index){
                            self$X <- self$X[,index]
                          }
                        ),
                        private = list(
                          mod_string = NULL,
                          form = NULL,
                          Xb = NULL,
                          funs = NULL,
                          vars = NULL,
                          hash = NULL,
                          hash_do = function(){
                            digest::digest(c(self$formula,self$data,
                                             self$parameters))
                          },
                          generate = function(verbose = FALSE){
                            
                            if(length(self$formula)==3)stop("formula should not have dependent variable.")
                            #check if all parameters in data
                            if(any(!all.vars(self$formula)%in%colnames(self$data)))stop("variables not in data frame")
                            
                            private$genTerms()
                            private$genX()
                            private$hash <- private$hash_do()
                          },
                          genTerms = function(){
                            mf1 <- self$formula[[2]]
                            checkTerm <- TRUE
                            iter <- 0
                            funs <- list()
                            vars <- list()
                            while(checkTerm){
                              iter <- iter + 1
                              checkTerm <- I(length(mf1)>1 && (mf1[[1]]=="+"|mf1[[1]]=="-"))
                              if(checkTerm){
                                vars[[iter]] <- all.vars(mf1[[3]])
                                if(length(mf1[[3]])==1){
                                  funs[[iter]] <- "identity"
                                } else {
                                  funs[[iter]] <- as.character(mf1[[3]][[1]])
                                }
                                if(length(vars[[iter]])==0){
                                  vars[[iter]] <- funs[[iter]] <- "RMINT"
                                }
                                mf1 <- mf1[[2]]
                              } else {
                                vars[[iter]] <- all.vars(mf1)
                                if(length(mf1)==1){
                                  funs[[iter]] <- "identity"
                                } else {
                                  funs[[iter]] <- as.character(mf1[[1]])
                                }
                              }
                            }

                            private$funs <- rev(funs)
                            private$vars <- rev(vars)

                          },
                          genX = function(){
                            # generate model matrix X, including linearisation of non-linear terms,
                            X <- matrix(1,nrow=self$n(),ncol=1)
                            colnames(X) <- "(Intercept)"
                            for(i in 1:length(private$funs)){
                              if(private$funs[[i]]=="RMINT")next
                              Xadd <- do.call(paste0("d",private$funs[[i]]),list(list(
                                data = as.matrix(self$data[,private$vars[[i]]]),
                                pars = self$parameters[[i]]
                              )))
                              #add colnames depending on the function
                              if(private$funs[[i]] == "factor"){
                                colnames(Xadd) <- paste0(private$vars[[i]],levels(factor(self$data[,private$vars[[i]]])))
                              } else if(private$funs[[i]] == "identity"){
                                colnames(Xadd) <- private$vars[[i]]
                              } else {
                                colnames(Xadd) <- paste0(private$vars[[i]],ncol(Xadd))
                              }
                              
                              X <- cbind(X,Xadd)
                            }
                            if(any(private$funs=="RMINT"))X <- X[,-1]
                            if(is.null(self$parameters))self$parameters <- rep(0,ncol(X))
                            if(ncol(X)!=length(unlist(self$parameters)))warning("wrong number of parameters")
                            private$Xb <- X %*% matrix(unlist(self$parameters[1:ncol(X)]),ncol=1)
                            self$X <- Matrix::Matrix(X)
                            
                          }
                          
                        ))




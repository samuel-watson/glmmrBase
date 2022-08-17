#' R6 Class representing a mean function function and data
#' 
#' For the generalised linear mixed model 
#' 
#' \deqn{Y \sim F(\mu,\sigma)}
#' \deqn{\mu = h^-1(X\beta + Z\gamma)}
#' \deqn{\gamma \sim MVN(0,D)}
#' 
#' this class defines the family F, link function h, and fixed effects design matrix X. 
#' The mean function is defined by a model formula, data, and parameters.
#' A new instance can be generated with $new(). The class will generate the 
#' relevant matrix X automatically. 
#' @export
MeanFunction <- R6::R6Class("MeanFunction",
                        public = list(
                          #' @field formula model formula for the fixed effects
                          formula = NULL,
                          #' @field data Data frame with data required to build X
                          data = NULL,
                          #' @field family One of the family function used in R's glm functions. See \link[stats]{family} for details
                          family = NULL,
                          #' @field parameters A vector of parameter values for \eqn{\beta} used for simulating data and calculating
                          #' covariance matrix of observations for non-linear models.
                          parameters = NULL,
                          #' @field randomise A function that generates a new set of values representing the treatment allocation in an 
                          #' experimental study
                          randomise = NULL,
                          #' @field treat_var A string naming the column in data that represents the treatment variable in data. Used
                          #' to identify where to replace allocation when randomiser is used.
                          treat_var = NULL,
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
                          #'                         parameters = c(-1,1),
                          #'                         family = stats::binomial()
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
                          #'                         parameters = c(-1,1),
                          #'                         family = stats::binomial()
                          #'                         )
                          #' mf1$parameters <- c(0,0)
                          #' mf1$check()
                          check = function(verbose=TRUE){
                            if(private$hash != private$hash_do()){
                              if(verbose)message("changes found, updating")
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
                          #' by first-order approximation. Available functions are the same as for the covariance functions 
                          #' see \link[glmmr]{Covariance}. The user can add additional functions by specifying a new function that takes as 
                          #' an input a named list with elements data and pars, and outputs a matrix with the linearised components. 
                          #' The function name must begin with `d`, e.g. the function to provide a first order approximation to
                          #' the exponential function (see \link[glmmr]{fexp}) is named `dfexp`.
                          #' @param formula A \link[stats]{formula} object that describes the mean function, see Details
                          #' @param data A data frame containing the covariates in the model, named in the model formula
                          #' @param family A family object expressing the distribution and link function of the model, see \link[stats]{family}
                          #' @param parameters A vector with the values of the parameters \eqn{\beta} to use in data simulation and covariance calculations
                          #' @param verbose Logical indicating whether to report detailed output
                          #' @param random_function A string naming a function in the global environment that produces a vector of data describing a new
                          #' treatment allocation in an experimental model. When used, the output of this function replaces the column of data named by
                          #' `treat_var`
                          #' @param treat_var The name of a column in data (or the name to give a new column) that a random treatment allocation generated
                          #' by `random_function` replaces.
                          #' @return A MeanFunction object
                          #' @examples 
                          #' df <- nelder(~(cl(4)*t(5)) > ind(5))
                          #' df$int <- 0
                          #' df[df$cl <= 5, 'int'] <- 1
                          #' mf1 <- MeanFunction$new(formula = ~ int ,
                          #'                         data=df,
                          #'                         parameters = c(-1,1),
                          #'                         family = stats::binomial()
                          #'                         )
                          initialize = function(formula,
                                                data,
                                                family,
                                                parameters ,
                                                verbose = FALSE,
                                                random_function=NULL,
                                                treat_var = NULL
                          ){
                            if(any(missing(formula),missing(family),missing(parameters))){
                              cat("not all inputs set. call generate() when set")
                            } else {
                              self$formula <- as.formula(formula, env=.GlobalEnv)
                              self$family <- family
                              self$parameters <- parameters

                              if(!is(data,"data.frame"))stop("data must be data frame")
                              # self$n <- nrow(data)
                              if(!is.null(random_function)){
                                if(is.null(treat_var))stop("provide name of treatment variable treat_var")
                                
                                #test random function
                                test <- random_function()
                                if(!is(test,"numeric") || length(test)!=nrow(data))stop("random_function does not produce a vector")
                                if(verbose)message(paste0("randomise function provided, treatment variable '",treat_var,"' will be the last column of X,
and the parameters should also be in this order"))
                                # check it produces a varia
                                self$randomise <- random_function
                                self$treat_var <- treat_var
                              }
                              self$data <- data
                              private$generate(verbose=verbose)
                              if(verbose)self$print()
                            }},
                          #' @description 
                          #' Prints details about the object
                          #' 
                          #' @param ... ignored
                          print = function(){
                            cat("Mean Function")
                            print(self$family)
                            cat("Formula:")
                            print(self$formula)
                            if(!is.null(self$randomise)){
                              cat(paste0("Treatment variable: ",self$treat_var))
                              cat(paste0("\nRandom allocation function provided, e.g.: ",paste0(head(self$randomise()),collapse = "")))
                            }
                            cat("\n X matrix: \n")
                            print(head(self$X))
                            # cat("Data:\n")
                            # print(head(self$data))
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
                          #'                         parameters = c(-1,1),
                          #'                         family = stats::binomial()
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
                          #'                         parameters = c(-1,1),
                          #'                         family = stats::binomial()
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
                          #'                         parameters = c(-1,1),
                          #'                         family = stats::binomial()
                          #'                         )
                          #' mf1$subset_cols(1:2) 
                          subset_cols = function(index){
                            self$X <- self$X[,index]
                          },
                          #' @description 
                          #' Generates a new random allocation
                          #' 
                          #' If a randomising function has been provided then a new random allocation will
                          #' be generated, and will replace the exisitng data at `treat_var` in the X matrix
                          #' @param ... ignored
                          #' @return Nothing is returned, the X matrix is updated
                          rerandomise = function(){
                            new_draw <- self$randomise()
                            if(!self$treat_var %in% colnames(self$X)){
                              self$X <- cbind(self$X,new_draw)
                              colnames(self$X)[ncol(self$X)] <- self$treat_var
                            }
                            # } else {
                            #   self$X[,self$treat_var] <- new_draw
                            # }
                            private$Xb <- Matrix::drop(self$X %*% matrix(unlist(self$parameters[1:ncol(self$X)]),ncol=1))
                          }
                        ),
                        private = list(
                          mod_string = NULL,
                          form = NULL,
                          Xb = NULL,
                          funs = NULL,
                          vars = NULL,
                          hash = NULL,
                          treat_par = NULL,
                          hash_do = function(){
                            digest::digest(c(self$formula,self$data,self$family,
                                             self$parameters,
                                             self$randomiser))
                            # digest::digest(c(self$formula,self$data,self$family,
                            #                  digest::digest(as.character(self$parameters),serialize = FALSE),
                            #                  self$randomiser))
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
                            if((ncol(X)!=length(unlist(self$parameters))&is.null(self$randomise)) || 
                              (!(length(unlist(self$parameters))%in%c(ncol(X),ncol(X)+1))&!is.null(self$randomise)))warning("wrong number of parameters")
                            private$Xb <- X %*% matrix(unlist(self$parameters[1:ncol(X)]),ncol=1)
                            self$X <- Matrix::Matrix(X)
                            
                            #add random function data
                            if(!is.null(self$randomise))self$rerandomise()
                          }
                          
                        ))




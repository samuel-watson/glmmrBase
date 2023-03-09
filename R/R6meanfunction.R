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
                          #' @param offset A vector of offset values (optional)
                          #' @param verbose Logical indicating whether to report detailed output
                          #' @return A MeanFunction object
                          #' @examples 
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
                                                verbose = FALSE
                          ){

                            allset <- TRUE
                            self$formula <- Reduce(paste,as.character(formula))
                            if(!is(data,"data.frame"))stop("data must be data frame")
                            self$data <- data
                            
                            private$original_formula <- self$formula
                            
                            if(!is.null(parameters)){
                              self$parameters <- parameters
                            } 
                            
                            if(is.null(offset) & !is.null(data)){
                              self$offset <- rep(0,nrow(self$data))
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
                            cat("\n     \U2BA1 Formula: ~",self$formula)
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
                          update_parameters = function(parameters){
                            self$parameters <- parameters
                            self$check(FALSE)
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
                            self$offset <- self$offset[index]
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
                          },
                          #' @description 
                          #' Returns the linear predictor 
                          #' 
                          #' Returns the linear predictor, X * beta
                          #' @return A vector
                          linear_predictor = function(){
                            xb <- self$X %*% self$parameters + self$offset
                            if(is(xb,"matrix"))xb <- drop(xb)
                            if(is(xb,"Matrix"))xb <- Matrix::drop(xb)
                            return(xb)
                          }
                        ),
                        private = list(
                          mod_string = NULL,
                          form = NULL,
                          hash = NULL,
                          original_formula = NULL,
                          hash_do = function(){
                            digest::digest(c(self$formula,self$data,
                                             self$parameters))
                          },
                          generate = function(verbose = FALSE){
                            self$formula <- private$original_formula
                            if(grepl("~",self$formula) && length(as.formula(self$formula))==3)stop("formula should not have dependent variable.")
                            if(grepl("~",self$formula))self$formula <- gsub("~","",self$formula)
                            self$formula <- gsub(" ","",self$formula)
                            #need to remove random effect terms from the formula
                            re <- .re_names(self$formula)
                            if(length(re)>0){
                              for(i in 1:length(re)){
                                re[i] <- gsub("\\(","\\\\(",re[i])
                                re[i] <- gsub("\\)","\\\\)",re[i])
                                re[i] <- gsub("\\|","\\\\|",re[i])
                                self$formula <- gsub(paste0("\\+",re[i]),"",self$formula)
                              }
                            }
                            
                            ## add handling of factors
                            if(grepl("factor[^ \\[]+[ \\s\\+]",self$formula)){
                              
                              rm_int <- grepl("-1",self$formula)
                              cstart <- ifelse(rm_int,1,2)
                              regres <- gregexpr("factor\\(.*\\)",self$formula)
                              for(i in 1:length(regres[[1]])){
                                tmpstr <- substr(self$formula,regres[[1]][i],regres[[1]][i]+attr(regres[[1]],"match.length")[i]-1)
                                tmpdat <- stats::model.matrix(as.formula(paste0("~",tmpstr,"-1")),data=self$data)
                                f1 <- self$formula
                                for(j in (cstart):ncol(tmpdat)){
                                  cname1 <- colnames(tmpdat)[j]
                                  cname1 <- gsub("\\(","\\[",cname1)
                                  cname1 <- gsub("\\)","\\]",cname1)
                                  colnames(tmpdat)[j] <- cname1
                                  if(j==cstart){
                                    f2 <- cname1
                                  } else {
                                    f2 <- paste0(f2," + ",cname1)
                                  }
                                }
                                self$data <- cbind(self$data,tmpdat[,cstart:ncol(tmpdat)])
                                tmpform <- ""
                                if(grepl("[^ \\s\\~]",substr(f1,1,(regres[[1]][i]-1)))){
                                  tmpform <- paste0(tmpform,substr(f1,1,(regres[[1]][i]-1))," + ",f2)
                                } else {
                                  tmpform <- f2
                                }
                                if((regres[[1]][i]+attr(regres[[1]],"match.length")[i]) <= nchar(self$formula)){
                                  tmpform <- paste0(tmpform, substr(f1,regres[[1]][i]+attr(regres[[1]],"match.length")[i],nchar(f1)))
                                }
                                self$formula <- tmpform
                              }
                            }
                            # change the brackets to avoid confusion
                            
                            self$formula <- gsub("-[ \\s+]1","-1",self$formula)
                            
                            private$genX()
                            private$hash <- private$hash_do()
                          },
                          genX = function(){
                            self$X <- .genX(self$formula,as.matrix(self$data),colnames(self$data))
                            if(is.null(self$parameters))self$parameters <- rep(0,ncol(self$X))
                            cnames <- .x_names(self$formula)
                            if(!grepl("-1",self$formula)) cnames <- c("[Intercept]",cnames)
                            colnames(self$X) <- cnames
                          }
                          
                        ))




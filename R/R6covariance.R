#' R6 Class representing a covariance function and data
#'
#' For the generalised linear mixed model
#'
#' \deqn{Y \sim F(\mu,\sigma)}
#' \deqn{\mu = h^-1(X\beta + Z\gamma)}
#' \deqn{\gamma \sim MVN(0,D)}
#'
#' where h is the link function, this class defines Z and D. The covariance is defined by a covariance function, data, and parameters.
#' A new instance can be generated with $new(). The class will generate the
#' relevant matrices Z and D automatically. See \href{https://github.com/samuel-watson/glmmrBase/blob/master/README.md}{glmmrBase} for a
#' detailed guide on model specification.
#' @export
Covariance <- R6::R6Class("Covariance",
                      public = list(
                        #' @field data Data frame with data required to build covariance
                        data=NULL,
                        #' @field formula Covariance function formula.
                        formula = NULL,
                        #' @field parameters Model parameters specified in order of the functions in the formula.
                        parameters = NULL,
                        #' @field Z Design matrix
                        Z = NULL,
                        #' @field D Covariance matrix of the random effects
                        D = NULL,
                        #' @description
                        #' Return the size of the design
                        #' @return Scalar
                        n= function(){
                          nrow(self$Z)
                        },
                        #' @description
                        #' Create a new Covariance object
                        #' @param formula Formula describing the covariance function. See Details
                        #' @param data (Optional) Data frame with data required for constructing the covariance.
                        #' @param parameters (Optional) Vector with parameter values for the functions in the model
                        #' formula. See Details.
                        #' @param verbose Logical whether to provide detailed output.
                        #' @details
                        #' **Intitialisation**
                        #' A covariance function is specified as an additive formula made up of
                        #' components with structure \code{(1|f(j))}. The left side of the vertical bar
                        #' specifies the covariates in the model that have a random effects structure.
                        #' The right side of the vertical bar specify the covariance function `f` for
                        #' that term using variable named in the data `j`.
                        #' Covariance functions on the right side of the vertical bar are multiplied
                        #' together, i.e. \code{(1|f(j)*g(t))}.
                        #'
                        #' There are several common functions included for a named variable in data \code{x}.
                        #' A non-exhaustive list (see \href{https://github.com/samuel-watson/glmmrBase/blob/master/README.md}{glmmrBase} for a full list):
                        #' * \code{gr(x)}: Indicator function (1 parameter)
                        #' * \code{fexp(x)}: Exponential function (2 parameters)
                        #' * \code{ar1(x)}: AR1 function (1 parameter)
                        #' * \code{sqexp(x)}: Squared exponential (1 parameter)
                        #' * \code{matern(x)}: Matern function (2 parameters)
                        #' * \code{bessel(x)}: Modified Bessel function of the 2nd kind (1 parameter)
                        #'
                        #' Parameters are provided to the covariance function as a vector.
                        #' The parameters in the vector for each function should be provided
                        #' in the order the covariance functions are written are written.
                        #' For example,
                        #' * Formula: `~(1|gr(j))+(1|gr(j*t))`; parameters: `c(0.25,0.1)`
                        #' * Formula: `~(1|gr(j)*fexp(t))`; parameters: `c(0.25,1,0.5)`
                        #' Note that it is also possible to specify a group membership with two
                        #' variable alternatively as `(1|gr(j)*gr(t))`, for example, but this
                        #' will require two parameters to be specified, so it is recommended against.
                        #'
                        #' If not all of `formula`, `data`, and `parameters` are not specified then the linked matrices
                        #' are not calculated. These options can be later specified, or updated via a \link[glmmrBase]{Model} object.
                        #' If these arguments are updated or changed then call `self$check()` to update linked matrices. Updating of
                        #' parameters is automatic if using the `update_parameters()` member function.
                        #' @return A Covariance object
                        #' @examples
                        #' df <- nelder(~(cl(5)*t(5)) > ind(5))
                        #' cov <- Covariance$new(formula = ~(1|gr(cl)*ar1(t)),
                        #'                       parameters = c(0.25,0.7),
                        #'                       data= df)
                        initialize = function(formula,
                                              data = NULL,
                                              parameters= NULL,
                                              verbose=TRUE){
                          if(missing(formula))stop("formula required.")
                          self$formula = Reduce(paste0,as.character(formula))
                          allset <- TRUE
                          if(!is.null(data)){
                            self$data <- data
                          } else {
                            allset <- FALSE
                          }

                          if(!is.null(parameters)){
                            self$parameters <- parameters
                          }

                          if(allset){
                            private$cov_form()
                          } else {
                            private$hash <- digest::digest(1)
                          }

                        },
                        #' @description
                        #' Check if anything has changed and update matrices if so.
                        #' @param verbose Logical whether to report if any changes detected.
                        #' @return NULL
                        #' @examples
                        #' df <- nelder(~(cl(5)*t(5)) > ind(5))
                        #' cov <- Covariance$new(formula = ~(1|gr(cl)*ar1(t)),
                        #'                       parameters = c(0.15,0.8),
                        #'                       data= df)
                        #' cov$parameters <- c(0.25,0.1)
                        #' cov$check(verbose=FALSE)
                        check = function(verbose=TRUE){
                          new_hash <- private$hash_do()
                          if(private$hash != new_hash){
                            private$cov_form()
                            if(verbose)message(paste0("Generating the ",nrow(self$Z)," x ",ncol(self$Z)," matrix Z"))
                            if(verbose)message(paste0("Generating the ",nrow(self$D)," x ",ncol(self$D)," matrix D"))
                            private$genD()
                          } else {
                            if(verbose)message("Covariance up to date")
                          }
                          invisible(self)
                        },
                        #' @description
                        #' Updates the covariance parameters
                        #'
                        #' @details
                        #' Using `update_parameters()` is the preferred way of updating the parameters of the
                        #' mean or covariance objects as opposed to direct assignment, e.g. `self$parameters <- c(...)`.
                        #' The function calls check functions to automatically update linked matrices with the new parameters.
                        #' If using direct assignment, call `self$check()` afterwards.
                        #'
                        #' @param parameters A vector of parameters for the covariance function(s). See Details.
                        update_parameters = function(parameters){
                          self$parameters <- parameters
                          .Covariance__Update_parameters(private$ptr,parameters)
                          self$check(FALSE)
                        },
                        #' @description
                        #' Show details of Covariance object
                        #' @param ... ignored
                        #' @examples
                        #' df <- nelder(~(cl(5)*t(5)) > ind(5))
                        #' Covariance$new(formula = ~(1|gr(cl)*ar1(t)),
                        #'                       parameters = c(0.05,0.8),
                        #'                       data= df)
                        print = function(){
                          re <- .re_names(self$formula)
                          cat("\U2BC8 Covariance")
                          cat("\n   \U2BA1 Terms:",re)
                          cat("\n   \U2BA1 Parameters: ",self$parameters)
                        },
                        #' @description
                        #' Keep specified indices and removes the rest
                        #' @param index vector of indices to keep
                        #' @examples
                        #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                        #' cov <- Covariance$new(formula = ~(1|gr(cl)*ar1(t)),
                        #'                       parameters = c(0.05,0.8),
                        #'                       data= df)
                        #' cov$subset(1:100)
                        subset = function(index){
                          self$data <- self$data[index,]
                          self$check()
                        },
                        #' @description
                        #' Returns the Cholesky decomposition of the covariance matrix D
                        #' @return A matrix
                        get_chol_D = function(){
                          return(Matrix::Matrix(.Covariance__D_chol(private$ptr)))
                        },
                        #' @description
                        #' The function returns the values of the multivariate Gaussian log likelihood
                        #' with mean zero and covariance D for a given vector of random effect terms.
                        #' @param u Vector of random effects
                        #' @return Value of the log likelihood
                        log_likelihood = function(u){
                          Q <- .Covariance__Q(private$ptr)
                          if(length(u)!=Q)stop("Vector not equal to number of random effects")
                          loglik <- .Covariance__log_likelihood(private$ptr,u)
                          return(loglik)
                        },
                        #' @description
                        #' Simulates a set of random effects from the multivariate Gaussian distribution
                        #' with mean zero and covariance D.
                        #' @return A vector of random effect values
                        simulate_re = function(){
                          re <- .Covariance__simulate_re(private$ptr)
                          return(re)
                        },
                        #' @description
                        #' If this function is called then sparse matrix methods will be used for calculations
                        #' involving D
                        #' @return None. Called for effects.
                        sparse = function(sparse = TRUE){
                          if(sparse){
                            .Covariance__make_sparse(private$ptr)
                          } else {
                            .Covariance__make_dense(private$ptr)
                          }
                        },
                        #' @description 
                        #' Returns a table showing which parameters are members of which covariance 
                        #' function term.
                        #' @return A data frame
                        parameter_table = function(){
                          re <- .Covariance__re_terms(private$ptr)
                          paridx <- .Covariance__parameter_fn_index(private$ptr)+1
                          recount <- .Covariance__re_count(private$ptr)
                          partable <- data.frame(id = paridx, term = re[paridx], parameter = self$parameters,count = recount[paridx])
                          return(partable)
                        }
                      ),
                      private = list(
                        hash = NULL,
                        hash_do = function(){
                          c(digest::digest(c(self$data,self$formula,self$parameters)))
                        },
                        parcount = NULL,
                        ptr = NULL,
                        cov_form = function(){
                          self$formula <- gsub("\\s","",self$formula)
                          self$formula <- gsub("~","",self$formula)
                          private$ptr <- .Covariance__new(self$formula,
                                                         as.matrix(self$data),
                                                         colnames(self$data))
                          private$parcount <- .Covariance__n_cov_pars(private$ptr)
                          if(is.null(self$parameters))self$parameters <- rep(0.5,private$parcount)
                          .Covariance__Update_parameters(private$ptr,self$parameters)
                          private$genD()
                          self$Z <- .Covariance__Z(private$ptr)
                        },
                        genD = function(update=TRUE){
                          if(private$parcount != length(self$parameters))stop(paste0("Wrong number of parameters for covariance function(s). "))
                          if(.Covariance__any_gr(private$ptr)).Covariance__make_sparse(private$ptr)
                          D <- .Covariance__D(private$ptr)
                          if(update){
                            self$D <- Matrix::Matrix(D)
                            private$hash <- private$hash_do()
                          } else {
                            return(D)
                          }

                        }
                      ))


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
                        #' * \code{ar(x)}: AR function (2 parameters)
                        #' * \code{sqexp(x)}: Squared exponential (1 parameter)
                        #' * \code{matern(x)}: Matern function (2 parameters)
                        #' * \code{bessel(x)}: Modified Bessel function of the 2nd kind (1 parameter)
                        #' For many 2 parameter functions, such as `ar` and `fexp`, alternative one parameter 
                        #' versions are also available as `ar0` and `fexp0`. These function omit the variance 
                        #' parameter and so can be used in combination with `gr` functions such as `gr(j)*ar0(t)`.
                        #'
                        #' Parameters are provided to the covariance function as a vector.
                        #' The parameters in the vector for each function should be provided
                        #' in the order the covariance functions are written are written.
                        #' For example,
                        #' * Formula: `~(1|gr(j))+(1|gr(j*t))`; parameters: `c(0.05,0.01)`
                        #' * Formula: `~(1|gr(j)*fexp0(t))`; parameters: `c(0.05,0.5)`
                        #'
                        #' Updating of parameters is automatic if using the `update_parameters()` member function.
                        #' @return A Covariance object
                        #' @examples
                        #' \dontshow{
                        #' setParallel(FALSE) # for the CRAN check
                        #' }
                        #' df <- nelder(~(cl(5)*t(5)) > ind(5))
                        #' cov <- Covariance$new(formula = ~(1|gr(cl)*ar0(t)),
                        #'                       parameters = c(0.05,0.7),
                        #'                       data= df)
                        initialize = function(formula,
                                              data = NULL,
                                              parameters= NULL){
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
                          if(allset) private$cov_form()
                        },
                        #' @description
                        #' Updates the covariance parameters
                        #'
                        #' @details
                        #' Using `update_parameters()` is the preferred way of updating the parameters of the
                        #' mean or covariance objects as opposed to direct assignment, e.g. `self$parameters <- c(...)`.
                        #' The function calls check functions to automatically update linked matrices with the new parameters.
                        #'
                        #' @param parameters A vector of parameters for the covariance function(s). See Details.
                        update_parameters = function(parameters){
                          if(is.null(private$ptr)){
                            if(is.null(self$data))stop("No data")
                            private$cov_form()
                          }
                          self$parameters <- parameters
                          if(is.null(private$model_ptr)){
                            Covariance__Update_parameters(private$ptr,parameters,private$type)
                            self$D <- Matrix::Matrix(Covariance__D(private$ptr,private$type))
                          } else {
                            Model__update_theta(private$model_ptr,parameters,private$type) 
                            self$D <- Matrix::Matrix(Model__D(private$model_ptr,private$type))
                          }
                        },
                        #' @description
                        #' Show details of Covariance object
                        #' @param ... ignored
                        #' @examples
                        #' \dontshow{
                        #' setParallel(FALSE) # for the CRAN check
                        #' }
                        #' df <- nelder(~(cl(5)*t(5)) > ind(5))
                        #' Covariance$new(formula = ~(1|gr(cl)*ar0(t)),
                        #'                       parameters = c(0.05,0.8),
                        #'                       data= df)
                        print = function(){
                          re <- re_names(self$formula)
                          cat("\U2BC8 Covariance")
                          cat("\n   \U2BA1 Terms:",re)
                          if(private$type == 1)cat(" (NNGP)")
                          if(private$type == 2)cat(" (HSGP)")
                          cat("\n   \U2BA1 Parameters: ",self$parameters)
                          cat("\n")
                        },
                        #' @description
                        #' Keep specified indices and removes the rest
                        #' @param index vector of indices to keep
                        #' @examples
                        #' \dontshow{
                        #' setParallel(FALSE) # for the CRAN check
                        #' }
                        #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                        #' cov <- Covariance$new(formula = ~(1|gr(cl)*ar0(t)),
                        #'                       parameters = c(0.05,0.8),
                        #'                       data= df)
                        #' cov$subset(1:100)
                        subset = function(index){
                          self$data <- self$data[index,]
                          if(is.null(private$model_ptr))private$cov_form()
                        },
                        #' @description
                        #' Returns the Cholesky decomposition of the covariance matrix D
                        #' @return A matrix
                        chol_D = function(){
                          if(is.null(private$model_ptr)){
                            return(Matrix::Matrix(Covariance__D_chol(private$ptr,private$type)))
                          } else {
                            return(Matrix::Matrix(Model__D_chol(private$model_ptr,private$type)))
                          }
                        },
                        #' @description
                        #' The function returns the values of the multivariate Gaussian log likelihood
                        #' with mean zero and covariance D for a given vector of random effect terms.
                        #' @param u Vector of random effects
                        #' @return Value of the log likelihood
                        log_likelihood = function(u){
                          if(is.null(private$model_ptr)){
                            Q <- Covariance__Q(private$ptr,private$type)
                            if(length(u)!=Q)stop("Vector not equal to number of random effects")
                            loglik <- Covariance__log_likelihood(private$ptr,u,private$type)
                          } else {
                            Q <- Model__Q(private$model_ptr,private$type)
                            if(length(u)!=Q)stop("Vector not equal to number of random effects")
                            loglik <- Model__u_log_likelihood(private$model_ptr,u,private$type)
                          }
                          return(loglik)
                        },
                        #' @description
                        #' Simulates a set of random effects from the multivariate Gaussian distribution
                        #' with mean zero and covariance D.
                        #' @return A vector of random effect values
                        simulate_re = function(){
                          if(is.null(private$model_ptr)){
                            re <- Covariance__simulate_re(private$ptr,private$type)
                          } else {
                            re <- Model__simulate_re(private$model_ptr,private$type)
                          }
                          return(re)
                        },
                        #' @description
                        #' If this function is called then sparse matrix methods will be used for calculations involving D
                        #' @param sparse Logical. Whether to use sparse methods (TRUE) or not (FALSE)
                        #' @param amd Logical indicating whether to use and Approximate Minimum Degree algorithm to calculate an efficient permutation matrix so 
                        #' that the Cholesky decomposition of PAP^T is calculated rather than A.
                        #' @return None. Called for effects.
                        sparse = function(sparse = TRUE, amd = TRUE){
                          if(sparse){
                            if(is.null(private$model_ptr)){
                              Covariance__make_sparse(private$ptr,amd,private$type)
                            } else {
                              Model__make_sparse(private$model_ptr,amd,private$type)
                            }
                          } else {
                            if(is.null(private$model_ptr)){
                              Covariance__make_dense(private$ptr,private$type)
                            } else {
                              Model__make_dense(private$model_ptr,private$type)
                            }
                          }
                        },
                        #' @description
                        #' Returns a table showing which parameters are members of which covariance
                        #' function term.
                        #' @return A data frame
                        parameter_table = function(){
                          if(is.null(private$model_ptr)){
                            re <- Covariance__re_terms(private$ptr,private$type)
                            paridx <- Covariance__parameter_fn_index(private$ptr,private$type)+1
                            recount <- Covariance__re_count(private$ptr,private$type)
                          } else {
                            re <- Model__re_terms(private$model_ptr,private$type)
                            paridx <- Model__parameter_fn_index(private$model_ptr,private$type)+1
                            recount <- Model__re_count(private$model_ptr,private$type)
                          }
                          partable <- data.frame(id = paridx, term = re[paridx], parameter = self$parameters,count = recount[paridx])
                          return(partable)
                        },
                        #' @description 
                        #' Reports or sets the parameters for the nearest neighbour Gaussian process
                        #' @param nn Integer. Number of nearest neighbours. Optional - leave as NULL to return
                        #' details of the NNGP instead.
                        #' @return If `nn` is NULL then the function will either return FALSE if not using a 
                        #' Nearest neighbour approximation, or TRUE and the number of nearest neighbours, otherwise
                        #' it will return nothing.
                        nngp = function(nn = NULL){
                          if(!is.null(nn)){
                            if(!is(nn,"numeric") || nn%%1 != 0 || nn <= 0)stop("nn must be a positive integer")
                            private$nn <- nn
                          } else {
                            if(private$type == 1){
                              return(c(TRUE,private$nn))
                            } else {
                              return(c(FALSE))
                            }
                          }
                        },
                        #' @description 
                        #' Reports or sets the parameters for the Hilbert Space Gaussian process
                        #' @param m Integer or vector of integers. Number of basis functions per dimension. If only
                        #' a single number is provided and there is more than one dimension the same number will be applied
                        #' to all dimensions.
                        #' @param L Decimal. The boundary extension.
                        #' @return If `m` and `L` are NULL then the function will either return FALSE if not using a 
                        #' Hilbert space approximation, or TRUE and the number of bases functions and boundary value, otherwise
                        #' it will return nothing.
                        hsgp = function(m = NULL, L = NULL){
                          if(private$type == 2){
                            dim <- Model_hsgp__dim(private$model_ptr)
                            if(!is.null(m)){
                              if(length(m)==1 & dim > 1){
                                private$m <- rep(m,dim)
                              } else if(length(m) > 1 & dim > 1){
                                if(length(m)==dim){
                                  private$m <- m
                                } else (
                                  stop("m wrong dimension")
                                )
                              } else if(length(m) == 1 & dim == 1){
                                private$m <- m
                              }
                            }
                            if(!is.null(L)){
                              if(length(L)==1 & dim > 1){
                                private$L <- rep(L,dim)
                              } else if(length(L) > 1 & dim > 1){
                                if(length(L)==dim){
                                  private$L <- L
                                } else (
                                  stop("m wrong dimension")
                                )
                              } else if(length(m) == 1 & dim == 1){
                                private$L <- L
                              }
                            }
                            if(is.null(m) & is.null(L)){
                              return(c(TRUE,private$m,private$L))
                            } else {
                              if(is.null(private$model_ptr)){
                                Covariance_hsgp__set_approx_pars(private$ptr,private$m,private$L)
                              } else {
                                Model_hsgp__set_approx_pars(private$model_ptr,private$m,private$L)
                              }
                            }
                          } else {
                            return(FALSE)
                          }
                        }
                      ),
                      private = list(
                        parcount = NULL,
                        ptr = NULL,
                        model_ptr = NULL,
                        type = 0,
                        nn = 10,
                        m = 10,
                        L = 1.5,
                        cov_form = function(){
                          self$formula <- gsub("\\s","",self$formula)
                          self$formula <- gsub("~","",self$formula)
                          re <- re_names(self$formula)
                          if(any(sapply(re,function(i)grepl("nngp",i)))){
                            if(length(re)>1)stop("NNGP only available as a single covariance function currently.")
                            private$type <- 1
                            re[1] <- gsub("nngp_","",re[1])
                          } else if(any(sapply(re,function(i)grepl("hsgp",i)))){
                            if(length(re)>1)stop("HSGP only available as a single covariance function currently.")
                            private$type <- 2
                            re[1] <- gsub("hsgp_","",re[1])
                          }
                          self$formula <- re[1]
                          if(length(re)>1){
                            for(i in 2:length(re)){
                              self$formula <- paste0(self$formula,"+",re[i])
                            }
                          }
                          
                          if(private$type == 0){
                            private$ptr <- Covariance__new(self$formula,
                                                           as.matrix(self$data),
                                                           colnames(self$data))
                          } else if(private$type == 1){
                            private$ptr <- Covariance_nngp__new(self$formula,
                                                           as.matrix(self$data),
                                                           colnames(self$data))
                            Covariance__set_nn(private$ptr,private$nn)
                          } else if(private$type==2){
                            private$ptr <- Covariance_hsgp__new(self$formula,
                                                                as.matrix(self$data),
                                                                colnames(self$data))
                          }
                          
                          private$parcount <- Covariance__n_cov_pars(private$ptr,private$type)
                          if(is.null(self$parameters))self$parameters <- runif(private$parcount,0,1)
                          Covariance__Update_parameters(private$ptr,self$parameters,private$type)
                          private$genD()
                          self$Z <- Covariance__Z(private$ptr,private$type)
                          if(private$type==2){
                            dim <- Model_hsgp__dim(private$ptr)
                            private$m <- rep(10,dim)
                            private$L <- rep(1.5,dim)
                          }
                        },
                        genD = function(update=TRUE){
                          if(private$parcount != length(self$parameters))stop(paste0("Wrong number of parameters for covariance function(s). "))
                          if(private$type == 0 & Covariance__any_gr(private$ptr))Covariance__make_sparse(private$ptr)
                          D <- Covariance__D(private$ptr,private$type)
                          if(update){
                            self$D <- Matrix::Matrix(D)
                          } else {
                            return(D)
                          }

                        }
                      ))


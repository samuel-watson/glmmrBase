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
                        #' @field eff_range The effective range of covariance functions, specified in order of the functions in the formula. Only 
                        #' the functions with compact support require effective range parameters. 
                        eff_range = NULL,
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
                        #' @param eff_range (Optional) Vector with the effective range parameter for covariance
                        #' functions that require it, i.e. those with compact support.
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
                                              eff_range = NULL,
                                              verbose=FALSE){
                          # if(any(is.null(data),is.null(formula),is.null(parameters))){
                          #   message("not all attributes set. call check() when all attributes set.")
                          # } else {
                          #   self$data = data
                          #   self$formula = as.formula(formula, env=.GlobalEnv)
                          #   self$parameters = parameters
                          #   self$eff_range = eff_range
                          #   private$cov_form()
                          # }
                          if(missing(formula))stop("formula required.")
                          self$formula = as.formula(formula, env=.GlobalEnv)
                          allset <- TRUE
                          if(!is.null(data)){
                            self$data <- data 
                          } else {
                            allset <- FALSE
                          }
                          
                          if(!is.null(parameters)){
                            self$parameters <- parameters
                          }
                          
                          self$eff_range = eff_range
                          
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
                          if(private$hash[1] != new_hash[1]){
                            if(verbose)message("changes found, updating Z")
                            private$cov_form()
                          } else if(private$hash[2] != new_hash[2]){
                            if(verbose)message("changes found, updating D")
                            private$genD()
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
                        #' @param verbose Logical indicating whether to provide more detailed feedback
                        update_parameters = function(parameters,
                                                     verbose = FALSE){
                          self$parameters <- parameters
                          self$check(verbose)
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
                          # MAKE CLEARER ABOUT FUNCTIONS AND PARAMETERS?

                          cat("Covariance\n")
                          print(self$formula)
                          cat("Parameters:")
                          print(unlist(self$parameters))
                          #print(head(self$data))
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
                        #' Generate a new D matrix
                        #' 
                        #' D is the covariance matrix of the random effects terms in the generalised linear mixed
                        #' model. This function will return a matrix D for a given set of parameters.
                        #' @param parameters list of lists, see `initialize()`
                        #' @return matrix 
                        #' @examples 
                        #' df <- nelder(~(cl(10)*t(5)) > ind(10))
                        #' cov <- Covariance$new(formula = ~(1|gr(cl)*ar1(t)),
                        #'                       parameters = c(0.05,0.8),
                        #'                       data= df)
                        #' cov$sampleD(c(0.01,0.1))
                        sampleD = function(parameters){
                          return(private$genD(update = FALSE,
                                              new_pars = parameters))
                        },
                        #' @description 
                        #' Returns the list specifying the covariance matrix D
                        #' @return A list
                        get_D_data = function(){
                          return(private$D_data)
                        },
                        #' @description 
                        #' Returns the Cholesky decomposition of the covariance matrix D
                        #' @param parameters (Optional) Vector of parameters, if specified then the Cholesky
                        #' factor is calculated with these parameter values rather than the ones stored in the
                        #' object.
                        #' @return A list of matrices
                        get_chol_D = function(parameters=NULL){
                          if(is.null(parameters)){
                            L = do.call(genCholD,append(private$D_data,list(eff_range = self$eff_range, gamma=self$parameters)))#append(private$D_data,list(gamma=self$parameters))
                          } else {
                            L = do.call(genCholD,append(private$D_data,list(eff_range = self$eff_range, gamma=parameters)))
                          }
                          return(L)
                        }
                      ),
                      private = list(
                        D_data = NULL,
                        flist = NULL,
                        flistlabs = NULL,
                        flistvars = NULL,
                        Zlist = NULL,
                        hash = NULL,
                        hash_do = function(){
                          c(digest::digest(c(self$data)),digest::digest(c(self$formula,self$parameters)))
                        },
                        cov_form = function(){
                          #1. split into independent components that can be combined in block diagonal form
                          flist <- list()
                          flistvars <- list()
                          formExt <- TRUE
                          count <- 0
                          form0 <- self$formula[[2]]
                          while(formExt){
                            count <- count + 1
                            formExt <- I(form0[[1]]=="+")
                            if(formExt){
                              flist[[count]] <- form0[[3]][[2]]
                              form0 <- form0[[2]]
                            } else {
                              flist[[count]] <- form0[[2]]
                            }
                            if("+"%in%all.names(flist[[count]][[3]]))stop("covariance only products")
                            rhsvars <- all.vars(flist[[count]][[3]])
                            funlist <- list()
                            rhsvargroup <- rep(NA,length(rhsvars))
                            formMult <- TRUE
                            countMult <- 0
                            form3 <- flist[[count]][[3]]
                            while(formMult){
                              countMult <- countMult + 1
                              formMult <- I(form3[[1]] == "*")
                              if(formMult){
                                funlist[[countMult]] <- form3[[3]][[1]]
                                rhsvargroup[match(all.vars(form3[[3]]),rhsvars)] <- countMult
                                form3 <- form3[[2]]
                              } else {
                                funlist[[countMult]] <- form3[[1]]
                                rhsvargroup[match(all.vars(form3),rhsvars)] <- countMult
                              }

                            }
                            flistvars[[count]] <- list(lhs=all.vars(flist[[count]][[2]]),
                                                       rhs = rhsvars,
                                                       funs = funlist,
                                                       groups = rhsvargroup)

                          }

                          # build each Z matrix and cbind
                          Zlist <- list()
                          Distlist <- list()
                          flistcount <- list()
                          flistlabs <- list()
                          
                          for(i in 1:length(flist)){
                            data_nodup <- self$data[!duplicated(self$data[,flistvars[[i]]$rhs]),flistvars[[i]]$rhs]
                            if(!is(data_nodup,"data.frame")){
                              data_nodup <- data.frame(data_nodup)
                              colnames(data_nodup) <- flistvars[[i]]$rhs
                            }
                            flistcount[[i]] <- nrow(data_nodup)
                            data_nodup_lab <- data_nodup
                            for(k in 1:ncol(data_nodup_lab))data_nodup_lab[,k] <- paste0(colnames(data_nodup_lab)[k],data_nodup_lab[,k])
                            flistlabs[[i]] <- apply(data_nodup_lab,1,paste0,collapse="")
                            Distlist[[i]] <- as.matrix(data_nodup)
                            zdim2 <- nrow(data_nodup)
                            Zlist[[i]] <- match_rows(self$data,data_nodup,by=flistvars[[i]]$rhs)
                            if(length(flistvars[[i]]$lhs)>0){
                              ZlistNew <- list()
                              for(j in 1:length(flistvars[[i]]$lhs)){
                                ZlistNew[[j]] <- Zlist[[i]]*self$data[,flistvars[[i]]$lhs[j]]
                              }
                              Zlist[[i]] <- Reduce(cbind,ZlistNew)
                            }
                          }
                          
                          fl <- rev(flistvars)
                          fnames <- c("gr","fexp0","ar1","sqexp","matern","bessel","wend0","wend1","wend2","prodwm",
                                      "prodcb","prodek","fexp","sqexp0")
                          fnpar <- c(1,1,1,2,2,1,2,2,2,2,2,2,2,1)
                          parcount <- 0
                          Funclist <- list()
                          Distlist <- rev(Distlist)
                          
                          # v0.2 update to new data system
                          D_data <- list()
                          B <- length(flist)
                          
                          N_dim <- unlist(rev(flistcount))
                          N_func <- unlist(lapply(fl,function(x)length(x$funs)))
                          D_data$cov <- rbind(rep(1:B,N_func),rep(N_dim,N_func),rep(0,sum(N_func)),rep(0,sum(N_func)),rep(0,sum(N_func)))
                          D_data$data <- c()
                          fvar <- lapply(rev(flistvars),function(x)x$groups)
                          for(b in 1:B){
                            D_data$cov[3,D_data$cov[1,]==b] <- match(unlist(rev(fl[[b]]$funs)),fnames) # function definition
                            D_data$cov[4,D_data$cov[1,]==b] <- rev(unname(table(fvar[[b]]))) #number of variables
                            D_data$cov[5,D_data$cov[1,]==b] <- fnpar[D_data$cov[3,D_data$cov[1,]==b]] # number of parameters
                          }
                          D_data$cov[5, ] <- cumsum(D_data$cov[5,]) - 1
                          D_data$data <- Reduce(append,lapply(Distlist,as.vector))
                          
                          # split into sub blocks
                          for(b in 1:B){
                            if(any(D_data$cov[3,D_data$cov[1,]==b] == 1)&!all(D_data$cov[3,D_data$cov[1,]==b] == 1)){
                              #col1 <- which(D_data$cov[3,D_data$cov[1,]==b]==1)
                              col1 <- which(D_data$cov[1,]==b & D_data$cov[3,] == 1)
                             #get range of data 
                              nvar <- D_data$cov[2,]*D_data$cov[4,]
                              nfunc <- ncol(D_data$cov[,D_data$cov[1,]==b]) 
                              dstart <- 1
                              if(col1 > 1){
                                dstart <- dstart + nvar[1:(col1-1)]
                              }
                              dend <- dstart + nvar[col1] - 1
                              dend2 <- dstart + sum(nvar[D_data$cov[1,]==b]) - 1
                              tabgr <- as.data.frame(table(D_data$data[dstart:dend]))
                              #duplicate columns
                              newcov <- D_data$cov[,D_data$cov[1,]==b ,drop=FALSE]
                              newcov <- newcov[,rep(1:ncol(newcov),nrow(tabgr))]
                              newcov[2,] <- tabgr$Freq
                              newcov[1,] <- rep(1:nrow(tabgr),each=nfunc)
                              D_data$cov[1,D_data$cov[1,]>b] <-  D_data$cov[1,D_data$cov[1,]>b] + nrow(tabgr) - 1
                              D_data$cov <- matrix(c(as.vector(D_data$cov[D_data$cov[1,]>b,]),
                                                   as.vector(newcov),
                                                   as.vector(D_data$cov[D_data$cov[1,]>b,])),nrow=5)
                              
                              #reorder data
                              #if multiple variables reorder
                              dat <- matrix(D_data$data[dstart:dend2],nrow=sum(tabgr$Freq))
                              newdat <- c()
                              idx <- 1
                              for(i in 1:nrow(tabgr)){
                                newdat <- c(newdat,as.vector(dat[idx:(idx+tabgr$Freq[i]-1),])) 
                                idx <- idx + tabgr$Freq[i]
                              }
                              D_data$data[dstart:dend2] <- newdat
                              
                            }
                          }
                          
                          D_data$cov[1,] = D_data$cov[1,] - 1
                          D_data$cov <- t(D_data$cov)
                          
                          if(is.null(self$eff_range))self$eff_range <- rep(0,nrow(D_data$cov))
                          
                          Z <- Reduce(cbind,rev(Zlist))
                          Z <- Matrix::Matrix(Z)
                          if(nrow(Z) < ncol(Z))warning("More random effects than observations")
                          self$Z <- Z
                          private$D_data <- D_data
                          for(i in 1:length(Zlist))Zlist[[i]] <- Matrix::Matrix(Zlist[[i]])
                          private$Zlist <- Zlist
                          private$flist <- flist
                          private$flistlabs <- flistlabs
                          private$genD()
                          private$flistvars <- flistvars

                        },
                        genD = function(update=TRUE,
                                        new_pars=NULL){
                          
                          # calculate number of parameters
                          fnpar <- c(1,1,1,2,2,1,2,2,2,2,2,2,2,1)
                          idmax <- which.max(private$D_data$cov[,5])[1]
                          par_count <- private$D_data$cov[idmax,5] + fnpar[private$D_data$cov[idmax,3]]
                          if(is.null(self$parameters))self$parameters <- rep(0.5,par_count)
                          if(par_count != length(self$parameters))stop(paste0("Wrong number of parameters for covariance function(s). "))
                          
                          
                          D <- do.call(genD,append(private$D_data,list(eff_range = self$eff_range,gamma=self$parameters)))#list(private$D_data,gamma=self$parameters)
                          #D <- blockMat(D)
                          if(update){
                            self$D <- Matrix::Matrix(D)
                            private$hash <- private$hash_do()
                          } else {
                            return(D)
                          }

                        }
                      ))


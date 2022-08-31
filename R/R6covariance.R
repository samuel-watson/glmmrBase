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
#' relevant matrices Z and D automatically.  
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
                        #' @param data Data frame with data required for constructing the covariance.
                        #' @param parameters List of lists with parameter values for the functions in the model
                        #' formula. See Details.
                        #' @param verbose Logical whether to provide detailed output.
                        #' @details 
                        #' **Intitialisation**
                        #' A covariance function is specified as an additive formula made up of 
                        #' components with structure \code{(1|f(j))}. The left side of the vertical bar 
                        #' specifies the covariates in the model that have a random effects structure. 
                        #' The right side of the vertical bar specify the covariance function `f` for 
                        #' that term using variable named in the data `j`. If there are multiple 
                        #' covariates on the left side, it is assumed their random effects are 
                        #' correlated, e.g. \code{(1+x|f(j))}. Additive functions are assumed to be 
                        #' independent, for example, \code{(1|f(j))+(x|f(j))} would create random effects 
                        #' with zero correlation for the intercept and the parameter on covariate \code{x}. 
                        #' Covariance functions on the right side of the vertical bar are multiplied 
                        #' together, i.e. \code{(1|f(j)*g(t))}. 
                        #' 
                        #' There are several common functions included for a named variable in data \code{x}:
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
                        #' @return A Covariance object
                        #' @examples 
                        #' df <- nelder(~(cl(5)*t(5)) > ind(5))
                        #' cov <- Covariance$new(formula = ~(1|gr(cl)*ar1(t)),
                        #'                       parameters = c(0.25,0.7),
                        #'                       data= df)
                        initialize = function(formula=NULL,
                                              data = NULL,
                                              parameters= NULL,
                                              eff_range = NULL,
                                              verbose=TRUE){
                          if(any(is.null(data),is.null(formula),is.null(parameters))){
                            cat("not all attributes set. call check() when all attributes set.")
                          } else {
                            self$data = data
                            self$formula = as.formula(formula, env=.GlobalEnv)
                            self$parameters = parameters
                            self$eff_range = eff_range
                            private$cov_form()
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
                        #' @param parameters list of lists, see initialize()
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
                        #' @return A list of matrices
                        get_chol_D = function(parameters=NULL){
                          if(is.null(parameters)){
                            L = do.call(genCholD,list(private$D_data,gamma=self$parameters))#append(private$D_data,list(gamma=self$parameters))
                          } else {
                            L = do.call(genCholD,list(private$D_data,gamma=parameters))
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
                          D_data <- list(B = length(flist))
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
                                ZlistNew[[j]] <- Zlist[[i]]*df[,flistvars[[i]]$lhs[j]]
                              }
                              Zlist[[i]] <- Reduce(cbind,ZlistNew)
                            }
                          }
                          
                          fl <- rev(flistvars)
                          fnames <- c("gr","fexp0","ar1","sqexp","matern","bessel","wend0","wend1","wend2","prodwm",
                                      "prodcb","prodek","fexp","sqexp0")
                          fnpar <- c(1,1,1,2,2,1,1,1,1,1,1,1,2,1)
                          parcount <- 0
                          Funclist <- list()
                          Distlist <- rev(Distlist)
                          #I'm certain this can be neater and better!
                          D_data$N_dim <- unlist(rev(flistcount))
                          D_data$N_func <- unlist(lapply(fl,function(x)length(x$funs)))
                          D_data$func_def <- matrix(0,nrow=D_data$B,ncol=max(D_data$N_func))
                          for(b in 1:D_data$B)D_data$func_def[b,1:D_data$N_func[b]] <- match(unlist(rev(fl[[b]]$funs)),fnames)
                          fvar <- lapply(rev(flistvars),function(x)x$groups)
                          nvar <- list()
                          for(b in 1:D_data$B){
                            nvar[[b]] <- rev(unname(table(fvar[[b]])))
                          }
                          D_data$N_var_func <- matrix(0,ncol=max(D_data$N_func),nrow=D_data$B)
                          for(b in 1:D_data$B)D_data$N_var_func[b,1:D_data$N_func[b]] <- nvar[[b]]
                          D_data$eff_range <- matrix(0,ncol=max(D_data$N_func),nrow=D_data$B)
                          nfcount <- 1
                          if(!is.null(self$eff_range)){
                            for(b in 1:D_data$B){
                              for(k in 1:D_data$N_func[b]){
                                if(D_data$func_def[b,k]%in%7:12){
                                  D_data$eff_range[b,k] <- self$eff_range[nfcount]
                                }
                              }
                              nfcount <- nfcount + 1
                            }
                          }
                          D_data$col_id <- array(0,dim=c(max(D_data$N_func),max(D_data$N_var_func),D_data$B))
                          for(b in 1:D_data$B){
                            for(k in 1:D_data$N_func[b]){
                              vnames <- rev(flistvars)[[b]]
                              vnames <- vnames$rhs[vnames$groups==(D_data$N_func[b]+1-k)]
                              D_data$col_id[k,1:D_data$N_var_func[b,k],b] <- match(vnames,colnames(Distlist[[b]]))
                            }
                          }
                          D_data$N_par <- matrix(0,nrow=D_data$B,ncol=max(D_data$N_func))
                          for(b in 1:D_data$B)for(k in 1:D_data$N_func[b])D_data$N_par[b,k] <- fnpar[D_data$func_def[b,k]]
                          #D_data$sum_N_par <- sum(D_data$N_par)
                          D_data$N_par <- matrix(cumsum(t(D_data$N_par))-1,nrow=D_data$B,ncol=max(D_data$N_func))
                          D_data$cov_data <- array(0,dim=c(max(D_data$N_dim),max(rowSums(D_data$N_var_func)),D_data$B))
                          for(b in 1:D_data$B) D_data$cov_data[1:D_data$N_dim[b],1:ncol(Distlist[[b]]),b] <- Distlist[[b]]
                          #split group blocks further
                          for(b in 1:D_data$B){
                            if(any(D_data$func_def[b,1:D_data$N_func[b]] == 1)&!all(D_data$func_def[b,1:D_data$N_func[b]] == 1)){
                              col1 <- which(D_data$func_def[b,]==1)
                              colid1 <- D_data$col_id[col1,1:D_data$N_var_func[col1,b],b]
                              tabgr <- as.data.frame(table(D_data$cov_data[1:D_data$N_dim[b],colid1,b]))
                              ids <- D_data$cov_data[1:D_data$N_dim[b],colid1,b,drop=FALSE]
                              D_data$N_dim <- D_data$N_dim[-b]
                              D_data$N_dim <- append(D_data$N_dim,tabgr$Freq,b-1)
                              nf <- D_data$N_func[b]
                              D_data$N_func <- D_data$N_func[-b]
                              D_data$N_func <- append(D_data$N_func,rep(nf,nrow(tabgr)),b-1)
                              nf <- D_data$N_var_func[b,]
                              D_data$N_var_func <- rbind(D_data$N_var_func[-b,],t(matrix(nf,ncol=nrow(tabgr),nrow=length(nf))))
                              nf <- D_data$eff_range[b,]
                              D_data$eff_range <- rbind(D_data$eff_range[-b,],t(matrix(nf,ncol=nrow(tabgr),nrow=length(nf))))
                              nf <- D_data$func_def[b,]
                              D_data$func_def <- rbind(D_data$func_def[-b,],t(matrix(nf,ncol=nrow(tabgr),nrow=length(nf))))
                              nf <- D_data$N_par[b,]
                              D_data$N_par <- rbind(D_data$N_par[-b,],t(matrix(nf,ncol=nrow(tabgr),nrow=length(nf))))
                              colidnew <- array(D_data$col_id[,,b],dim = c(dim(D_data$col_id)[1:2],nrow(tabgr)))
                              if(D_data$B==1){
                                D_data$col_id <- array(D_data$col_id[,,b],dim = c(dim(D_data$col_id)[1:2],nrow(tabgr)))
                              } else {
                                if(b==1){
                                  D_data$col_id <- array(c(rep(D_data$col_id[,,b],nrow(tabgr)),D_data$col_id[,,2:D_data$B]),dim = c(dim(D_data$col_id)[1:2],nrow(tabgr)+D_data$B-1))
                                } else if(b>1&b<D_data$B){
                                  D_data$col_id <- array(c(D_data$col_id[,,1:(b-1)],rep(D_data$col_id[,,b],nrow(tabgr)),D_data$col_id[,,(b+1):D_data$B]),dim = c(dim(D_data$col_id)[1:2],nrow(tabgr)+D_data$B-1))
                                } else if(b==D_data$B){
                                  D_data$col_id <- array(c(D_data$col_id[,,1:(b-1)],rep(D_data$col_id[,,b],nrow(tabgr))),dim = c(dim(D_data$col_id)[1:2],nrow(tabgr)+D_data$B-1))
                                }
                              }
                              cov_datanew <- array(0,dim=c(max(D_data$N_dim),max(rowSums(D_data$N_var_func)),D_data$B+ nrow(tabgr)-1))
                              iter <- 0
                              for(i in 1:D_data$B){
                                iter <- iter + 1
                                if(i == b){
                                  #idx <- cumsum(tabgr$Freq)-tabgr[1,'Freq']+1
                                  idsu <- unique(ids)
                                  idx <- match(ids,idsu)
                                  for(j in 1:nrow(tabgr)){
                                    cov_datanew[1:tabgr[j,'Freq'],,iter] <- D_data$cov_data[,,b][which(idx==j),]#[which(ids == idsu[j]),]
                                    iter <- iter + 1
                                  }
                                } else {
                                  cov_datanew[,,iter] <- D_data$cov_data[,,i]
                                }
                              }
                              D_data$cov_data <- cov_datanew
                              D_data$B <- D_data$B + nrow(tabgr) -1
                            }
                          }
                          
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
                          D <- do.call(genD,list(private$D_data,gamma=self$parameters))#append(private$D_data,list(gamma=self$parameters))
                          D <- blockMat(D)
                          if(update){
                            self$D <- Matrix::Matrix(D)
                            private$hash <- private$hash_do()
                          } else {
                            return(D)
                          }

                        }
                      ))


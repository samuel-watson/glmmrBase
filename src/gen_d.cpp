#include <cmath>  
#include <RcppArmadillo.h>
#include "../inst/include/glmmr.h"
using namespace Rcpp;
using namespace arma;

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

//' Generates the covariance matrix of the random effects
//' 
//' Generates the covariance matrix of the random effects from a sparse representation
//' @param B Integer specifying the number of blocks in the matrix
//' @param N_dim Vector of integers, which each value specifying the dimension of each block
//' @param N_func Vector of integers specifying the number of functions in the covariance function 
//' for each block.
//' @param func_def Matrix of integers where each column specifies the function definition for each function in each block. 
//' @param N_var_func Matrix of integers of same size as `func_def` with each column specying the number 
//' of variables in the argument to each function in each block
//' @param col_id 3D array (cube) of integers of dimension length(func_def) x max(N_var_func) x B 
//' where each slice the respective column indexes of `cov_data` for each function in the block
//' @param N_par Matrix of integers of same size as `func_def` with each column specifying the number
//' of parameters in the function in each block
//' @param sum_N_par Total number of parameters
//' @param cov_data 3D array (cube) holding the data for the covariance matrix where each of the B slices
//' is the data required for each block
//' @param gamma Vector of covariance parameters specified in order they appear column wise in the functions 
//' specified by `func_def`
//' @return A symmetric positive definite covariance matrix
// [[Rcpp::export]]
arma::field<arma::mat> genD(Rcpp::List D_data,
                            const arma::vec &gamma){
  
  DMatrix dmat(D_data,gamma);
  arma::field<arma::mat> DBlocks = dmat.genD();
  return(DBlocks);
}

// arma::field<arma::mat> genD(const arma::uword &B,
//                                    const arma::uvec &N_dim,
//                                    const arma::uvec &N_func,
//                                    const arma::umat &func_def,
//                                    const arma::umat &N_var_func,
//                                    const arma::ucube &col_id,
//                                    const arma::umat &N_par,
//                                    const arma::cube &cov_data,
//                                    const arma::vec &gamma){
//   
//   DMatrix dmat(B,N_dim,N_func,func_def,N_var_func,col_id,N_par,cov_data,gamma);
//   arma::field<arma::mat> DBlocks = dmat.genD();
//   return(DBlocks);
// }

//' Generates the cholesky decomposition covariance matrix of the random effects
//' 
//' Generates the covariance matrix of the random effects from a sparse representation
//' @param B Integer specifying the number of blocks in the matrix
//' @param N_dim Vector of integers, which each value specifying the dimension of each block
//' @param N_func Vector of integers specifying the number of functions in the covariance function 
//' for each block.
//' @param func_def Matrix of integers where each column specifies the function definition for each function in each block. 
//' @param N_var_func Matrix of integers of same size as `func_def` with each column specying the number 
//' of variables in the argument to each function in each block
//' @param col_id 3D array (cube) of integers of dimension length(func_def) x max(N_var_func) x B 
//' where each slice the respective column indexes of `cov_data` for each function in the block
//' @param N_par Matrix of integers of same size as `func_def` with each column specifying the number
//' of parameters in the function in each block
//' @param sum_N_par Total number of parameters
//' @param cov_data 3D array (cube) holding the data for the covariance matrix where each of the B slices
//' is the data required for each block
//' @param gamma Vector of covariance parameters specified in order they appear column wise in the functions 
//' specified by `func_def`
//' @return A lower triangular matrix matrix
// [[Rcpp::export]]
arma::field<arma::mat> genCholD(Rcpp::List D_data,
                                const arma::vec &gamma){
  DMatrix dmat(D_data,gamma);
  arma::field<arma::mat> DBlocks = dmat.genCholD();
  return(DBlocks);
}

// arma::field<arma::mat> genCholD(const arma::uword &B,
//                             const arma::uvec &N_dim,
//                             const arma::uvec &N_func,
//                             const arma::umat &func_def,
//                             const arma::umat &N_var_func,
//                             const arma::ucube &col_id,
//                             const arma::umat &N_par,
//                             const arma::cube &cov_data,
//                             const arma::vec &gamma){
//   DMatrix dmat(B,N_dim,N_func,func_def,N_var_func,col_id,N_par,cov_data,gamma);
//   arma::field<arma::mat> DBlocks = dmat.genCholD();
//   return(DBlocks);
// }

//' Returns log likelihood for a set of observatios
//' 
//' Generates the covariance matrix of the random effects from a sparse representation
//' @param B Integer specifying the number of blocks in the matrix
//' @param N_dim Vector of integers, which each value specifying the dimension of each block
//' @param N_func Vector of integers specifying the number of functions in the covariance function 
//' for each block.
//' @param func_def Matrix of integers where each column specifies the function definition for each function in each block. 
//' @param N_var_func Matrix of integers of same size as `func_def` with each column specying the number 
//' of variables in the argument to each function in each block
//' @param col_id 3D array (cube) of integers of dimension length(func_def) x max(N_var_func) x B 
//' where each slice the respective column indexes of `cov_data` for each function in the block
//' @param N_par Matrix of integers of same size as `func_def` with each column specifying the number
//' of parameters in the function in each block
//' @param sum_N_par Total number of parameters
//' @param cov_data 3D array (cube) holding the data for the covariance matrix where each of the B slices
//' is the data required for each block
//' @param gamma Vector of covariance parameters specified in order they appear column wise in the functions 
//' specified by `func_def`
//' @param u A realisation of the random effects
//' @return A lower triangular matrix matrix
// [[Rcpp::export]]
double loglikD(Rcpp::List D_data,
               const arma::vec &gamma,
               const arma::vec &u){
  DMatrix dmat(D_data,gamma);
  dmat.gen_blocks_byfunc();
  double ll = dmat.loglik(u);//const_cast<double*>(u.memptr())
  return(ll);
}

// double loglikD(const arma::uword &B,
//                                 const arma::uvec &N_dim,
//                                 const arma::uvec &N_func,
//                                 const arma::umat &func_def,
//                                 const arma::umat &N_var_func,
//                                 const arma::ucube &col_id,
//                                 const arma::umat &N_par,
//                                 const arma::cube &cov_data,
//                                 const arma::vec &gamma,
//                                 const arma::vec &u){
//   DMatrix dmat(B,N_dim,N_func,func_def,N_var_func,col_id,N_par,cov_data,gamma);
//   dmat.gen_blocks_byfunc();
//   double ll = dmat.loglik(u);//const_cast<double*>(u.memptr())
//   return(ll);
// }

// //' Returns log likelihood for a set of observatios
// //' 
// //' Generates the covariance matrix of the random effects from a sparse representation
// //' @param B Integer specifying the number of blocks in the matrix
// //' @param N_dim Vector of integers, which each value specifying the dimension of each block
// //' @param N_func Vector of integers specifying the number of functions in the covariance function 
// //' for each block.
// //' @param func_def Matrix of integers where each column specifies the function definition for each function in each block. 
// //' @param N_var_func Matrix of integers of same size as `func_def` with each column specying the number 
// //' of variables in the argument to each function in each block
// //' @param col_id 3D array (cube) of integers of dimension length(func_def) x max(N_var_func) x B 
// //' where each slice the respective column indexes of `cov_data` for each function in the block
// //' @param N_par Matrix of integers of same size as `func_def` with each column specifying the number
// //' of parameters in the function in each block
// //' @param sum_N_par Total number of parameters
// //' @param cov_data 3D array (cube) holding the data for the covariance matrix where each of the B slices
// //' is the data required for each block
// //' @param gamma Vector of covariance parameters specified in order they appear column wise in the functions 
// //' specified by `func_def`
// //' @param u A realisation of the random effects
// //' @return A lower triangular matrix matrix
// // [[Rcpp::export]]
// arma::rowvec loggradD(const arma::uword &B,
//                const arma::uvec &N_dim,
//                const arma::uvec &N_func,
//                const arma::umat &func_def,
//                const arma::umat &N_var_func,
//                const arma::ucube &col_id,
//                const arma::umat &N_par,
//                const arma::uword &sum_N_par,
//                const arma::cube &cov_data,
//                const arma::vec &gamma,
//                const arma::vec &u){
//   DMatrix dmat(B,N_dim,N_func,func_def,N_var_func,col_id,N_par,sum_N_par,cov_data,gamma);
//   dmat.gen_blocks_byfunc();
//   arma::rowvec ll = dmat.log_gradient(u);
//   return(ll);
// }

//' Generates the derivative of the link function with respect to the mean
//' 
//' @param xb Vector with mean function value evaluated at fitted model parameters
//' @param family String declaring model family
//' @param link String declaring model link function
//' @return Vector of derivative values
// [[Rcpp::export]]
arma::vec gen_dhdmu(const arma::vec &xb,
                std::string family,
                std::string link){
  arma::vec out = dhdmu(xb,family,link);
  return out;
}

//' Combines a field of matrices into a block diagonal matrix
//' 
//' Combines a field of matrices into a block diagonal matrix. Used on
//' the output of `genD`
//' @param matfield A field of matrices
//' @return A block diagonal matrix
// [[Rcpp::export]]
arma::mat blockMat(arma::field<arma::mat> matfield){
  return(blockMatComb(matfield));
}
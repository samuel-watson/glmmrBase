#include <cmath>  
#include <RcppArmadillo.h>
#include "../inst/include/glmmr.h"
using namespace Rcpp;
using namespace arma;

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]

// //' Generates a block of the random effects covariance matrix
// //' 
// //' Generates a block of the random effects covariance matrix
// //' @details 
// //' Using the sparse representation of the random effects covariance matrix, constructs
// //' one of the blocks. The function definitions are: 1 indicator, 2 exponential,
// //' 3 AR-1, 4 squared exponential, 5 matern, 6 Bessel.
// //' @param N_dim Integer specifying the dimension of the matrix
// //' @param N_func Integer specifying the number of functions in the covariance function 
// //' for this block.
// //' @param func_def Vector of integers of same length as `func_def` specifying the function definition for each function. 
// //' @param N_var_func Vector of integers of same length as `func_def` specying the number 
// //' of variables in the argument to the function
// //' @param col_id Matrix of integers of dimension length(func_def) x max(N_var_func) that indicates
// //' the respective column indexes of `cov_data` 
// //' @param N_par Vector of integers of same length as `func_def` specifying the number
// //' of parameters in the function
// //' @param cov_data Matrix holding the data for the covariance matrix
// //' @param gamma Vector of covariance parameters specified in order they appear in the functions 
// //' specified by `func_def`
// //' @return A symmetric positive definite matrix
// // [[Rcpp::export]]
// arma::mat genBlockD(size_t N_dim,
//                            size_t N_func,
//                            const arma::uvec &func_def,
//                            const arma::uvec &N_var_func,
//                            const arma::umat &col_id,
//                            const arma::uvec &N_par,
//                            const arma::mat &cov_data,
//                            const arma::vec &gamma){
//   arma::mat D(N_dim,N_dim,fill::zeros);
//   if(!all(func_def == 1)){
// #pragma omp parallel for
//     for(arma::uword i=0;i<(N_dim-1);i++){
//       for(arma::uword j=i+1;j<N_dim;j++){
//         double val = 1;
//         size_t gamma_idx = 0;
//         for(arma::uword k=0;k<N_func;k++){
//           double dist = 0;
//           for(arma::uword p=0; p<N_var_func(k); p++){
//             dist += pow(cov_data(i,col_id(k,p)-1) - cov_data(j,col_id(k,p)-1),2);
//           }
//           dist= pow(dist,0.5);
//           
//           if(func_def(k)==1){
//             if(dist==0){
//               val = val*pow(gamma(gamma_idx),2);
//             } else {
//               val = 0;
//             }
//           } else if(func_def(k)==2){
//             val = val*exp(-1*dist/gamma(gamma_idx));//fexp(dist,gamma(gamma_idx));
//           }else if(func_def(k)==3){
//             val = val*pow(gamma(gamma_idx),dist);
//           } else if(func_def(k)==4){
//             val = val*gamma(gamma_idx)*exp(-1*pow(dist,2)/pow(gamma(gamma_idx+1),2));
//           } else if(func_def(k)==5){
//             double xr = pow(2*gamma(gamma_idx+1),0.5)*dist/gamma(gamma_idx);
//             double ans = 1;
//             if(xr!=0){
//               if(gamma(gamma_idx+1) == 0.5){
//                 ans = exp(-xr);
//               } else {
//                 double cte = pow(2,-1*(gamma(gamma_idx+1)-1))/R::gammafn(gamma(gamma_idx+1));
//                 ans = cte*pow(xr, gamma(gamma_idx+1))*R::bessel_k(xr,gamma(gamma_idx+1),1);
//               }
//             }
//             val = val*ans;
//           } else if(func_def(k)==6){
//             val = val* R::bessel_k(dist/gamma(gamma_idx),1,1);
//           }
//           gamma_idx += N_par(k);      
//         }
//         
//         D(i,j) = val;
//         D(j,i) = val;
//       }
//     }
//   }
//   
// #pragma omp parallel for
//   for(arma::uword i=0;i<N_dim;i++){
//     double val = 1;
//     size_t gamma_idx_ii = 0;
//     for(arma::uword k=0;k<N_func;k++){
//       if(func_def(k)==1){
//         val = val*pow(gamma(gamma_idx_ii),2);
//       } 
//     }
//     D(i,i) = val;
//   }
//   
//   return D;
// }

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
//' @param cov_data 3D array (cube) holding the data for the covariance matrix where each of the B slices
//' is the data required for each block
//' @param gamma Vector of covariance parameters specified in order they appear column wise in the functions 
//' specified by `func_def`
//' @return A symmetric positive definite covariance matrix
// [[Rcpp::export]]
arma::field<arma::mat> genD(const arma::uword &B,
                                   const arma::uvec &N_dim,
                                   const arma::uvec &N_func,
                                   const arma::umat &func_def,
                                   const arma::umat &N_var_func,
                                   const arma::ucube &col_id,
                                   const arma::umat &N_par,
                                   const arma::uword &sum_N_par,
                                   const arma::cube &cov_data,
                                   const arma::vec &gamma){
  // arma::field<arma::mat> DBlocks(B);
  // arma::uword g_idx = 0;
  // arma::uword sumpar;
  // for(arma::uword b=0;b<B;b++){
  //   sumpar = sum(N_par.row(b));
  //   DBlocks[b] = genBlockD(N_dim(b),
  //                          N_func(b),
  //                          func_def.row(b).t(),
  //                          N_var_func.row(b).t(),
  //                          col_id.slice(b),
  //                          N_par.row(b).t(),
  //                          cov_data.slice(b),
  //                          gamma.subvec(g_idx,g_idx+sumpar-1));
  //   g_idx += sumpar;
  // }
  
  DMatrix dmat(B,N_dim,N_func,func_def,N_var_func,col_id,N_par,sum_N_par,cov_data,gamma);
  arma::field<arma::mat> DBlocks = dmat.genD();
  return(DBlocks);
}

// [[Rcpp::export]]
arma::vec gen_dhdmu(const arma::vec &xb,
                std::string family,
                std::string link){
  arma::vec out = dhdmu(xb,family,link);
  return out;
}
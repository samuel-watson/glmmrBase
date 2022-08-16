#ifndef GLMMR_H
#define GLMMR_H

#include <cmath> 
#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

inline double gaussian_cdf(double x){
  return R::pnorm(x, 0, 1, true, false);
}

inline arma::vec gaussian_cdf_vec(const arma::vec& v){
  arma::vec res = arma::zeros<arma::vec>(v.n_elem);
  for (arma::uword i = 0; i < v.n_elem; ++i)
    res[i] = gaussian_cdf(v[i]);
  return res;
}

inline double gaussian_pdf(double x){
  return R::dnorm(x, 0, 1, false);
}

inline arma::vec gaussian_pdf_vec(const arma::vec& v){
  arma::vec res = arma::zeros<arma::vec>(v.n_elem);
  for (arma::uword i = 0; i < v.n_elem; ++i)
    res[i] = gaussian_pdf(v[i]);
  return res;
}

inline double log_mv_gaussian_pdf(const arma::vec& u,
                                  const arma::mat& D,
                                  const double& logdetD){
  arma::uword Q = u.n_elem;
  return (-0.5*Q*log(2*arma::datum::pi)-
          0.5*logdetD - 0.5*arma::as_scalar(u.t()*D*u));
}

extern arma::mat genBlockD(size_t N_dim,
                           size_t N_func,
                           const arma::uvec &func_def,
                           const arma::uvec &N_var_func,
                           const arma::umat &col_id,
                           const arma::uvec &N_par,
                           const arma::mat &cov_data,
                           const arma::vec &gamma);

extern arma::field<arma::mat> genD(const arma::uword &B,
                                   const arma::uvec &N_dim,
                                   const arma::uvec &N_func,
                                   const arma::umat &func_def,
                                   const arma::umat &N_var_func,
                                   const arma::ucube &col_id,
                                   const arma::umat &N_par,
                                   const arma::uword &sum_N_par,
                                   const arma::cube &cov_data,
                                   const arma::vec &gamma);

// //' Exponential covariance function
// //' 
// //' Exponential covariance function
// //' @details
// //' \deqn{\theta_1 exp(-x/\theta_2)}
// //' @param x Numeric value 
// //' @param par1 First parameter of the distribution
// // [[Rcpp::export]]
// inline double fexp(const double &x,
//                    double par1) {
//   return exp(-1*x/par1);
// }

// inline double sqexp(const double &x, 
//                     double par1,
//                     double par2) {
//   return par1*exp(-1*pow(x,2)/pow(par2,2));
// }
// 
// inline double matern(const double &x,
//                      double rho, 
//                      double nu){
//   double xr = pow(2*nu,0.5)*x/rho;
//   double ans = 1;
//   if(xr!=0){
//     if(nu == 0.5){
//       ans = exp(-xr);
//     } else {
//       double cte = pow(2,-1*(nu-1))/R::gammafn(nu);
//       ans = cte*pow(xr, nu)*R::bessel_k(xr,nu,1);
//     }
//   }
//   return ans;
// }


#endif
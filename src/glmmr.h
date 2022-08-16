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

extern arma::vec gen_dhdmu(const arma::vec &xb,
                           std::string family,
                           std::string link);

#endif
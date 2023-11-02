#pragma once

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS 

#include "glmmr/general.h"
#include "glmmr/maths.h"
#include "glmmr/formula.hpp"
#include "glmmr/covariance.hpp"
#include "glmmr/linearpredictor.hpp"
#include "glmmr/model.hpp"
#include "glmmr/modelbits.hpp"
#include "glmmr/openmpheader.h"
#include "glmmr/nngpcovariance.hpp"
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

typedef glmmr::Model<bits > glmm;
typedef glmmr::Model<bits_nngp> glmm_nngp;
typedef glmmr::Model<bits_hsgp > glmm_hsgp;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

struct glmmrType
{
  std::variant<int, Rcpp::XPtr<glmm>, Rcpp::XPtr<glmm_nngp>, Rcpp::XPtr<glmm_hsgp> > ptr; 
  glmmrType(SEXP xp, int type) : ptr(0) {
    if(type == 0){
      Rcpp::XPtr<glmm> newptr(xp);
      ptr = newptr;
    } else if(type== 1){
      Rcpp::XPtr<glmm_nngp> newptr(xp);
      ptr = newptr;
    } else if(type == 2){
      Rcpp::XPtr<glmm_hsgp> newptr(xp);
      ptr = newptr;
    } 
  }
};

using returnType = std::variant<int, double, Eigen::VectorXd, Eigen::ArrayXd, Eigen::MatrixXd, dblvec, strvec, intvec, vector_matrix, matrix_matrix, kenward_data, std::vector<Eigen::MatrixXd> >;

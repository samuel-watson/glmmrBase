#pragma once

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

enum class Type {
  GLMM = 0,
  GLMM_NNGP = 1,
  GLMM_HSGP = 2
};

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

struct glmmrType
{
  std::variant<int, Rcpp::XPtr<glmm>, Rcpp::XPtr<glmm_nngp>, Rcpp::XPtr<glmm_hsgp> > ptr; 
  glmmrType(SEXP xp, Type type) : ptr(0) {
    using enum Type;
    if(type == GLMM){
      Rcpp::XPtr<glmm> newptr(xp);
      ptr = newptr;
    } else if(type== GLMM_NNGP){
      Rcpp::XPtr<glmm_nngp> newptr(xp);
      ptr = newptr;
    } else if(type == GLMM_HSGP){
      Rcpp::XPtr<glmm_hsgp> newptr(xp);
      ptr = newptr;
    } 
  }
};

using returnType = std::variant<int, double, Eigen::VectorXd, Eigen::ArrayXd, Eigen::MatrixXd, 
                                dblvec, strvec, intvec, vector_matrix, matrix_matrix, kenward_data, 
                                std::vector<Eigen::MatrixXd>, std::pair<double,double> >;

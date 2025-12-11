#pragma once

#include <variant>
#include "glmmr/model.hpp"
#include "glmmr/openmpheader.h"
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

typedef glmmr::Model<bits > glmm;
typedef glmmr::Model<bits_ar1 > glmm_ar1;
typedef glmmr::Model<bits_nngp> glmm_nngp;
typedef glmmr::Model<bits_hsgp > glmm_hsgp;

enum class Type {
  GLMM = 0,
  GLMM_NNGP = 1,
  GLMM_HSGP = 2,
  GLMM_AR1 = 3
};

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

struct glmmrType
{
  std::variant<int, Rcpp::XPtr<glmm>, Rcpp::XPtr<glmm_nngp>, Rcpp::XPtr<glmm_hsgp>, Rcpp::XPtr<glmm_ar1> > ptr; 
  glmmrType(SEXP xp, Type type) : ptr(0) {
    if(type == Type::GLMM){
      Rcpp::XPtr<glmm> newptr(xp);
      ptr = newptr;
    } else if(type== Type::GLMM_NNGP){
      Rcpp::XPtr<glmm_nngp> newptr(xp);
      ptr = newptr;
    } else if(type == Type::GLMM_HSGP){
      Rcpp::XPtr<glmm_hsgp> newptr(xp);
      ptr = newptr;
    } else if(type == Type::GLMM_AR1){
      Rcpp::XPtr<glmm_ar1> newptr(xp);
      ptr = newptr;
    } 
  }
};

using returnType = std::variant<int, double, bool, Eigen::VectorXd, Eigen::ArrayXd, Eigen::MatrixXd, 
                                dblvec, strvec, intvec, VectorMatrix, MatrixMatrix, CorrectionData<glmmr::SE::KR>,
                                CorrectionData<glmmr::SE::KR2>, CorrectionData<glmmr::SE::KRBoth>,
                                CorrectionData<glmmr::SE::Sat>, std::vector<Eigen::MatrixXd>, std::pair<double,double>, BoxResults,
                                std::pair<int,int> >;


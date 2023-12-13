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
    if(type == Type::GLMM){
      Rcpp::XPtr<glmm> newptr(xp);
      ptr = newptr;
    } else if(type== Type::GLMM_NNGP){
      Rcpp::XPtr<glmm_nngp> newptr(xp);
      ptr = newptr;
    } else if(type == Type::GLMM_HSGP){
      Rcpp::XPtr<glmm_hsgp> newptr(xp);
      ptr = newptr;
    } 
  }
};

using returnType = std::variant<int, double, Eigen::VectorXd, Eigen::ArrayXd, Eigen::MatrixXd, 
                                dblvec, strvec, intvec, VectorMatrix, MatrixMatrix, CorrectionData<glmmr::SE::KR>,
                                CorrectionData<glmmr::SE::KR2>, CorrectionData<glmmr::SE::KRBoth>,
                                CorrectionData<glmmr::SE::Sat>, std::vector<Eigen::MatrixXd>, std::pair<double,double>, BoxResults >;

// WORKING ON A SOLUTION TO REDUCE SIZE OF CPP CODE WITH FUNCTION 
// TEMPLATE - HAVEN'T WORKED OUT HOW TO PASS FUNCTION ARGUMENTS AND ALSO ONLY REMOVES ONE LINE PER FUNCTION
// SO FAR! WILL COME BACK TO THIS
// struct FnBase {
//   glmmrType model;
//   FnBase(SEXP xp, int type = 0) : model(xp,static_cast<Type>(type)) {};
// };
// 
// template<typename T>
// struct Fn : public FnBase
// {
//   Fn(SEXP xp, int type = 0) : FnBase(xp, type) {};
//   template<class Visitor>
//   constexpr T operator()(Visitor vis){
//     auto S = std::visit(vis,model.ptr);
//     return std::get<T>(S);
//   }
// };
// 
// struct FnVoid : public FnBase
// {
//   FnVoid(SEXP xp, int type = 0) : FnBase(xp, type) {};
//   template<class Visitor>
//   constexpr void operator()(Visitor vis){
//     auto S = std::visit(vis,model.ptr);
//   }
// };

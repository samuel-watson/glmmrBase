#pragma once

//defines
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS 
#define _USE_MATH_DEFINES
// #define ENABLE_DEBUG // COMMENT/UNCOMMENT FOR DEBUG - currently only useful in R builds as uses R print, will add more general error logging
#define R_BUILD //Uncomment to build for R with RCPP

// includes
#ifdef R_BUILD
#include <RcppEigen.h>
#else
#include <Eigen/Core>
#endif
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <stack>
#include <variant>
#include <SparseChol.h>
#include <set>
#include <queue>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>
#include <random>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/random.hpp>
#include <rbobyqa.h>

using namespace Eigen;
using namespace SparseOperators;

typedef std::string str;
typedef std::vector<str> strvec;
typedef std::vector<int> intvec;
typedef std::vector<double> dblvec;
typedef std::vector<strvec> strvec2d;
typedef std::vector<dblvec> dblvec2d;
typedef std::vector<intvec> intvec2d;
typedef std::vector<dblvec2d> dblvec3d;
typedef std::vector<intvec2d> intvec3d;
typedef std::pair<double, double> dblpair;
typedef std::pair<std::string, double> strdblpair;

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

namespace glmmr {

enum class CovarianceFunction {
  gr = 0,
  ar = 1,
  fexp0 = 2,
  fexp = 3,
  sqexp0 = 4,
  sqexp = 5,
  bessel = 6,
  matern = 7,
  wend0 = 8,
  wend1 = 9,
  wend2 = 10,
  prodwm = 11,
  prodcb = 12,
  prodek = 13,
  ar0 = 14,
  ar1 = 15,
  dist = 16
};

enum class FamilyDistribution {
  gaussian = 0,
  bernoulli = 1,
  poisson = 2,
  gamma = 3,
  beta = 4,
  binomial = 5
};

enum class LinkDistribution {
  logit = 0,
  loglink = 1, // to avoid conflicting with log() function
  probit = 2,
  identity = 3,
  inverse = 4
};

const std::map<str, FamilyDistribution> str_to_family = {
  {"gaussian",FamilyDistribution::gaussian},
  {"bernoulli",FamilyDistribution::bernoulli},
  {"poisson",FamilyDistribution::poisson},
  {"gamma",FamilyDistribution::gamma},
  {"Gamma",FamilyDistribution::gamma},
  {"beta",FamilyDistribution::beta},
  {"binomial",FamilyDistribution::binomial}
};

const std::map<str, LinkDistribution> str_to_link = {
  {"logit",LinkDistribution::logit},
  {"log",LinkDistribution::loglink},
  {"probit",LinkDistribution::probit},
  {"identity",LinkDistribution::identity},
  {"inverse",LinkDistribution::inverse}
};

const std::map<str, CovarianceFunction> str_to_covfunc = {
  {"gr", CovarianceFunction::gr},
  {"ar", CovarianceFunction::ar},
  {"fexp0", CovarianceFunction::fexp0},
  {"fexp", CovarianceFunction::fexp},
  {"sqexp0",CovarianceFunction::sqexp0},
  {"sqexp",CovarianceFunction::sqexp},
  {"bessel",CovarianceFunction::bessel},
  {"matern",CovarianceFunction::matern},
  {"wend0",CovarianceFunction::wend0},
  {"wend1",CovarianceFunction::wend1},
  {"wend2",CovarianceFunction::wend2},
  {"prodwm",CovarianceFunction::prodwm},
  {"prodcb",CovarianceFunction::prodcb},
  {"prodek",CovarianceFunction::prodek},
  {"ar0", CovarianceFunction::ar0},
  {"ar1", CovarianceFunction::ar1},
  {"dist",CovarianceFunction::dist}
};

// unfortunately need bidirectional map so need to duplicate this unless there's
// a better way??
const std::map<CovarianceFunction, str> covfunc_to_str = {
  {CovarianceFunction::gr, "gr"},
  {CovarianceFunction::ar, "ar"},
  {CovarianceFunction::fexp0, "fexp0"},
  {CovarianceFunction::fexp, "fexp"},
  {CovarianceFunction::sqexp0, "sqexp0"},
  {CovarianceFunction::sqexp, "sqexp"},
  {CovarianceFunction::bessel, "bessel"},
  {CovarianceFunction::matern, "matern"},
  {CovarianceFunction::wend0, "wend0"},
  {CovarianceFunction::wend1, "wend1"},
  {CovarianceFunction::wend2, "wend2"},
  {CovarianceFunction::prodwm, "prodwm"},
  {CovarianceFunction::prodcb, "prodcb"},
  {CovarianceFunction::prodek, "prodek"},
  {CovarianceFunction::ar0, "ar0"},
  {CovarianceFunction::ar1, "ar1"},
  {CovarianceFunction::dist, "dist"}
};

const std::map<CovarianceFunction, int> covfunc_to_nvar = {
  {CovarianceFunction::gr, 1},
  {CovarianceFunction::ar, 2},
  {CovarianceFunction::fexp0, 1},
  {CovarianceFunction::fexp, 2},
  {CovarianceFunction::sqexp0, 1},
  {CovarianceFunction::sqexp, 2},
  {CovarianceFunction::bessel, 1},
  {CovarianceFunction::matern, 2},
  {CovarianceFunction::wend0, 2},
  {CovarianceFunction::wend1, 2},
  {CovarianceFunction::wend2, 2},
  {CovarianceFunction::prodwm, 2},
  {CovarianceFunction::prodcb, 2},
  {CovarianceFunction::prodek, 2},
  {CovarianceFunction::ar0, 1},
  {CovarianceFunction::ar1, 1},
  {CovarianceFunction::dist, 0}
};

const static std::unordered_map<str, double> nvars = {  
  {"gr", 1},
  {"ar", 2},
  {"fexp0", 1},
  {"fexp", 2},
  {"sqexp0",1},
  {"sqexp",2},
  {"bessel",1},
  {"matern",2},
  {"wend0",2},
  {"wend1",2},
  {"wend2",2},
  {"prodwm",2},
  {"prodcb",2},
  {"prodek",2},
  {"ar0", 1},
  {"ar1", 1}
};

inline bool validate_fn(const str& fn){
  bool not_fn = str_to_covfunc.find(fn) == str_to_covfunc.end();
  return not_fn;
}

//const static intvec xvar_rpn = {0,1,4,17};

template<typename T>
inline void print_vec_1d(const T& vec){
  Rcpp::Rcout << "\n[1]: ";
  for(auto j: vec) Rcpp::Rcout << j << " ";
}

template<typename T>
inline void print_vec_2d(const T& vec){
  for(int i = 0; i < vec.size(); i++){
    Rcpp::Rcout << "\n[" << i << "]: ";
    for(auto j: vec[i]) Rcpp::Rcout << j << " ";
  }
}

template<typename T>
inline void print_vec_3d(const T& vec){
  for(int i = 0; i < vec.size(); i++){
    Rcpp::Rcout << "\n[" << i << "]:";
    for(int k = 0; k < vec[i].size(); k++){
      Rcpp::Rcout << "\n   [" << i <<","<< k << "]: ";
      for(auto j: vec[i][k]) Rcpp::Rcout << j << " ";
    }
  }
}

inline void print_sparse(const sparse& A){
  Rcpp::Rcout << "\nmatL Ap: ";
  for(auto i: A.Ap)Rcpp::Rcout << " " << i;
  Rcpp::Rcout << "\nmatL Ai: ";
  for(auto i: A.Ai)Rcpp::Rcout << " " << i;
  Rcpp::Rcout << "\nmatL Ax: ";
  for(auto i: A.Ax)Rcpp::Rcout << " " << i;
}

inline bool is_number(const std::string& s)
{
  bool isnum = true;
  try {
    float a = std::stod(s);
  }
  catch (std::invalid_argument const& ex)
  {
    #ifdef ENABLE_DEBUG
    Rcpp::Rcout << " Not double: " << ex.what() << '\n';
    #endif
    isnum = false;
  }
  return isnum;
}

inline bool isalnum_or_uscore(const char& s)
{
  return (isalnum(s) || s=='_');
}

template<typename T>
inline bool expect_number_of_unique_elements(const std::vector<T> vec,
                                             int n){
  int vec_size = std::set<T>(vec.begin(),vec.end()).size();
  return vec_size==n;
}
}

struct vector_matrix{
public:
  VectorXd vec;
  MatrixXd mat;
  vector_matrix(int n): vec(n), mat(n,n) {};
  vector_matrix(const vector_matrix& x) : vec(x.vec), mat(x.mat) {};
  vector_matrix& operator=(vector_matrix x){
    vec = x.vec;
    mat = x.mat;
    return *this;
  };
};

struct matrix_matrix{
public:
  MatrixXd mat1;
  MatrixXd mat2;
  double a = 0;
  double b = 0;
  matrix_matrix(int n1, int m1, int n2, int m2): mat1(n1,m1), mat2(n2,m2) {};
  matrix_matrix(const matrix_matrix& x) : mat1(x.mat1), mat2(x.mat2) {};
  matrix_matrix& operator=(matrix_matrix x){
    mat1 = x.mat1;
    mat2 = x.mat2;
    a = x.a;
    b = x.b;
    return *this;
  };
};

struct kenward_data{
public:
  MatrixXd vcov_beta;
  MatrixXd vcov_theta;
  VectorXd dof;
  VectorXd lambda;
  kenward_data(int n1, int m1, int n2, int m2): vcov_beta(n1,m1), vcov_theta(n2,m2), dof(n1), lambda(n1) {};
  kenward_data(const kenward_data& x) : vcov_beta(x.vcov_beta), vcov_theta(x.vcov_theta), dof(x.dof), lambda(x.lambda) {};
  kenward_data& operator=(kenward_data x){
    vcov_beta = x.vcov_beta;
    vcov_theta = x.vcov_theta;
    dof = x.dof;
    lambda = x.lambda;
    return *this;
  };
};

#ifndef MATHS_H
#define MATHS_H

#define _USE_MATH_DEFINES

#include <cmath> 
//#include <Eigen/Dense>
#include <unordered_map>
#include <RcppEigen.h>
#include "algo.h"
//#include <Rcpp.h>

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {
namespace maths {

inline double gaussian_cdf(double value)
{
  return 0.5 * erfc(-value * 0.707106781186547524401);
}

inline Eigen::VectorXd gaussian_cdf_vec(const Eigen::VectorXd& v) {
  Eigen::VectorXd res(v.size());
  for (int i = 0; i < v.size(); ++i)
    res(i) = gaussian_cdf(v(i));
  return res;
}

template <typename T>
inline T gaussian_pdf(T x)
{
  static const T inv_sqrt_2pi = 0.3989422804014327;
  return inv_sqrt_2pi * std::exp(-T(0.5) * x * x);
}

inline Eigen::VectorXd gaussian_pdf_vec(const Eigen::VectorXd& v)
{
  Eigen::VectorXd res(v.size());
  for (int i = 0; i < v.size(); ++i)
    res(i) = gaussian_pdf(v(i));
  return res;
}

inline Eigen::VectorXd exp_vec(const Eigen::VectorXd& x,
                               bool logit = false)
{
  Eigen::VectorXd z(x.size());
  for (int i = 0; i < x.size(); i++)
  {
    z(i) = logit ? std::exp(x(i)) / (1 + std::exp(x(i))) : std::exp(x(i));
  }
  return z;
}

inline Eigen::VectorXd mod_inv_func(Eigen::VectorXd mu,
                                    std::string link)
{
  const static std::unordered_map<std::string, int> string_to_case{
    {"logit",1},
    {"log",2},
    {"probit",3},
    {"identity",4}
  };
  switch (string_to_case.at(link)) {
  case 1:
    mu = exp_vec(mu, true);
    break;
  case 2:
    mu = exp_vec(mu);
    break;
  case 3:
    mu = gaussian_cdf_vec(mu);
    break;
  case 4:
    break;
  }
  return mu;
}

inline Eigen::VectorXd dhdmu(const Eigen::VectorXd& xb,
                             std::string family,
                             std::string link) {
  
  Eigen::VectorXd wdiag(xb.size());
  Eigen::ArrayXd p(xb.size());
  const static std::unordered_map<std::string, int> string_to_case{
    {"poissonlog",1},
    {"poissonidentity",2},
    {"binomiallogit",3},
    {"binomiallog",4},
    {"binomialidentity",5},
    {"binomialprobit",6},
    {"gaussianidentity",7},
    {"gaussianlog",8},
    {"Gammalog",9},
    {"Gammainverse",10},
    {"Gammaidentity",11},
    {"betalogit",12}
  };
  
  switch (string_to_case.at(family + link)) {
  case 1:
    wdiag = exp_vec(-1.0 * xb);
    break;
  case 2:
    wdiag = exp_vec(xb);
    break;
  case 3:
    p = mod_inv_func(xb, "logit");
    for(int i =0; i< xb.size(); i++){
      wdiag(i) = 1/(p(i)*(1.0 - p(i)));
    }
    break;
  case 4:
    p = mod_inv_func(xb, "log");
    for(int i =0; i< xb.size(); i++){
      wdiag(i) = (1.0 - p(i))/p(i);
    }
    break;
  case 5:
    p = mod_inv_func(xb, "identity");
    wdiag = (p * (1 - p)).matrix();
    break;
  case 6:
    {
      p = mod_inv_func(xb, "probit");
      Eigen::ArrayXd pinv = gaussian_pdf_vec(xb);
      for(int i =0; i< xb.size(); i++){
        wdiag(i) = (p(i) * (1-p(i)))/pinv(i);
      }
      break;
    }
  case 7:
    for(int i =0; i< xb.size(); i++){
      wdiag(i) = 1.0;
    }
    break;
  case 8:
    for(int i =0; i< xb.size(); i++){
      wdiag(i) = 1/exp(xb(i));
    }
    break;
  case 9:
    for(int i =0; i< xb.size(); i++){
      wdiag(i) = 1.0;
    }
    break;
  case 10:
    for(int i =0; i< xb.size(); i++){
      wdiag(i) = 1/(xb(i)*xb(i));
    }
    break;
  case 11:
    for(int i =0; i< xb.size(); i++){
      wdiag(i) = (xb(i)*xb(i));
    }
    break;
  case 12:
    p = mod_inv_func(xb, "logit");
    for(int i =0; i< xb.size(); i++){
      wdiag(i) = 1/(p(i)*(1.0 - p(i)));
    }
    break;
  }
  
  return wdiag;
}

}

namespace tools {
inline std::vector<int> linseq(int start, int end) {
  std::vector<int> idx;
  for (int i = start; i <= end; i++) {
    idx.push_back(i);
  }
  return idx;
}
}
}







#endif
#ifndef ALGO_H
#define ALGO_H

#include <cmath> 
#include <RcppEigen.h>
#include "general.h"

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {
namespace algo {
inline double inner_sum(double* li, double* lj, int n)
{
  double s = 0;
  for (int i = 0; i < n; i++) {
    s += li[i] * lj[i];
  }
  return s;
}

inline int get_flink(const std::string &family,
                     const std::string &link){
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
  
  return string_to_case.at(family + link);
}

inline Eigen::VectorXd forward_sub(const Eigen::MatrixXd& U,
                                   const Eigen::VectorXd& u,
                                   const intvec& idx)
{
  Eigen::VectorXd y(idx.size());
  for (int i = 0; i < idx.size(); i++) {
    double lsum = 0;
    for (int j = 0; j < i; j++) {
      lsum += U(i,j) * y(j);
    }
    y(i) = (u(idx[i]) - lsum) / U(i,i);
  }
  return y;
}


}
}







#endif
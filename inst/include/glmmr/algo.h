#ifndef ALGO_H
#define ALGO_H

#include <cmath> 
#include <RcppEigen.h>

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

// inline Eigen::VectorXd forward_sub(const Eigen::MatrixXd& U,
//                                    const Eigen::VectorXd& u)
// {
//   int n = u.size();
//   std::vector<double> y(n);
//   for (int i = 0; i < n; i++) {
//     double lsum = glmmr::algo::inner_sum(U.col(i).data(), &y[0], i);
//     y.push_back((u(i) - lsum) / U(i, i));
//   }
//   Eigen::VectorXd z = Eigen::Map<Eigen::VectorXd>(y.data(), n);
//   return z;
// }


}
}







#endif
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
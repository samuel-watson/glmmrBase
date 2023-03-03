#ifndef GLMM_H
#define GLMM_H

#include <RcppEigen.h>

namespace glmmr {

template<typename T>
class glmm {
public:
  T D;
  const Eigen::MatrixXd &Z_;
  const Eigen::MatrixXd &X_;
  const Eigen::VectorXd &y_; 
  std::string family_; 
  std::string link_;
  
  glmm(const Eigen::ArrayXXi &cov,
       const Eigen::ArrayXd &data,
       const Eigen::ArrayXd &eff_range,
       const Eigen::MatrixXd &Z, 
       const Eigen::MatrixXd &X,
       const Eigen::VectorXd &y, 
       std::string family, 
       std::string link) :
    dat_(cov,data,eff_range), D(&dat_,Eigen::VectorXd::Constant(0.1,data_.n_cov_pars())),
    Z_(Z), X_(X), y_(y), family_(family), link_(link) {
    
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
    
    flink_ = string_to_case.at(family + link);
  }
  
protected:
  int flink_;
  glmmr::DData dat_;
  
};

}

#endif
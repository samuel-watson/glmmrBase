#ifndef LINEARPREDICTOR_HPP
#define LINEARPREDICTOR_HPP

#include "general.h"
#include "interpreter.h"
#include "formula.hpp"

namespace glmmr {

class LinearPredictor {
public:
  const Eigen::ArrayXXd data_;
  const strvec colnames_;
  Eigen::VectorXd parameters_;
  
  LinearPredictor(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames) : 
    data_(data), colnames_(colnames), parameters_(form.fe_.size()), form_(form), 
    P_(form.fe_.size()), X_(data.rows(),form.RM_INT ? form.fe_.size() : form.fe_.size()+1) {
    parameters_.setZero();
    parse();
  };
  
  LinearPredictor(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) : 
    data_(data), colnames_(colnames), parameters_(form.fe_.size()), form_(form), 
    P_(form_.fe_.size()), X_(data.rows(),form.RM_INT ? form.fe_.size() : form.fe_.size()+1) {
    for(int i = 0; i < parameters.size(); i++)parameters_(i) = parameters[i];
    parse();
  };
  
  LinearPredictor(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const Eigen::ArrayXd& parameters) : 
    data_(data), colnames_(colnames), parameters_(parameters.matrix()), form_(form), 
    P_(form_.fe_.size()), X_(data.rows(),form.RM_INT ? form.fe_.size() : form.fe_.size()+1) {
    parse();
  };
  
  void update_parameters(const dblvec& parameters){
    if(parameters.size()!=parameters_.size())Rcpp::stop("wrong number of parameters");
    for(int i = 0; i < parameters.size(); i++)parameters_(i) = parameters[i];
  };
  
  void update_parameters(const Eigen::ArrayXd& parameters){
    if(parameters.size()!=parameters_.size())Rcpp::stop("wrong number of parameters");
    parameters_ = parameters;
  };
  
  int P(){
    return P_;
  }
  
  void parse();
  
  Eigen::VectorXd xb(){
    return X_*parameters_;
  }
  
  Eigen::MatrixXd X(){
    return X_;
  }
  
  // void update_formula(const glmmr::Formula& form){
  //   form_ = glmmr::Formula(form);
  //   X_.resize(data_.rows(),form.fe_.size());
  //   P_ = form.fe_.size();
  //   parse();
  // }
  
  
private:
  const glmmr::Formula& form_;
  int P_;
  intvec x_cols_;
  Eigen::MatrixXd X_;
};
}

inline void glmmr::LinearPredictor::parse(){
  P_ = form_.fe_.size();
  int int_log = form_.RM_INT ? 0 : 1;
  if(!form_.RM_INT) X_.col(0) = Eigen::VectorXd::Ones(X_.rows());
  if(P_>0){
    for(int i = 0; i<P_; i++){
      auto colidx = std::find(colnames_.begin(),colnames_.end(),form_.fe_[i]);
      if(colidx == colnames_.end()){
        Rcpp::stop("X variable not in colnames");
      } else {
        int colidxi = colidx - colnames_.begin();
        x_cols_.push_back(colidxi);
        X_.col(i+int_log) = data_.col(colidxi);
      }
    }
  }
}

#endif
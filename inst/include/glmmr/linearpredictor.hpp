#ifndef LINEARPREDICTOR_HPP
#define LINEARPREDICTOR_HPP

#include "general.h"
#include "interpreter.h"
#include "formula.hpp"
#include "xbformula.h"

namespace glmmr {

class LinearPredictor {
public:
  const Eigen::ArrayXXd data_;
  const strvec colnames_;
  Eigen::VectorXd parameters_;

  LinearPredictor(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames) :
    data_(data), colnames_(colnames), 
    parameters_(form.RM_INT ? form.fe_.size() : form.fe_.size()+1), 
    form_(form),
    P_(form.RM_INT ? form.fe_.size() : form.fe_.size()+1),
    n_(data.rows()),
    X_(n_,P_) {
    parameters_.setZero();
    parse();
  };

  LinearPredictor(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) :
    data_(data), colnames_(colnames), 
    parameters_(parameters.size()), 
    form_(form),
    P_(form.RM_INT ? form.fe_.size() : form.fe_.size()+1),
    n_(data.rows()),
    X_(n_,P_) {
    for(int i = 0; i < parameters.size(); i++)parameters_(i) = parameters[i];
    parse();
  };

  LinearPredictor(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const Eigen::ArrayXd& parameters) :
    data_(data), colnames_(colnames), 
    parameters_(parameters.matrix()), 
    form_(form),
    P_(form.RM_INT ? form.fe_.size() : form.fe_.size()+1),
    n_(data.rows()),
    X_(n_,P_) {
    parse();
  };

  void update_parameters(const dblvec& parameters){
    if(parameters.size()!=P_)Rcpp::stop("wrong number of parameters");
    for(int i = 0; i < P_; i++)parameters_(i) = parameters[i];
  };

  void update_parameters(const Eigen::ArrayXd& parameters){
    if(parameters.size()!=P_)Rcpp::stop("wrong number of parameters");
    parameters_ = parameters;
  };
  
  void update_parameters2(const dblvec& parameters){
    if(parameters.size()!=P_)Rcpp::stop("wrong number of parameters");
    for(int i = 0; i < P_; i++)parameters_(i) = parameters[i];
    int par_iterator = 0;
    for(int j = 0; j < n_fe_components_; j++){
      int n_par = x_components[j].pars();
      dblvec newpar(parameters.begin()+par_iterator,parameters.begin()+par_iterator+n_par-1);
      x_components[j].update_parameters(newpar);
      par_iterator += n_par;
    }
  };

  int P(){
    return P_;
  }

  void parse();
  
  void parse2();

  Eigen::VectorXd xb(){
    return X_*parameters_;
  }

  Eigen::MatrixXd X(){
    return X_;
  }

private:
  const glmmr::Formula& form_;
  int P_;
  int n_fe_components_;
  int n_;
  intvec x_cols_;
  Eigen::MatrixXd X_;
  std::vector<glmmr::xbFormula> x_components;
};
}

inline void glmmr::LinearPredictor::parse(){
  glmmr::print_vec_1d<strvec>(form_.fe_);
  P_ = form_.RM_INT ? form_.fe_.size() : form_.fe_.size()+1;
  int int_log = form_.RM_INT ? 0 : 1;
  if(!form_.RM_INT) X_.col(0) = Eigen::VectorXd::Ones(X_.rows());
  if(form_.fe_.size()>0){
    for(int i = 0; i<form_.fe_.size(); i++){
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


inline void glmmr::LinearPredictor::parse2(){
  int parcounter = 0;
  P_ = 0;
  if(!form_.RM_INT){
    glmmr::xbFormula f1(n_);
    x_components.push_back(f1);
    parcounter++;
    P_++;
  }
  for(int i = 0; i<form_.fe_.size(); i++){
    glmmr::xbFormula f1(form_.fe_[i],data_,colnames_);
    x_components.push_back(f1);
    int npar = f1.pars();
    P_+=npar;
  }
  
  n_fe_components_ = x_components.size();
}


#endif
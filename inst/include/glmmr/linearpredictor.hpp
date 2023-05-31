#ifndef LINEARPREDICTOR_HPP
#define LINEARPREDICTOR_HPP

#include "general.h"
#include "interpreter.h"
#include "calculator.h"
#include "formula.hpp"

namespace glmmr {

class LinearPredictor {
public:
  dblvec parameters_;
  glmmr::calculator calc_;

  LinearPredictor(glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames) :
    colnames_(colnames),  
    form_(form),
    n_(data.rows()),
    X_(Eigen::MatrixXd::Zero(n_,1)) {
      form_.calculate_linear_predictor(calc_,data,colnames);
      glmmr::print_vec_1d<intvec>(calc_.instructions);
      glmmr::print_vec_1d<intvec>(calc_.indexes);
      glmmr::print_vec_1d<strvec>(calc_.parameter_names);
      P_ = calc_.parameter_names.size();
      X_.conservativeResize(n_,P_);
      X_.setZero();
    };

  LinearPredictor(glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) :
    colnames_(colnames), 
    form_(form),
    n_(data.rows()),
    X_(Eigen::MatrixXd::Zero(n_,1)) {
      form_.calculate_linear_predictor(calc_,data,colnames);
      update_parameters(parameters);
      P_ = calc_.parameter_names.size();
      X_.conservativeResize(n_,P_);
      X_ = calc_.jacobian();
      x_set_ = true;
    };

  LinearPredictor(glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const Eigen::ArrayXd& parameters) :
    colnames_(colnames), 
    form_(form),
    n_(data.rows()),
    X_(Eigen::MatrixXd::Zero(n_,1)) {
      form_.calculate_linear_predictor(calc_,data,colnames);
      update_parameters(parameters);
      P_ = calc_.parameter_names.size();
      X_.conservativeResize(n_,P_);
      X_ = calc_.jacobian();
      x_set_ = true;
    };

  void update_parameters(const dblvec& parameters){
    if(parameters.size()!=P_)Rcpp::stop("wrong number of parameters");
    parameters_ = parameters;
    calc_.parameters = parameters;
    if(!x_set_){
      X_ = calc_.jacobian();
      x_set_ = true;
      Rcpp::Rcout << "\nX: \n" << X_.topRows(15);
    }
  };

  void update_parameters(const Eigen::ArrayXd& parameters){
    if(parameters.size()!=P_)Rcpp::stop("wrong number of parameters");
    dblvec new_parameters(parameters.data(),parameters.data()+parameters.size());
    update_parameters(new_parameters);
  };

  int P(){
    return P_;
  }
  
  strvec colnames(){
    return colnames_;
  }

  VectorXd xb(){
    VectorXd xb = calc_.calculate();
    return xb;
  }

  MatrixXd X(){
    if(calc_.any_nonlinear){
      X_ = calc_.jacobian();
    }
    return X_;
  }
  
  VectorXd parameter_vector(){
    VectorXd pars = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(parameters_.data(),parameters_.size());
    return pars;
  }
  
  bool any_nonlinear(){
    return calc_.any_nonlinear;
  }

private:
  glmmr::Formula& form_;
  strvec colnames_;
  int P_;
  int n_fe_components_;
  int n_;
  intvec x_cols_;
  Eigen::MatrixXd X_;
  bool x_set_ = false;
};
}

#endif
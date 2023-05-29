#ifndef LINEARPREDICTOR_HPP
#define LINEARPREDICTOR_HPP

#include "general.h"
#include "interpreter.h"
#include "calculator.h"
#include "formula.hpp"
//#include "xbformula.h"

namespace glmmr {

class LinearPredictor {
public:
  //const Eigen::ArrayXXd data_;
  //const strvec colnames_;data_(data), colnames_(colnames), 
  dblvec parameters_;
  glmmr::calculator calc_;

  LinearPredictor(glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames) :
    colnames_(colnames),  
    form_(form),
    n_(data.rows()),
    X_(Eigen::MatrixXd::Zero(n_,1)) {
      //parse(data,colnames);
      form_.calculate_linear_predictor(calc_,data,colnames);
      P_ = calc_.parameter_names.size();
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
      //parse(data,colnames);
      update_parameters(parameters);
      P_ = calc_.parameter_names.size();
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
      //parse(data,colnames);
      update_parameters(parameters);
      P_ = calc_.parameter_names.size();
    };

  void update_parameters(const dblvec& parameters){
    if(parameters.size()!=P_)Rcpp::stop("wrong number of parameters");
    parameters_ = parameters;
    calc_.parameters = parameters;
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
  
  // void parse(const ArrayXXd& data,
  //             const strvec& colnames);

  VectorXd xb(){
    VectorXd xb = VectorXd::Zero(n_);
    for(int i = 0; i < n_; i++){
      xb(i) = calc_.calculate(i);
    }
    return xb;
  }

  MatrixXd X(){
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
};
}


// inline void glmmr::LinearPredictor::parse(const ArrayXXd& data,
//                                            const strvec& colnames){
//   int parcounter = 0;
//   P_ = 0;
//   if(!form_.RM_INT){
//     glmmr::xbFormula f1(n_);
//     x_components.push_back(f1);
//     parcounter++;
//     P_++;
//   }
//   
//   for(int i = 0; i<form_.fe_.size(); i++){
//     glmmr::xbFormula f2(form_.fe_[i],data,colnames);
//     x_components.push_back(f2);
//     P_ += f2.pars();
//   }
//   
//   n_fe_components_ = x_components.size();
//   if(parameters_.size()>0){
//     if(parameters_.size()!=P_)Rcpp::stop("Linear predictor parameter vector size not equal to number of parameters");
//   } else {
//     parameters_.resize(P_);
//   }
// }


#endif
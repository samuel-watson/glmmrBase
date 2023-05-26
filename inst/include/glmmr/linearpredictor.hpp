#ifndef LINEARPREDICTOR_HPP
#define LINEARPREDICTOR_HPP

#include "general.h"
#include "interpreter.h"
#include "formula.hpp"
#include "xbformula.h"

namespace glmmr {

class LinearPredictor {
public:
  //const Eigen::ArrayXXd data_;
  //const strvec colnames_;data_(data), colnames_(colnames), 
  dblvec parameters_;

  LinearPredictor(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames) :
    colnames_(colnames),  
    form_(form),
    P_(form.RM_INT ? form.fe_.size() : form.fe_.size()+1),
    n_(data.rows()),
    X_(Eigen::MatrixXd::Zero(n_,P_)) {
      parse(data,colnames);
    };

  LinearPredictor(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) :
    colnames_(colnames), 
    form_(form),
    P_(form.RM_INT ? form.fe_.size() : form.fe_.size()+1),
    n_(data.rows()),
    X_(Eigen::MatrixXd::Zero(n_,P_)) {
      parse(data,colnames);
      update_parameters(parameters);
    };

  LinearPredictor(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const Eigen::ArrayXd& parameters) :
    colnames_(colnames), 
    form_(form),
    P_(form.RM_INT ? form.fe_.size() : form.fe_.size()+1),
    n_(data.rows()),
    X_(Eigen::MatrixXd::Zero(n_,P_)) {
      parse(data,colnames);
      update_parameters(parameters);
    };

  void update_parameters(const dblvec& parameters){
    if(parameters.size()!=P_)Rcpp::stop("wrong number of parameters");
    parameters_ = parameters;
    // update the parameters in the components
    int par_counter = 0;
    int par_fn;
    for(int i = 0; i < n_fe_components_; i++){
      par_fn = x_components[i].pars();
      dblvec newpars(par_fn);
      for(int j = 0; j < par_fn; j++)newpars[j] = parameters_[par_counter+j];
      for(auto j: newpars)Rcpp::Rcout << j;
      x_components[i].update_parameters(newpars);
      par_counter += par_fn;
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
  
  void parse(const ArrayXXd& data,
              const strvec& colnames);

  VectorXd xb(){
    //return X_*parameters_;
    VectorXd xb = VectorXd::Zero(n_);
    for(int i = 0; i < n_fe_components_; i++){
      xb += x_components[i].xb();
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
    int i=0;
    bool any_nonlin = false;
    while(!any_nonlin && i<n_fe_components_){
      any_nonlin = x_components[i].nonlinear();
      i++;
    }
    return any_nonlin;
  }

private:
  const glmmr::Formula& form_;
  strvec colnames_;
  int P_;
  int n_fe_components_;
  int n_;
  intvec x_cols_;
  Eigen::MatrixXd X_;
  std::vector<glmmr::xbFormula> x_components;
};
}


inline void glmmr::LinearPredictor::parse(const ArrayXXd& data,
                                           const strvec& colnames){
  int parcounter = 0;
  P_ = 0;
  if(!form_.RM_INT){
    glmmr::xbFormula f1(n_);
    x_components.push_back(f1);
    parcounter++;
    P_++;
  }
  
  for(int i = 0; i<form_.fe_.size(); i++){
    glmmr::xbFormula f2(form_.fe_[i],data,colnames);
    x_components.push_back(f2);
    P_ += f2.pars();
  }
  
  n_fe_components_ = x_components.size();
  if(parameters_.size()>0){
    if(parameters_.size()!=P_)Rcpp::stop("Linear predictor parameter vector size not equal to number of parameters");
  } else {
    parameters_.resize(P_);
  }
}


#endif
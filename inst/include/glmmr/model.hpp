#ifndef MODEL_HPP
#define MODEL_HPP

#include "general.h"
#include "modelbits.hpp"
#include "randomeffects.hpp"
#include "modelmatrix.hpp"
#include "modelmcmc.hpp"
#include "modeloptim.hpp"

namespace glmmr {

using namespace Eigen;

class Model {
public:
  glmmr::ModelBits& model;
  glmmr::RandomEffects re;
  glmmr::ModelMatrix matrix;
  glmmr::ModelOptim optim;
  glmmr::ModelMCMC mcmc;
  
  Model(glmmr::ModelBits& model_) : model(model_), re(model), matrix(model,re), optim(model,matrix,re), mcmc(model,matrix,re) {};
  
  void set_offset(const VectorXd& offset_);
  void set_weights(const ArrayXd& weights_);
  void set_y(const VectorXd& y_);
  void update_beta(const dblvec &beta_);
  void update_theta(const dblvec &theta_);
  void update_u(const MatrixXd &u_);
  void set_trace(int trace_);
};

}

inline void glmmr::Model::set_offset(const VectorXd& offset_){
  if(offset_.size()!=model.n())Rcpp::stop("offset wrong length");
  model.data.set_offset(offset_);
}

inline void glmmr::Model::set_weights(const ArrayXd& weights_){
  if(weights_.size()!=model.n())Rcpp::stop("weights wrong length");
  model.data.set_weights(weights_);
  if((weights_ != 1.0).any()){
    if(model.family.family!="gaussian")Rcpp::warning("Weighted regression with non-Gaussian models is currently experimental.");
    model.weighted = true;
  }
}

inline void glmmr::Model::set_y(const VectorXd& y_){
  if(y_.size()!=model.n())Rcpp::stop("y wrong length");
  model.data.update_y(y_);
}

inline void glmmr::Model::update_beta(const dblvec &beta_){
  if(beta_.size()!=model.linear_predictor.P())Rcpp::stop("beta wrong length");
  model.linear_predictor.update_parameters(beta_);
}

inline void glmmr::Model::update_theta(const dblvec &theta_){
  if(theta_.size()!=model.covariance.npar())Rcpp::stop("theta wrong length");
  model.covariance.update_parameters(theta_);
  re.ZL = model.covariance.ZL_sparse();
  re.zu_ = re.ZL*re.u_;
}

inline void glmmr::Model::update_u(const MatrixXd &u_){
  if(u_.rows()!=model.covariance.Q())Rcpp::stop("u has wrong number of random effects");
  if(u_.cols()!=re.u(false).cols()){
    Rcpp::Rcout << "\nDifferent numbers of random effect samples";
    re.u_.conservativeResize(model.covariance.Q(),u_.cols());
    re.zu_.conservativeResize(model.covariance.Q(),u_.cols());
  }
  re.u_ = u_;
  re.zu_ = re.ZL*re.u_;
}

inline void glmmr::Model::set_trace(int trace_){
  optim.trace = trace_;
  mcmc.trace = trace_;
  if(trace_ > 0){
    mcmc.verbose = true;
  } else {
    mcmc.verbose = false;
  }
}

#endif
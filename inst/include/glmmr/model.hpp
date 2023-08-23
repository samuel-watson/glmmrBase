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

template<typename cov, typename linpred>
class Model {
public:
  glmmr::ModelBits<cov, linpred>& model;
  glmmr::RandomEffects<cov, linpred> re;
  glmmr::ModelMatrix<cov, linpred> matrix;
  glmmr::ModelOptim<cov, linpred> optim;
  glmmr::ModelMCMC<cov, linpred> mcmc;
  
  Model(glmmr::ModelBits<cov, linpred>& model_) : model(model_), re(model), matrix(model,re), optim(model,matrix,re), mcmc(model,matrix,re) {};
  
  virtual void set_offset(const VectorXd& offset_);
  virtual void set_weights(const ArrayXd& weights_);
  virtual void set_y(const VectorXd& y_);
  virtual void update_beta(const dblvec &beta_);
  virtual void update_theta(const dblvec &theta_);
  virtual void update_u(const MatrixXd &u_);
  virtual void set_trace(int trace_);
};

}

template<typename cov, typename linpred>
inline void glmmr::Model<cov, linpred>::set_offset(const VectorXd& offset_){
  //if(offset_.size()!=model.n())Rcpp::stop("offset wrong length");
  model.data.set_offset(offset_);
}

template<typename cov, typename linpred>
inline void glmmr::Model<cov, linpred>::set_weights(const ArrayXd& weights_){
  //if(weights_.size()!=model.n())Rcpp::stop("weights wrong length");
  model.data.set_weights(weights_);
  if((weights_ != 1.0).any()){
    //if(model.family.family!="gaussian")Rcpp::warning("Weighted regression with non-Gaussian models is currently experimental.");
    model.weighted = true;
  }
}

template<typename cov, typename linpred>
inline void glmmr::Model<cov, linpred>::set_y(const VectorXd& y_){
  //if(y_.size()!=model.n())Rcpp::stop("y wrong length");
  model.data.update_y(y_);
}

template<typename cov, typename linpred>
inline void glmmr::Model<cov, linpred>::update_beta(const dblvec &beta_){
  //if(beta_.size()!=(unsigned)model.linear_predictor.P())Rcpp::stop("beta wrong length");
  model.linear_predictor.update_parameters(beta_);
}

template<typename cov, typename linpred>
inline void glmmr::Model<cov, linpred>::update_theta(const dblvec &theta_){
  //if(theta_.size()!=(unsigned)model.covariance.npar())Rcpp::stop("theta wrong length");
  model.covariance.update_parameters(theta_);
  re.ZL = model.covariance.ZL_sparse();
  re.zu_ = re.ZL*re.u_;
}

template<typename cov, typename linpred>
inline void glmmr::Model<cov, linpred>::update_u(const MatrixXd &u_){
  //if(u_.rows()!=(unsigned)model.covariance.Q())Rcpp::stop("u has wrong number of random effects");
  if(u_.cols()!=re.u(false).cols()){
    //Rcpp::Rcout << "\nDifferent numbers of random effect samples";
    re.u_.conservativeResize(model.covariance.Q(),u_.cols());
    re.zu_.conservativeResize(model.covariance.Q(),u_.cols());
  }
  re.u_ = u_;
  re.zu_ = re.ZL*re.u_;
}

template<typename cov, typename linpred>
inline void glmmr::Model<cov, linpred>::set_trace(int trace_){
  optim.trace = trace_;
  mcmc.trace = trace_;
  if(trace_ > 0){
    mcmc.verbose = true;
  } else {
    mcmc.verbose = false;
  }
}

#endif
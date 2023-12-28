#pragma once

#include "general.h"
#include "modelbits.hpp"
#include "randomeffects.hpp"
#include "modelmatrix.hpp"
#include "openmpheader.h"
#include "maths.h"
#include "algo.h"
#include "sparse.h"
#include "calculator.hpp"
#include "direct.h"

namespace glmmr {

using namespace Eigen;

template<typename modeltype>
class ModelOptim{
public:
  // members 
  modeltype&                        model;
  glmmr::ModelMatrix<modeltype>&    matrix;
  glmmr::RandomEffects<modeltype>&  re;
  int                               trace = 0;
  // constructor
  ModelOptim(modeltype& model_, glmmr::ModelMatrix<modeltype>& matrix_,glmmr::RandomEffects<modeltype>& re_) ;

  struct OptimControl {
    int     npt = 0;
    double  rhobeg = 0;
    double  rhoend = 0;
    bool    direct = false;
    int     max_iter_direct = 100; 
    double  epsilon = 1e-4; 
    bool    select_one = true; 
    bool    trisect_once = false; 
    bool    bobyqa_w_direct = false; 
    int     max_eval = 0; 
    bool    mrdirect = false;
  } control;
  
  // functions
  virtual void    update_beta(const dblvec &beta);
  virtual void    update_theta(const dblvec &theta);
  virtual void    update_beta(const VectorXd &beta);
  virtual void    update_theta(const VectorXd &theta);
  virtual void    update_u(const MatrixXd& u_);
  virtual double  log_likelihood();
  virtual double  full_log_likelihood();
  virtual void    nr_beta();
  virtual void    laplace_nr_beta_u();
  virtual void    update_var_par(const double& v);
  virtual void    update_var_par(const ArrayXd& v);
  virtual void    ml_beta();
  virtual void    ml_theta();
  virtual void    ml_all();
  virtual void    laplace_ml_beta_u();
  virtual void    laplace_ml_theta();
  virtual void    laplace_ml_beta_theta();
  virtual double  aic();
  virtual ArrayXd optimum_weights(double N, VectorXd C, double tol = 1e-5, int max_iter = 501);
  void            set_bobyqa_control(int npt_, double rhobeg_, double rhoend_);
  void            set_optim_control(bool direct = false, int max_iter = 100, double epsilon = 1e-4, bool select_one = true, bool trisect_once = false, 
                                      bool bobyqa = false, int max_eval = 0, bool mrdirect = false);
  virtual MatrixXd hessian_numerical(double tol = 1e-4);
  void            set_bound(const dblvec& bound, bool lower = true);
  void            set_theta_bound(const dblvec& bound, bool lower = true);
  int             P() const;
  int             Q() const;
  // functions to optimise
  double          log_likelihood_beta(const dblvec &beta);
  double          log_likelihood_theta(const dblvec &theta);
  double          log_likelihood_theta_hsgp(const dblvec &theta);
  
protected:
// objects
  dblvec  lower_bound;
  dblvec  upper_bound; // bounds for beta
  dblvec  lower_bound_theta;
  dblvec  upper_bound_theta; // bounds for beta
  // functions
  void    calculate_var_par();
  dblvec  get_start_values(bool beta, bool theta, bool var = true);
  dblvec  get_lower_values(bool beta, bool theta, bool var = true);
  dblvec  get_upper_values(bool beta, bool theta, bool var = true);
  
};

}



template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::ml_beta(){
  if(control.direct)
  {
    dblvec start = get_start_values(true,false,false);
    dblvec range(start.size());
    std::fill(range.begin(),range.end(),3.0);
    directd op(start,range);
    op.control.max_iter = control.max_iter_direct;
    op.control.epsilon = control.epsilon;
    op.control.select_one = control.select_one;
    op.control.trisect_once = control.trisect_once;
    op.control.trace = trace;
    op.control.mrdirect = control.mrdirect;
    if(bobyqa) op.control.max_eval = control.max_eval;
    if constexpr (std::is_same_v<modeltype,bits>)
    {
      op.fn<&glmmr::ModelOptim<bits>::log_likelihood_beta, glmmr::ModelOptim<bits> >(this);
    } else if constexpr (std::is_same_v<modeltype,bits_nngp>) {
      op.fn<&glmmr::ModelOptim<bits_nngp>::log_likelihood_beta, glmmr::ModelOptim<bits_nngp> >(this);
    } else if constexpr (std::is_same_v<modeltype,bits_hsgp>){
      op.fn<&glmmr::ModelOptim<bits_hsgp>::log_likelihood_beta, glmmr::ModelOptim<bits_hsgp> >(this);
    }
    op.minimise();
    dblvec final = op.values();
    if(control.bobyqa_w_direct){
      bobyqad opb(final);
      opb.control.trace = trace;
      opb.control.rhobeg = op.rect_size();
      opb.control.rhoend = op.rect_size() * 1e-3;
      if constexpr (std::is_same_v<modeltype,bits>)
      {
        opb.fn<&glmmr::ModelOptim<bits>::log_likelihood_beta, glmmr::ModelOptim<bits> >(this);
      } else if constexpr (std::is_same_v<modeltype,bits_nngp>) {
        opb.fn<&glmmr::ModelOptim<bits_nngp>::log_likelihood_beta, glmmr::ModelOptim<bits_nngp> >(this);
      } else if constexpr (std::is_same_v<modeltype,bits_hsgp>){
        opb.fn<&glmmr::ModelOptim<bits_hsgp>::log_likelihood_beta, glmmr::ModelOptim<bits_hsgp> >(this);
      }
      opb.minimise();
      final = opb.values();
    }
    update_beta(final);
  } else {
    dblvec start = get_start_values(true,false,false);
    bobyqad op(start);
    op.control.trace = trace;
    op.control.rhobeg = control.rhobeg;
    op.control.rhoend = control.rhoend;
    op.control.npt = control.npt;
    if constexpr (std::is_same_v<modeltype,bits>)
    {
      op.fn<&glmmr::ModelOptim<bits>::log_likelihood_beta, glmmr::ModelOptim<bits> >(this);
    } else if constexpr (std::is_same_v<modeltype,bits_nngp>) {
      op.fn<&glmmr::ModelOptim<bits_nngp>::log_likelihood_beta, glmmr::ModelOptim<bits_nngp> >(this);
    } else if constexpr (std::is_same_v<modeltype,bits_hsgp>){
      op.fn<&glmmr::ModelOptim<bits_hsgp>::log_likelihood_beta, glmmr::ModelOptim<bits_hsgp> >(this);
    }
    op.minimise();
    dblvec final = op.values();
  }
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::ml_theta(){
  if(re.scaled_u_.cols() != re.u_.cols())re.scaled_u_.conservativeResize(NoChange,re.u_.cols());
  re.scaled_u_ = model.covariance.Lu(re.u_);
  if(control.direct)
  {
    dblvec lower = get_lower_values(false,true,false);
    dblvec upper(lower.size());
    std::fill(upper.begin(),upper.end(),1.0);
    directd op(lower,upper,false);
    op.control.max_iter = control.max_iter_direct;
    op.control.epsilon = control.epsilon;
    op.control.select_one = control.select_one;
    op.control.trisect_once = control.trisect_once;
    op.control.trace = trace;
    op.control.mrdirect = control.mrdirect;
    if(bobyqa)op.control.max_eval = control.max_eval;
    if constexpr (std::is_same_v<modeltype,bits>)
    {
      op.fn<&glmmr::ModelOptim<bits>::log_likelihood_theta, glmmr::ModelOptim<bits> >(this);
    } else if constexpr (std::is_same_v<modeltype,bits_nngp>) {
      op.fn<&glmmr::ModelOptim<bits_nngp>::log_likelihood_theta, glmmr::ModelOptim<bits_nngp> >(this);
    } else if constexpr (std::is_same_v<modeltype,bits_hsgp>){
      op.fn<&glmmr::ModelOptim<bits_hsgp>::log_likelihood_theta_hsgp, glmmr::ModelOptim<bits_hsgp> >(this);
    }
    op.minimise();
    dblvec final = op.values();
    if(control.bobyqa_w_direct){
      bobyqad opb(final);
      opb.control.trace = trace;
      opb.control.rhobeg = op.rect_size();
      opb.control.rhoend = op.rect_size() * 1e-3;
      opb.set_bounds(lower);
      if constexpr (std::is_same_v<modeltype,bits>)
      {
        opb.fn<&glmmr::ModelOptim<bits>::log_likelihood_theta, glmmr::ModelOptim<bits> >(this);
      } else if constexpr (std::is_same_v<modeltype,bits_nngp>) {
        opb.fn<&glmmr::ModelOptim<bits_nngp>::log_likelihood_theta, glmmr::ModelOptim<bits_nngp> >(this);
      } else if constexpr (std::is_same_v<modeltype,bits_hsgp>){
        opb.fn<&glmmr::ModelOptim<bits_hsgp>::log_likelihood_theta_hsgp, glmmr::ModelOptim<bits_hsgp> >(this);
      }
      opb.minimise();
      final = opb.values();
    }
    update_beta(final);
  } else {
    dblvec start = get_start_values(false,true,false);
    bobyqad op(start);
    op.control.trace = trace;
    op.control.rhobeg = control.rhobeg;
    op.control.rhoend = control.rhoend;
    op.control.npt = control.npt;
    dblvec lower = get_lower_values(false,true,false);
    op.set_bounds(lower);
    if constexpr (std::is_same_v<modeltype,bits>)
    {
      op.fn<&glmmr::ModelOptim<bits>::log_likelihood_theta, glmmr::ModelOptim<bits> >(this);
    } else if constexpr (std::is_same_v<modeltype,bits_nngp>) {
      op.fn<&glmmr::ModelOptim<bits_nngp>::log_likelihood_theta, glmmr::ModelOptim<bits_nngp> >(this);
    } else if constexpr (std::is_same_v<modeltype,bits_hsgp>){
      op.fn<&glmmr::ModelOptim<bits_hsgp>::log_likelihood_theta_hsgp, glmmr::ModelOptim<bits_hsgp> >(this);
    }
    op.minimise();
    dblvec final = op.values();
  }
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::set_optim_control(bool direct, int max_iter, double epsilon, bool select_one, bool trisect_once, 
                                      bool bobyqa, int max_eval, bool mrdirect){
  control.direct = direct;
  control.max_iter_direct = max_iter;
  control.epsilon = epsilon;
  control.select_one = select_one;
  control.trisect_once = trisect_once;
  control.bobyqa_w_direct = bobyqa;
  control.max_eval = max_eval;
  control.mrdirect = mrdirect;
 
 }

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::ml_all(){
  
}


template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::laplace_ml_beta_u(){
  
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::laplace_ml_theta(){
  
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::laplace_ml_beta_theta(){
  
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::log_likelihood_beta(const dblvec& beta){
  model.linear_predictor.update_parameters(beta);
  double ll = log_likelihood();
  return -1*ll;
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::log_likelihood_theta(const dblvec& theta){
    model.covariance.update_parameters(theta);
    double logl = 0;
  #pragma omp parallel for reduction (+:logl)
    for(int i = 0; i < re.scaled_u_.cols(); i++)
    {
      logl += model.covariance.log_likelihood(re.scaled_u_.col(i));
    }
    return -1*logl/re.u_.cols();
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::log_likelihood_theta_hsgp(const dblvec& theta){
  model.covariance.update_parameters(theta);
  double ll = log_likelihood();
  return -1*ll;
}

template<typename modeltype>
inline glmmr::ModelOptim<modeltype>::ModelOptim(modeltype& model_, 
                                                glmmr::ModelMatrix<modeltype>& matrix_,
                                                glmmr::RandomEffects<modeltype>& re_) : model(model_), matrix(matrix_), re(re_) {};

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::set_bobyqa_control(int npt_, double rhobeg_, double rhoend_){
  control.npt = npt_;
  control.rhobeg = rhobeg_;
  control.rhoend = rhoend_;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_beta(const dblvec &beta){
  model.linear_predictor.update_parameters(beta);
}

template<typename modeltype>
inline int glmmr::ModelOptim<modeltype>::P() const {
  return model.linear_predictor.P();
}


template<typename modeltype>
inline int glmmr::ModelOptim<modeltype>::Q() const {
  return model.covariance.Q();
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_beta(const VectorXd &beta){
  model.linear_predictor.update_parameters(beta.array());
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_theta(const dblvec &theta){
  model.covariance.update_parameters(theta);
  re.zu_ = model.covariance.ZLu(re.u_);
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_theta(const VectorXd &theta){
  model.covariance.update_parameters(theta.array());
  re.zu_ = model.covariance.ZLu(re.u_);
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_u(const MatrixXd &u_){
  if(u_.cols()!=re.u(false).cols()){
    re.u_.conservativeResize(Q(),u_.cols());
    re.zu_.resize(Q(),u_.cols());
  }
  re.u_ = u_;
  re.zu_ = model.covariance.ZLu(re.u_);
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::log_likelihood() {
  double ll = 0;
  ArrayXd xb(model.xb());
  
  if(model.weighted){
    if(model.family.family==Fam::gaussian){
#pragma omp parallel for reduction (+:ll) collapse(2)
      for(int j=0; j<re.Zu().cols() ; j++){
        for(int i = 0; i<model.n(); i++){
          ll += glmmr::maths::log_likelihood(model.data.y(i),xb(i) + re.zu_(i,j),
                                             model.data.variance(i)/model.data.weights(i),
                                             model.family.family,model.family.link);
        }
      }
    } else {
#pragma omp parallel for reduction (+:ll) collapse(2)
      for(int j=0; j<re.Zu().cols() ; j++){
        for(int i = 0; i<model.n(); i++){
          ll += model.data.weights(i)*glmmr::maths::log_likelihood(model.data.y(i),xb(i) + re.zu_(i,j),
                                   model.data.variance(i),model.family.family,model.family.link);
        }
      }
      ll *= model.data.weights.sum()/model.n();
    }
  } else {
#pragma omp parallel for reduction (+:ll) collapse(2)
    for(int j=0; j<re.Zu().cols() ; j++){
      for(int i = 0; i<model.n(); i++){
        ll += glmmr::maths::log_likelihood(model.data.y(i),xb(i) + re.zu_(i,j),
                                           model.data.variance(i),model.family.family,
                                           model.family.link);
      }
    }
  }
  
  // to use the calculator object instead... seems to be generally slower so have opted 
  // for specific formulae above. Will try to optimise this in future versions
  // #pragma omp parallel for reduction (+:ll) collapse(2) 
  //  for(int j=0; j<zu_.cols() ; j++){
  //    for(int i = 0; i<n_; i++){
  //      double ozu = offset_(i)+zu_(i,j);
  //      ll += calc_.calculate(i,linpred_.parameters_,linpred_.Xdata_,0,0,ozu)[0];
  //    }
  //  }
  
  return ll/re.Zu().cols();
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::full_log_likelihood(){
  double ll = log_likelihood();
  double logl = 0;
  MatrixXd Lu = model.covariance.Lu(re.u(false));
#pragma omp parallel for reduction (+:logl)
  for(int i = 0; i < Lu.cols(); i++){
    logl += model.covariance.log_likelihood(Lu.col(i));
  }
  logl *= 1/Lu.cols();
  return ll+logl;
}

template<typename modeltype>
inline dblvec glmmr::ModelOptim<modeltype>::get_start_values(bool beta, bool theta, bool var){
  dblvec start;
  if(beta){
    for(const auto& i: model.linear_predictor.parameters)start.push_back(i);
    if(theta)for(const auto& j: model.covariance.parameters_)start.push_back(j);
  } else {
    start = model.covariance.parameters_;
  }
  if(var && (model.family.family==Fam::gaussian||model.family.family==Fam::gamma||model.family.family==Fam::beta)){
    start.push_back(model.data.var_par);
  }
  return start;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::set_bound(const dblvec& bound, bool lower){
#ifdef R_BUILD
  if(bound.size()!=P())Rcpp::stop("Bound not equal to number of parameters");
#endif
  if(lower){
    lower_bound = bound; 
  } else {
    upper_bound = bound;
  }
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::set_theta_bound(const dblvec& bound, bool lower){
  if(lower){
    lower_bound_theta = bound; 
  } else {
    upper_bound_theta = bound;
  }
}


template<typename modeltype>
inline dblvec glmmr::ModelOptim<modeltype>::get_lower_values(bool beta, bool theta, bool var){
#ifndef R_BUILD
  double R_NegInf = -1.0 * std::numeric_limits<double>::infinity();
#endif
  dblvec lower;
  if(beta){
    if(lower_bound.size()==0){
      for(int i = 0; i< P(); i++)lower.push_back(R_NegInf);
    } else {
      lower = lower_bound;
    }
  } 
  if(theta){
    if(lower_bound_theta.size()==0){
      for(int i=0; i< model.covariance.npar(); i++)lower.push_back(1e-6);
    } else {
      for(const auto& par: lower_bound_theta)lower.push_back(par);
    }
  }
  if(var && (model.family.family==Fam::gaussian||model.family.family==Fam::gamma||model.family.family==Fam::beta)){
    lower.push_back(0.0);
  }
  return lower;
}

template<typename modeltype>
inline dblvec glmmr::ModelOptim<modeltype>::get_upper_values(bool beta, bool theta, bool var){
  dblvec upper;
#ifndef R_BUILD
  double R_PosInf = std::numeric_limits<double>::infinity();
#endif
  if(beta){
    if(upper_bound.size()==0){
      for(int i = 0; i< P(); i++)upper.push_back(R_PosInf);
    } else {
      upper = upper_bound;
    }
  } 
  if(theta){
    if(upper_bound_theta.size()==0){
      for(int i=0; i< model.covariance.npar(); i++)upper.push_back(1e-6);
    } else {
      for(const auto& par: upper_bound_theta)upper.push_back(par);
    }
  }
  if(var && (model.family.family==Fam::gaussian||model.family.family==Fam::gamma||model.family.family==Fam::beta)){
    upper.push_back(R_PosInf);
  }
  return upper;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::nr_beta(){
  int niter = re.u(false).cols();
  MatrixXd zd = matrix.linpred();
  ArrayXd sigmas(niter);
//   if(model.linear_predictor.any_nonlinear()){
// #if defined(ENABLE_DEBUG) && defined(R_BUILD)
//     Rcpp::Rcout << "\nNR Beta: using non-linear";
// #endif
//     VectorMatrix score = matrix.b_score();
//     MatrixXd infomat = score.mat.llt().solve(MatrixXd::Identity(P(),P()));
//     VectorXd bplus = infomat*score.vec;
//     for(int i = 0; i < bplus.size(); i++)model.linear_predictor.parameters[i] += bplus(i);
//   } else {
// 
//   }
  
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
  Rcpp::Rcout << "\nNR Beta: XtWX";
#endif
  MatrixXd XtXW = MatrixXd::Zero(P()*niter,P());
  MatrixXd Wu = MatrixXd::Zero(model.n(),niter);
  MatrixXd X = model.linear_predictor.X();
  ArrayXd nvar_par(model.n());
  switch(model.family.family){
  case Fam::gaussian:
    nvar_par = model.data.variance;
    break;
  case Fam::gamma:
    nvar_par = model.data.variance.inverse();
    break;
  case Fam::beta:
    nvar_par = (1+model.data.variance);
    break;
  case Fam::binomial:
    nvar_par = model.data.variance.inverse();
    break;
  default:
    nvar_par.setConstant(1.0);
  }
  
#pragma omp parallel for
  for(int i = 0; i < niter; ++i){
    VectorXd w = glmmr::maths::dhdmu(zd.col(i),model.family);
    w = ((w.array() *nvar_par).inverse() * model.data.weights).matrix();
    // w = ((w.array() *nvar_par) * model.data.weights.inverse()).matrix();
    VectorXd zdu = glmmr::maths::mod_inv_func(zd.col(i), model.family.link);
    VectorXd dmu = glmmr::maths::detadmu(zd.col(i),model.family.link);
    if(model.family.family == Fam::binomial){
      zdu = zdu.cwiseProduct(model.data.variance.matrix());
      dmu = dmu.cwiseProduct(model.data.variance.inverse().matrix());
    }
    ArrayXd resid = (model.data.y - zdu);
    XtXW.block(P()*i, 0, P(), P()) = X.transpose() * w.asDiagonal() * X;
    w = w.cwiseProduct(dmu);
    w = w.cwiseProduct(resid.matrix());
    Wu.col(i) = w;
  }
  XtXW *= (double)1.0/niter;
  MatrixXd XtWXm = XtXW.block(0,0,P(),P());
  for(int i = 1; i<niter; i++) XtWXm += XtXW.block(P()*i,0,P(),P());
  XtWXm = XtWXm.llt().solve(MatrixXd::Identity(P(),P()));
  VectorXd Wum = Wu.rowwise().mean();
  VectorXd bincr = XtWXm * X.transpose() * Wum;
  update_beta(model.linear_predictor.parameter_vector() + bincr);
  calculate_var_par();
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::laplace_nr_beta_u(){
  matrix.W.update();
  VectorXd zd = (matrix.linpred()).col(0);
  VectorXd dmu =  glmmr::maths::detadmu(zd,model.family.link);
  MatrixXd infomat = matrix.observed_information_matrix();
  infomat = infomat.llt().solve(MatrixXd::Identity(P()+Q(),P()+Q()));
  VectorXd zdu =  glmmr::maths::mod_inv_func(zd, model.family.link);
  if(model.family.family == Fam::binomial){
    zdu = zdu.cwiseProduct(model.data.variance.matrix());
    dmu = dmu.cwiseProduct(model.data.variance.inverse().matrix());
  }
  ArrayXd resid = (model.data.y - zdu).array();
  VectorXd w = matrix.W.W();//.array().inverse().matrix();
  w = w.cwiseProduct(dmu);
  w = w.cwiseProduct(resid.matrix());
  VectorXd params(P()+Q());
  params.head(P()) = model.linear_predictor.parameter_vector();
  params.tail(Q()) = re.u_.col(0);
  VectorXd pderiv(P()+Q());
  pderiv.head(P()) = (model.linear_predictor.X()).transpose() * w;
  pderiv.tail(Q()) = matrix.log_gradient(re.u_.col(0));
  params += infomat*pderiv;
  update_beta(params.head(P()));
  update_u(params.tail(Q()));
  calculate_var_par();
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_var_par(const double& v){
  model.data.var_par = v;
  model.data.variance.setConstant(v);
  model.calc.variance = model.data.variance;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_var_par(const ArrayXd& v){
  model.data.variance = v;
  model.calc.variance = model.data.variance;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::calculate_var_par(){
  if(model.family.family==Fam::gaussian){
    // revise this for beta and Gamma re residuals
    int niter = re.u(false).cols();
    ArrayXd sigmas(niter);
    MatrixXd zd = matrix.linpred();
#pragma omp parallel for
    for(int i = 0; i < niter; ++i){
      VectorXd zdu = glmmr::maths::mod_inv_func(zd.col(i), model.family.link);
      ArrayXd resid = (model.data.y - zdu);
      resid *= model.data.weights.sqrt();
      sigmas(i) = (resid - resid.mean()).square().sum()/(resid.size()-1);
    }
    update_var_par(sigmas.mean());
  }
}


// 
// template<typename modeltype>
// inline double glmmr::ModelOptim<modeltype>::L_likelihood::operator()(const dblvec &par) {
//   M.update_beta(par);
//   ll = M.log_likelihood();
//   return -1*ll;
// }
// 
// template<typename modeltype>
// inline double glmmr::ModelOptim<modeltype>::D_likelihood::operator()(const dblvec &par) {
//   M.update_theta(par);
//   logl = 0;
// #pragma omp parallel for reduction (+:logl)
//   for(int i = 0; i < Lu.cols(); i++){
//     logl += M.model.covariance.log_likelihood(Lu.col(i));
//   }
//   return -1*logl/Lu.cols();
// }
// 
// template<typename modeltype>
// inline double glmmr::ModelOptim<modeltype>::Lu_likelihood::operator()(const dblvec &par) {
//   auto first = par.begin();
//   auto last1 = par.begin() + M.P();
//   auto last2 = par.begin() + M.P() + M.Q();
//   dblvec beta(first,last1);
//   dblvec u(last1,last2);
//   MatrixXd umat = Map<MatrixXd>(u.data(),u.size(),1);
//   M.update_beta(beta);
//   M.update_u(umat);
//   ll = M.log_likelihood();
//   for(int i = 0; i < M.Q(); i++){
//     ll -= 0.5*u[i]*u[i];
//   }
//   return -1*ll;
// }
// 
// template<typename modeltype>
// inline double glmmr::ModelOptim<modeltype>::F_likelihood::operator()(const dblvec &par) {
//   auto first = par.begin();
//   auto last1 = par.begin() + M.P();
//   auto last2 = par.begin() + M.P() + G;
//   dblvec beta(first,last1);
//   dblvec theta(last1,last2);
//   M.update_beta(beta);
//   M.update_theta(theta);
//   if(M.model.family.family==Fam::gaussian || M.model.family.family==Fam::gamma || M.model.family.family==Fam::beta)M.update_var_par(par[M.P()+G]);
//   ll = M.full_log_likelihood();
//   if(importance){
//     return -1.0 * log(exp(ll)/ exp(denomD));
//   } else {
//     return -1.0*ll;
//   }
// }
// 
// template<typename modeltype>
// inline double glmmr::ModelOptim<modeltype>::LA_likelihood::operator()(const dblvec &par) {
//   logl = 0;
//   auto start = par.begin();
//   auto end = par.begin()+M.P();
//   dblvec beta(start,end);
//   for(int i = 0; i<M.Q(); i++)v(i,0) = par[M.P() + i];
//   M.update_beta(beta);
//   M.update_u(v);
//   logl = v.col(0).transpose()*v.col(0);
//   ll = M.log_likelihood();
//   if(M.model.family.family!=Fam::gaussian){
//     M.matrix.W.update();
//     LZWZL = M.model.covariance.LZWZL(M.matrix.W.W());
//     LZWdet = glmmr::maths::logdet(LZWZL);
//   }
//   return -1.0*(ll - 0.5*logl - 0.5*LZWdet);
// }
// 
// template<typename modeltype>
// inline double glmmr::ModelOptim<modeltype>::LA_likelihood_cov::operator()(const dblvec &par) {
//   M.update_theta(par);
//   M.matrix.W.update();
//   logl = M.re.u_.col(0).transpose() * M.re.u_.col(0);
//   ll = M.log_likelihood();
//   LZWZL = M.model.covariance.LZWZL(M.matrix.W.W());
//   LZWdet = glmmr::maths::logdet(LZWZL);
//   return -1*(ll - 0.5*logl - 0.5*LZWdet);
// }
// 
// template<typename modeltype>
// inline double glmmr::ModelOptim<modeltype>::LA_likelihood_btheta::operator()(const dblvec &par) {
//   auto start = par.begin();
//   auto end1 = par.begin() +M.P();
//   auto end2 = par.begin() + M.P() + M.model.covariance.npar();
//   dblvec beta(start,end1);
//   dblvec theta(end1,end2);
//   M.update_beta(beta);
//   M.update_theta(theta);
//   ll = M.log_likelihood();
//   logl = M.re.u_.col(0).transpose() * M.re.u_.col(0);
//   M.matrix.W.update();
//   LZWZL = M.model.covariance.LZWZL(M.matrix.W.W());
//   LZWdet = glmmr::maths::logdet(LZWZL);
//   return -1*(ll - 0.5*logl - 0.5*LZWdet);
// }
// 
// template<typename modeltype>
// inline double glmmr::ModelOptim<modeltype>::D_likelihood_hsgp::operator()(const dblvec &par) {
//   M.update_theta(par);
//   logl = M.log_likelihood();
//   return -1*logl;
// }

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::aic(){
  MatrixXd Lu = re.u();
  int dof = P() + model.covariance.npar();
  double logl = 0;
#pragma omp parallel for reduction (+:logl)
  for(int i = 0; i < Lu.cols(); i++){
    logl += model.covariance.log_likelihood(Lu.col(i));
  }
  double ll = log_likelihood();
  
  return (-2*( ll + logl ) + 2*dof); 
}

template<typename modeltype>
inline MatrixXd glmmr::ModelOptim<modeltype>::hessian_numerical(double tol){
  
}

template<typename modeltype>
inline ArrayXd glmmr::ModelOptim<modeltype>::optimum_weights(double N, 
                                                             VectorXd C,
                                                             double tol,
                                                             int max_iter){
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
  if(C.size()!=P())Rcpp::stop("C is wrong size");
#endif 
  
  VectorXd Cvec(C);
  ArrayXd weights = ArrayXd::Constant(model.n(),1.0*model.n());
  VectorXd holder(model.n());
  weights = weights.inverse();
  ArrayXd weightsnew(weights);
  ArrayXd w = (matrix.W.W()).array().inverse();
  std::vector<MatrixXd> ZDZ;
  std::vector<MatrixXd> Sigmas;
  std::vector<MatrixXd> Xs;
  std::vector<glmmr::SigmaBlock> SB(matrix.get_sigma_blocks());
#ifdef R_BUILD
  Rcpp::Rcout << "\n### Preparing data ###";
  Rcpp::Rcout << "\nThere are " << SB.size() << " independent blocks and " << model.n() << " cells.";
#endif
  int maxprint = model.n() < 10 ? model.n() : 10;
  for(auto& sb: SB){
    sparse ZLs = submat_sparse(model.covariance.ZL_sparse(),sb.RowIndexes);
    MatrixXd ZL = sparse_to_dense(ZLs,false);
    MatrixXd S = ZL * ZL.transpose();
    ZDZ.push_back(S);
    Sigmas.push_back(S);
    ArrayXi rows = Map<ArrayXi,Unaligned>(sb.RowIndexes.data(),sb.RowIndexes.size());
    MatrixXd X = glmmr::Eigen_ext::submat(model.linear_predictor.X(),rows,ArrayXi::LinSpaced(P(),0,P()-1));
    Xs.push_back(X);
  }
  
  double diff = 1;
  int block_size;
  MatrixXd M(P(),P());
  int iter = 0;
#ifdef R_BUILD
  Rcpp::Rcout << "\n### Starting optimisation ###";
#endif
  while(diff > tol && iter < max_iter){
    iter++;
#ifdef R_BUILD
    Rcpp::Rcout << "\nIteration " << iter << "\n------------\nweights: [" << weights.segment(0,maxprint).transpose() << " ...]";
#endif
    //add check to remove weights that are below a certain threshold
    if((weights < 1e-8).any()){
      for(unsigned int i = 0 ; i < SB.size(); i++){
        auto it = SB[i].RowIndexes.begin();
        while(it != SB[i].RowIndexes.end()){
          if(weights(*it) < 1e-8){
            weights(*it) = 0;
            int idx = it - SB[i].RowIndexes.begin();
            glmmr::Eigen_ext::removeRow(Xs[i],idx);
            glmmr::Eigen_ext::removeRow(ZDZ[i],idx);
            glmmr::Eigen_ext::removeColumn(ZDZ[i],idx);
            Sigmas[i].conservativeResize(ZDZ[i].rows(),ZDZ[i].cols());
            it = SB[i].RowIndexes.erase(it);
#ifdef R_BUILD
            Rcpp::Rcout << "\n Removing point " << idx << " in block " << i;
#endif
          } else {
            it++;
          }
        }
      }
    }
    
    M.setZero();
    for(unsigned int i = 0 ; i < SB.size(); i++){
      Sigmas[i] = ZDZ[i];
      for(int j = 0; j < Sigmas[i].rows(); j++){
        // sigma_sq
        Sigmas[i](j,j) += w(SB[i].RowIndexes[j])/(N*weights(SB[i].RowIndexes[j]));
      }
      Sigmas[i] = Sigmas[i].llt().solve(MatrixXd::Identity(Sigmas[i].rows(),Sigmas[i].cols()));
      M += Xs[i].transpose() * Sigmas[i] * Xs[i];
    }
    
    //check if positive definite, if not remove the offending column(s)
    bool isspd = glmmr::Eigen_ext::issympd(M);
    if(isspd){
#ifdef R_BUILD
      Rcpp::Rcout << "\n Information matrix not postive definite: ";
#endif
      ArrayXd M_row_sums = M.rowwise().sum();
      int fake_it = 0;
      int countZero = 0;
      for(int j = 0; j < M_row_sums.size(); j++){
        if(M_row_sums(j) == 0){
#ifdef R_BUILD
          Rcpp::Rcout << "\n   Removing column " << fake_it;
#endif
          for(unsigned int k = 0; k < Xs.size(); k++){
            glmmr::Eigen_ext::removeColumn(Xs[k],fake_it);
          }
          glmmr::Eigen_ext::removeElement(Cvec,fake_it);
          countZero++;
        } else {
          fake_it++;
        }
      }
      M.conservativeResize(M.rows()-countZero,M.cols()-countZero);
      M.setZero();
      for(unsigned int k = 0; k < SB.size(); k++){
        M += Xs[k].transpose() * Sigmas[k] * Xs[k];
      }
    }
    M = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
    VectorXd Mc = M*Cvec;
    weightsnew.setZero();
    for(unsigned int i = 0 ; i < SB.size(); i++){
      block_size = SB[i].RowIndexes.size();
      holder.segment(0,block_size) = Sigmas[i] * Xs[i] * Mc;
      for(int j = 0; j < block_size; j++){
        weightsnew(SB[i].RowIndexes[j]) = holder(j);
      }
    }
    weightsnew = weightsnew.abs();
    weightsnew *= 1/weightsnew.sum();
    diff = ((weights-weightsnew).abs()).maxCoeff();
    weights = weightsnew;
#ifdef R_BUILD
    Rcpp::Rcout << "\n(Max. diff: " << diff << ")\n";
#endif
  }
#ifdef R_BUILD
  if(iter<max_iter){
    Rcpp::Rcout << "\n### CONVERGED Final weights: [" << weights.segment(0,maxprint).transpose() << "...]";
  } else {
    Rcpp::Rcout << "\n### NOT CONVERGED Reached maximum iterations";
  }
#endif
  return weights;
}
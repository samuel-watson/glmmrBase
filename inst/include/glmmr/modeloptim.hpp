#pragma once

#include "modelbits.hpp"
#include "randomeffects.hpp"
#include "modelmatrix.hpp"
#include "openmpheader.h"
#include "calculator.hpp"
#include "optim/optim.h"

namespace glmmr {

using namespace Eigen;

template<typename modeltype>
class ModelOptim {
  
public:
  // members 
  modeltype&                        model;
  glmmr::ModelMatrix<modeltype>&    matrix;
  glmmr::RandomEffects<modeltype>&  re;
  int                               trace = 0;
  // ArrayXXd                          ll_previous; // log likelihood values for all u samples
  ArrayXXd                          ll_current;
  std::pair<double,double>          current_ll_values = {0.0,0.0};
  std::pair<double,double>          previous_ll_values = {0.0,0.0};
  std::pair<double,double>          current_ll_var = {0.0,0.0};
  std::pair<double,double>          previous_ll_var = {0.0,0.0};
  std::pair<int,int>                fn_counter = {0,0};
  ArrayXd                           gradients;
  std::deque<double>                gradient_history;
  dblvec                            converge_z;
  dblvec                            converge_bf;
  
  // constructor
  ModelOptim(modeltype& model_, glmmr::ModelMatrix<modeltype>& matrix_,glmmr::RandomEffects<modeltype>& re_) ;

  // control parameters for the optimisers - direct will be removed as its useless.
  struct OptimControl {
    int     npt = 0;
    double  rhobeg = 0;
    double  rhoend = 0;
    double  epsilon = 1e-4; 
    bool    select_one = true; 
    int     max_eval = 0; 
    double  alpha = 0.8;
    bool    saem = false;
    bool    pr_average = true;
    bool    reml = true;
  } control;
  
  // functions
  virtual void    update_beta(const dblvec &beta);
  virtual void    update_beta(const VectorXd &beta);
  virtual void    update_theta(const dblvec &theta);
  virtual void    update_theta(const VectorXd &theta);
  virtual void    update_u(const MatrixXd& u_, bool append); // two versions needed so CRAN won't complain with linked package
  virtual void    update_u(const MatrixXd& u_); 
  virtual double  log_likelihood(bool beta);
  virtual double  log_likelihood();
  virtual double  full_log_likelihood();
  virtual double  marginal_log_likelihood();
  virtual void    nr_beta();
  virtual void    nr_theta();
  virtual void    nr_beta_gaussian();
  virtual void    nr_theta_gaussian();
  virtual void    update_var_par(const double& v);
  virtual void    update_var_par(const ArrayXd& v);
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            ml_beta();
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            ml_theta();
  virtual double  aic();
  virtual ArrayXd optimum_weights(double N, VectorXd C, double tol = 1e-5, int max_iter = 501);
  void            set_bobyqa_control(int npt_, double rhobeg_, double rhoend_);
  void            set_bound(const dblvec& bound, bool lower = true);
  void            set_theta_bound(const dblvec& bound, bool lower = true);
  void            use_reml(bool reml);
  int             P() const;
  int             Q() const;
  double          ll_diff_variance(bool beta = true, bool theta = true);
  std::pair<double,double>  current_likelihood_values();
  std::pair<double,double>  u_diagnostic();
  void            reset_fn_counter();
  void            set_quantile(const double& q);
  // functions to optimise
  double          log_likelihood_beta(const dblvec &beta);
  double          log_likelihood_theta(const dblvec &theta);
  double          log_likelihood_all(const dblvec &par);
  bool            check_convergence(const double tol, const int hist, const int k, const int k0);
  
protected:
// objects
  dblvec    lower_bound;
  dblvec    upper_bound; // bounds for beta
  dblvec    lower_bound_theta;
  dblvec    upper_bound_theta; // bounds for beta
  bool      beta_bounded = false;
  double    quantile = 0;
  
  // functions
  void            calculate_var_par();
  dblvec          get_start_values(bool beta, bool theta, bool var = true);
  dblvec          get_lower_values(bool beta, bool theta, bool var = true, bool u = false);
  dblvec          get_upper_values(bool beta, bool theta, bool var = true, bool u = false);
  void            set_bobyqa_control(bobyqad& op);
  void            set_newuoa_control(newuoad& op);
  
private:
  
  // used for REML
  void        generate_czz();
  MatrixXd    CZZ = MatrixXd::Zero(1,1);
  double      saem_average(const int col);
  void        add_reml_corr(const int col);
};

}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::set_bobyqa_control(bobyqad& op){
  op.control.trace = trace;
  op.control.rhobeg = control.rhobeg;
  op.control.rhoend = control.rhoend;
  op.control.npt = control.npt;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::set_newuoa_control(newuoad& op){
  op.control.trace = trace;
  op.control.rhobeg = control.rhobeg;
  op.control.rhoend = control.rhoend;
  op.control.npt = control.npt;
}

template<typename modeltype>
template<class algo, typename>
inline void glmmr::ModelOptim<modeltype>::ml_beta(){
  dblvec start = get_start_values(true,false,false);
  if(ll_current.rows() != re.u_.cols())ll_current.resize(re.u_.cols(),NoChange);
  // store previous log likelihood values for convergence calculations
  previous_ll_values.first = current_ll_values.first;
  previous_ll_var.first = current_ll_var.first;
  
  optim<double(const std::vector<double>&),algo> op(start);
  if constexpr (std::is_same_v<algo,BOBYQA>) {
    set_bobyqa_control(op);
  } else if constexpr (std::is_same_v<algo,NEWUOA>) {
    set_newuoa_control(op);
  }
  if(beta_bounded) op.set_bounds(lower_bound,upper_bound);
  if constexpr (std::is_same_v<modeltype,bits>)
  {
    op.template fn<&glmmr::ModelOptim<bits>::log_likelihood_beta, glmmr::ModelOptim<bits> >(this);
  } else if constexpr (std::is_same_v<modeltype,bits_nngp>) {
    op.template fn<&glmmr::ModelOptim<bits_nngp>::log_likelihood_beta, glmmr::ModelOptim<bits_nngp> >(this);
  } else if constexpr (std::is_same_v<modeltype,bits_hsgp>){
    op.template fn<&glmmr::ModelOptim<bits_hsgp>::log_likelihood_beta, glmmr::ModelOptim<bits_hsgp> >(this);
  } else if constexpr (std::is_same_v<modeltype,bits_ar1>){
    op.template fn<&glmmr::ModelOptim<bits_ar1>::log_likelihood_beta, glmmr::ModelOptim<bits_ar1> >(this);
  } else if constexpr (std::is_same_v<modeltype,bits_spde>){
    op.template fn<&glmmr::ModelOptim<bits_spde>::log_likelihood_beta, glmmr::ModelOptim<bits_spde> >(this);
  }
  op.minimise();
  int eval_size = control.saem ? re.mcmc_block_size : ll_current.rows();
  current_ll_values.first = ll_current.col(0).tail(eval_size).mean();
  current_ll_var.first = (ll_current.col(0).tail(eval_size) - ll_current.col(0).tail(eval_size).mean()).square().sum() / (eval_size - 1);
  
}

template<typename modeltype>
template<class algo, typename>
inline void glmmr::ModelOptim<modeltype>::ml_theta(){ 
  if(model.covariance.parameters_.size()==0)throw std::runtime_error("no covariance parameters, cannot calculate log likelihood");
  if(ll_current.rows() != re.u_.cols())ll_current.resize(re.u_.cols(),NoChange);
  dblvec start = get_start_values(false,true,false);  
  dblvec lower = get_lower_values(false,true,false);
  dblvec upper = get_upper_values(false,true,false);
  // store previous log likelihood values for convergence calculations
  previous_ll_values.second = current_ll_values.second;
  previous_ll_var.second = current_ll_var.second;
  if(re.scaled_u_.cols() != re.u_.cols())re.scaled_u_.resize(NoChange,re.u_.cols());
  re.scaled_u_ = model.covariance.Lu(re.u_);  
  if(control.reml) generate_czz();
  // optimisation
  optim<double(const std::vector<double>&),algo> op(start);
  if constexpr (std::is_same_v<algo,BOBYQA>) {
    set_bobyqa_control(op);
    op.set_bounds(lower,upper);
  } else if constexpr (std::is_same_v<algo,NEWUOA>) {
    set_newuoa_control(op);
    op.set_bounds(lower,upper);
  }
  if constexpr (std::is_same_v<modeltype,bits>)
  {
    op.template fn<&glmmr::ModelOptim<bits>::log_likelihood_theta, glmmr::ModelOptim<bits> >(this);
  } else if constexpr (std::is_same_v<modeltype,bits_nngp>) {
    op.template fn<&glmmr::ModelOptim<bits_nngp>::log_likelihood_theta, glmmr::ModelOptim<bits_nngp> >(this);
  } else if constexpr (std::is_same_v<modeltype,bits_hsgp>){
    op.template fn<&glmmr::ModelOptim<bits_hsgp>::log_likelihood_theta, glmmr::ModelOptim<bits_hsgp> >(this);
  } else if constexpr (std::is_same_v<modeltype,bits_ar1>){
    op.template fn<&glmmr::ModelOptim<bits_ar1>::log_likelihood_theta, glmmr::ModelOptim<bits_ar1> >(this);
  }else if constexpr (std::is_same_v<modeltype,bits_spde>){
    op.template fn<&glmmr::ModelOptim<bits_spde>::log_likelihood_theta, glmmr::ModelOptim<bits_spde> >(this);
  }
  op.minimise();
  int eval_size = control.saem ? re.mcmc_block_size : ll_current.rows();
  current_ll_values.second = ll_current.col(1).tail(eval_size).mean();
  current_ll_var.second = (ll_current.col(1).tail(eval_size) - ll_current.col(1).tail(eval_size).mean()).square().sum() / (eval_size - 1);
  calculate_var_par();
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::reset_fn_counter()
{
  fn_counter.first = 0;
  fn_counter.second = 0;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::set_quantile(const double& q)
{
  if(q <= 0 || q >= 1)throw std::runtime_error("q !in [0,1]");
  quantile = q;
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::ll_diff_variance(bool beta, bool theta)
{
  double var = 0;
  if(beta) var += current_ll_var.first + previous_ll_var.first;
  if(theta) var += current_ll_var.second + previous_ll_var.second;
  return var ; 
}

template<typename modeltype>
inline std::pair<double,double> glmmr::ModelOptim<modeltype>::current_likelihood_values()
{
  return current_ll_values;
}

template<>
inline double glmmr::ModelOptim<bits_hsgp>::marginal_log_likelihood(){
  int n = model.data.y.size();
  int p = model.linear_predictor.P();
  int Mspec = model.covariance.Q();
  
  MatrixXd ZPhi = model.covariance.ZPhi();
  MatrixXd X = model.linear_predictor.X();
  ArrayXd Lambda = model.covariance.LambdaSPD();
  ArrayXd inv_lambda = 1.0 / Lambda;
  
  // Compute eta, mu, W at current beta and posterior mode of u
  VectorXd b = re.u_mean_;
  ArrayXd eta = (model.linear_predictor.xb() + ZPhi * b).array() 
    + model.data.offset.array();
  ArrayXd mu = maths::mod_inv_func(eta.matrix(), model.family.link).array();
  
  ArrayXd W_inv;  // working variance per observation
  ArrayXd working_resid;  // (y - mu) * d eta/d mu
  
  switch(model.family.family){
  case Fam::gaussian:
    W_inv = (model.data.variance / model.data.weights).matrix();
    working_resid = model.data.y.array() - mu;
    break;
  case Fam::binomial: case Fam::bernoulli: {
    ArrayXd p_logit = mu / model.data.variance;  // probability
    W_inv = 1.0 / (model.data.variance * p_logit * (1.0 - p_logit));
    working_resid = (model.data.y.array() - mu) * W_inv;  // (y - mu) / W = d eta/d mu * (y - mu)
    break;
  }
  case Fam::poisson:
    W_inv = 1.0 / mu;
    working_resid = (model.data.y.array() - mu) / mu;  // d eta/d mu = 1/mu
    break;
  default:
    throw std::runtime_error("Marginal QL only for Gaussian/Binomial/Poisson");
  }
  
  // Working response: y_tilde = X*beta + working_resid  (relative to fixed effect part)
  // Equivalently the residual r = working_resid (since we subtract X*beta below)
  VectorXd r = working_resid.matrix();
  
  // V = diag(W_inv) + ZPhi * diag(Lambda) * ZPhi^T
  // log|V| via Sylvester: log|V| = sum(log W_inv) + log|I + Lambda^{1/2} ZPhi^T W ZPhi Lambda^{1/2}|
  // Or use the same M-matrix trick as Gaussian but with W in place of sigma^{-2} I
  
  ArrayXd W_diag = 1.0 / W_inv;  // the IRLS weights
  MatrixXd WZPhi = (ZPhi.array().colwise() * W_diag).matrix();
  MatrixXd G = ZPhi.transpose() * WZPhi;  // Phi^T W Phi (n cancels into W)
  VectorXd c = WZPhi.transpose() * r;
  
  MatrixXd Mmat = G;
  Mmat.diagonal() += inv_lambda.matrix();
  LLT<MatrixXd> llt_M(Mmat);
  VectorXd M_inv_c = llt_M.solve(c);
  
  // log|V| = sum(log W_inv) + log|D| + log|M|  
  //        = -sum(log W) + sum(log Lambda) + log|M|
  double logdet_Winv = -W_diag.log().sum();
  double logdetD = Lambda.log().sum();
  double logdetM = 2.0 * llt_M.matrixL().toDenseMatrix().diagonal().array().log().sum();
  double logdetV = logdet_Winv + logdetD + logdetM;
  
  // r^T V^{-1} r via Woodbury
  double rWr = (r.array().square() * W_diag).sum();
  double quadform = rWr - c.dot(M_inv_c);
  
  double ll = -0.5 * (n * std::log(2*M_PI) + logdetV + quadform);
  
  return ll;
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::marginal_log_likelihood(){
  int n = model.data.y.size();
  int p = model.linear_predictor.P();
  int q = model.covariance.Q();
  
  double sigma2 = model.data.var_par;
  double sigma2_inv = 1.0 / sigma2;
  double sigma4_inv = sigma2_inv * sigma2_inv;
  
  MatrixXd Z = model.covariance.Z();
  MatrixXd X = model.linear_predictor.X();
  VectorXd r = model.data.y.matrix() - model.linear_predictor.xb();
  
  MatrixXd G = Z.transpose() * Z;
  MatrixXd C = Z.transpose() * X;
  VectorXd c = Z.transpose() * r;
  
  std::vector<MatrixXd> derivs;
  model.covariance.derivatives(derivs, 1);
  MatrixXd& D = derivs[0];
  
  LLT<MatrixXd> llt_D(D);
  MatrixXd D_inv = llt_D.solve(MatrixXd::Identity(q, q));
  MatrixXd M = D_inv + sigma2_inv * G;
  LLT<MatrixXd> llt_M(M);
  VectorXd M_inv_c = llt_M.solve(c);
  
  // log|V| = log|D| + n*log(sigma2) + log|M|
  double logdetD = 2.0 * llt_D.matrixL().toDenseMatrix().diagonal().array().log().sum();
  double logdetM = 2.0 * llt_M.matrixL().toDenseMatrix().diagonal().array().log().sum();
  double logdetV = logdetD + n * std::log(sigma2) + logdetM;
  
  // r^T V^{-1} r = sigma^{-2}*r^T*r - sigma^{-4}*c^T*M^{-1}*c
  double rtr = r.squaredNorm();
  double quadform = sigma2_inv * rtr - sigma4_inv * c.dot(M_inv_c);
  
  double ll;
  if(control.reml){
    // REML: add -0.5*log|X^T V^{-1} X|
    MatrixXd XtX = X.transpose() * X;
    MatrixXd M_inv_C = llt_M.solve(C);
    MatrixXd XtVinvX = sigma2_inv * XtX - sigma4_inv * C.transpose() * M_inv_C;
    LLT<MatrixXd> llt_XtVinvX(XtVinvX);
    double logdetXtVinvX = 2.0 * llt_XtVinvX.matrixL().toDenseMatrix().diagonal().array().log().sum();
    
    // Quadratic form uses P not V^{-1}
    VectorXd XtVinvr = sigma2_inv * X.transpose() * r - sigma4_inv * C.transpose() * M_inv_c;
    VectorXd adj = llt_XtVinvX.solve(XtVinvr);
    double adj_quadform = XtVinvr.dot(adj);
    quadform -= adj_quadform;
    
    ll = -0.5 * ((n - p) * std::log(2 * M_PI) + logdetV + logdetXtVinvX + quadform);
  } else {
    ll = -0.5 * (n * std::log(2 * M_PI) + logdetV + quadform);
  }
  
  return ll;
}

template<typename modeltype>
inline std::pair<double,double> glmmr::ModelOptim<modeltype>::u_diagnostic()
{
  std::pair<double, double> ll;
  ll.first = current_ll_values.first - previous_ll_values.first;
  ll.second = current_ll_values.second - previous_ll_values.second;
  return ll;
}


template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::saem_average(const int col){
  double  ll = 0;
  int     iteration = std::max((int)re.zu_.cols() / re.mcmc_block_size, 1);
  double  gamma = pow(1.0/iteration,control.alpha);
  double  ll_t = 0;
  double  ll_pr = 0;
  for(int i = 0; i < iteration; i++){
    int lower_range = i * re.mcmc_block_size;
    int upper_range = (i + 1) * re.mcmc_block_size;
    if(i == (iteration - 1) && iteration > 1){
      double ll_t_c = ll_t;
      double ll_pr_c = ll_pr;
      ll_t = ll_t + gamma*(ll_current.col(col).segment(lower_range, re.mcmc_block_size).mean() - ll_t);
      if(control.pr_average) ll_pr += ll_t;
      for(int j = lower_range; j < upper_range; j++)
      {
        ll_current(j,col) = ll_t_c + gamma*(ll_current(j,col) - ll_t_c);
        if(control.pr_average) ll_current(j,col) = (ll_current(j,col) + ll_pr_c)/((double)iteration);
      }
    } else {
      ll_t = ll_t + gamma*(ll_current.col(col).segment(lower_range, re.mcmc_block_size).mean() - ll_t);
      if(control.pr_average) ll_pr += ll_t;
    }
  }
  if(control.pr_average){
    ll = ll_pr / (double)iteration;
  } else {
    ll = ll_t;
  }
  return ll;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::add_reml_corr(const int col){
  // REML correction to the log-likelihood
  MatrixXd D = model.covariance.D().llt().solve(MatrixXd::Identity(Q(),Q()));
  double trCZZ = 0;
  for(int i = 0; i < Q(); i++){
    for(int j = 0; j < Q(); j++){
      trCZZ += D(i,j)*CZZ(j,i);
    }
  }
  ll_current.col(col).array() += -0.5*trCZZ;
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::log_likelihood_all(const dblvec &par)
{
  int G = model.covariance.npar();
  auto first = par.begin();
  auto last1 = par.begin() + P();
  auto last2 = par.begin() + P() + G;
  dblvec beta(first,last1);
  dblvec theta(last1,last2);
  model.linear_predictor.update_parameters(beta);
  model.covariance.update_parameters(theta);
  re.zu_ = model.covariance.ZLu(re.u_);
  fn_counter.second += re.scaled_u_.cols();
  double ll = log_likelihood();
  if(control.reml)add_reml_corr(0);
  if(control.saem) ll = saem_average(0);
  ll_current.col(1) = ll_current.col(0);
  return -1*ll;
}



template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::log_likelihood_beta(const dblvec& beta)
{
  model.linear_predictor.update_parameters(beta);
  double ll = log_likelihood();
  fn_counter.first += re.scaled_u_.cols();
  if(control.saem)ll = saem_average(0);
  return -1*ll;
}


template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::log_likelihood_theta(const dblvec& theta)
{
  model.covariance.update_parameters(theta);
  fn_counter.second += re.scaled_u_.cols();
//#pragma omp parallel if(re.scaled_u_.cols() > 300)
  for(int i = 0; i < re.scaled_u_.cols(); i++)
  {
    ll_current(i,1) = model.covariance.log_likelihood(re.scaled_u_.col(i));
  }
  if(control.reml) add_reml_corr(1);
  double ll = 0;
  if(control.saem){
    ll = saem_average(1);
  } else {
    for(int i = 0; i < ll_current.rows(); i++) ll += re.u_weight_(i) * ll_current(i,1);
    //ll = ll_current.col(1).mean();
  }
  return -1*ll;
}

template<>
inline double glmmr::ModelOptim<bits_hsgp>::log_likelihood_theta(const dblvec& theta){
  if(control.reml) throw std::runtime_error("REML not currently available with HSGP");
  model.covariance.update_parameters(theta);
  re.zu_ = model.covariance.ZLu(re.u_);
  double ll = log_likelihood(false);
  fn_counter.first += re.scaled_u_.cols();
  if(control.saem)ll = saem_average(1);
  return -1*ll;
}

template<typename modeltype>
inline glmmr::ModelOptim<modeltype>::ModelOptim(modeltype& model_, 
                                                glmmr::ModelMatrix<modeltype>& matrix_,
                                                glmmr::RandomEffects<modeltype>& re_) : model(model_), matrix(matrix_), re(re_), ll_current(ArrayXXd::Zero(re_.mcmc_block_size,2)), 
                                                gradients(model.linear_predictor.P() + model.covariance.npar()) {}; //ll_previous(ArrayXXd::Zero(re_.mcmc_block_size,2)), 

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::set_bobyqa_control(int npt_, double rhobeg_, double rhoend_){
  control.npt = npt_;
  control.rhobeg = rhobeg_;
  control.rhoend = rhoend_;
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
inline void glmmr::ModelOptim<modeltype>::update_beta(const dblvec &beta){
  bool update = true;
  if(beta_bounded)
  {
    for(int i = 0; i < beta.size(); i++)
    {
      if(beta[i] < lower_bound[i] || beta[i] > upper_bound[i]) 
      {
        update = false;
        throw std::runtime_error("beta out of bounds");
      }
    }
  }
  if(update) model.linear_predictor.update_parameters(beta);
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_beta(const VectorXd &beta){
  bool update = true;
  if(beta_bounded)
  {
    for(int i = 0; i < beta.size(); i++)
    {
      if(beta(i) < lower_bound[i] || beta(i) > upper_bound[i]) 
      {
        update = false;
        throw std::runtime_error("beta out of bounds");
      }
    }
  }
  if(update)model.linear_predictor.update_parameters(beta.array());
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
inline void glmmr::ModelOptim<modeltype>::update_u(const MatrixXd &u_, bool append){
  bool action_append = append;
  // if HSGP then check and update the size of u is m has changed
  if constexpr (std::is_same_v<modeltype,bits_hsgp>){
    if(model.covariance.Q() != re.u_.rows()){
      re.u_.resize(model.covariance.Q(),1);
      re.u_.setZero();
    }
  }
  
  int newcolsize = u_.cols();
  int currcolsize = re.u_.cols();
  // check if the existing samples are a single column of zeros - if so remove them
  if(append && re.u_.cols() == 1 && re.u_.col(0).isZero()) action_append = false;
  // update stored ll values 
  // if(ll_previous.rows() != ll_current.rows()) ll_previous.resize(ll_current.rows(),NoChange);
  // ll_previous = ll_current;
  
  if(action_append){
    re.u_.conservativeResize(NoChange,currcolsize + newcolsize);
    re.zu_.conservativeResize(NoChange,currcolsize + newcolsize);
    re.u_.rightCols(newcolsize) = u_;
    ll_current.resize(currcolsize + newcolsize,NoChange);
  } else {
    if(u_.cols()!=re.u_.cols()){
      re.u_.resize(NoChange,newcolsize);
      re.zu_.resize(NoChange,newcolsize);
    }
    re.u_ = u_;
    if(re.u_.cols() != ll_current.rows()) ll_current.resize(newcolsize,NoChange);
  }
  re.zu_ = model.covariance.ZLu(re.u_);
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_u(const MatrixXd &u_){
  int newcolsize = u_.cols();  
  if(u_.cols()!=re.u_.cols()){
      re.u_.resize(NoChange,newcolsize);
      re.zu_.resize(NoChange,newcolsize);
    }
  re.u_ = u_;
  if(newcolsize != ll_current.rows()) ll_current.resize(newcolsize,NoChange);
  re.zu_ = model.covariance.ZLu(re.u_);
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::log_likelihood(bool beta) {
  ArrayXd xb(model.xb());
  int llcol = beta ? 0 : 1;
  ll_current.col(llcol).setZero();
  
  if(model.weighted){
    if(model.family.family==Fam::gaussian){
#pragma omp parallel for 
      for(int j= 0; j< re.zu_.cols() ; j++){
        ll_current(j,llcol ) = glmmr::maths::log_likelihood(model.data.y.array(),xb + re.zu_.col(j).array(),
                   model.data.variance * model.data.weights.inverse(),
                   model.family);
      }
    } else {
//#pragma omp parallel for 
      for(int j=0; j< re.zu_.cols() ; j++){
        for(int i = 0; i<model.n(); i++){
          ll_current(j,llcol) += model.data.weights(i)*glmmr::maths::log_likelihood(model.data.y(i),xb(i) + re.zu_(i,j),
                                   model.data.variance(i),model.family);
        }
      }
      ll_current.col(llcol) *= model.data.weights.sum()/model.n();
    }
  } else {
#pragma omp parallel for if(re.zu_.cols() > 50)
    for(int j= 0; j< re.zu_.cols() ; j++){
      ll_current(j,llcol) = glmmr::maths::log_likelihood(model.data.y.array(),xb + re.zu_.col(j).array(),
                 model.data.variance,model.family);
    }
  }
  double out = 0;
  for(int j = 0; j< ll_current.rows(); j++) out += re.u_weight_(j) * ll_current(j,llcol);
  return out; //ll_current.col(llcol).mean();
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::log_likelihood(){
  double ll = log_likelihood(true);
  return ll;
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::full_log_likelihood(){
  double ll = log_likelihood();
  double logl = 0;
  MatrixXd Lu = model.covariance.Lu(re.u(false));
#pragma omp parallel for reduction (+:logl) if(Lu.cols() > 50)
  for(int i = 0; i < Lu.cols(); i++){
    logl += model.covariance.log_likelihood(Lu.col(i));
  }
  logl *= 1/Lu.cols();
  return ll+logl;
}

template<typename modeltype>
inline dblvec glmmr::ModelOptim<modeltype>::get_start_values(bool beta, bool theta, bool var)
{
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
inline void glmmr::ModelOptim<modeltype>::set_bound(const dblvec& bound, bool lower)
{
  if(static_cast<int>(bound.size())!=P())throw std::runtime_error("Bound not equal to number of parameters");
  if(lower){
    if(lower_bound.size() != bound.size())lower_bound.resize(P());
    lower_bound = bound; 
  } else {
    if(upper_bound.size() != bound.size())upper_bound.resize(P());
    upper_bound = bound;
  }
  beta_bounded = true;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::set_theta_bound(const dblvec& bound, bool lower)
{
  if(lower){
    lower_bound_theta = bound; 
  } else {
    upper_bound_theta = bound;
  }
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::use_reml(bool reml)
{
  control.reml = reml;
}


template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::generate_czz()
{
  CZZ.resize(Q(),Q());
  CZZ = MatrixXd::Identity(Q(), Q());
  matrix.W.update(re.centred_u_mean());
  VectorXd w = matrix.W.W();
  // w = w.array().inverse().matrix();
  bool nonlinear_w = model.family.family != Fam::gaussian || (model.data.weights != 1).any();
  if (control.reml) {
      MatrixXd X = model.linear_predictor.X();
      MatrixXd WX = X;
      if (nonlinear_w) {
          WX = w.asDiagonal() * X;
      }
      MatrixXd XtWX = X.transpose() * WX;
      WX = model.covariance.Z().transpose() * WX;
      // WX.applyOnTheLeft(model.covariance.Z().transpose());
      XtWX = XtWX.llt().solve(MatrixXd::Identity(P(), P()));
      MatrixXd WZ = model.covariance.Z();
      if (nonlinear_w) {
        WZ = w.asDiagonal() * WZ;
          //WZ.applyOnTheLeft(w.asDiagonal());
      }

      MatrixXd ZWZ = model.covariance.Z().transpose() * WZ;
      CZZ = ZWZ - WX * (XtWX * WX.transpose());
      if (!nonlinear_w) CZZ *= 1.0 / model.data.var_par;
  }
  else {
      MatrixXd WZ = model.covariance.Z();
      if (nonlinear_w) {
        WZ = w.asDiagonal() * WZ;
          //WZ.applyOnTheLeft(w.asDiagonal());
      }
      CZZ = model.covariance.Z().transpose() * WZ;
      if (!nonlinear_w) CZZ *= 1.0 / model.data.var_par;
  }
  MatrixXd D = model.covariance.D();
  //D = D.llt().solve(MatrixXd::Identity(Q(), Q()));
  
  if (model.covariance.all_group_re()) {
      for (int i = 0; i < D.rows(); i++) D(i, i) = 1 / D(i, i);
  }
  else {
      D = D.llt().solve(MatrixXd::Identity(Q(), Q()));
  }  
  CZZ += D;
  CZZ = CZZ.llt().solve(MatrixXd::Identity(Q(), Q()));
}

template<typename modeltype>
inline dblvec glmmr::ModelOptim<modeltype>::get_lower_values(bool beta, bool theta, bool var, bool u)
{
#ifndef R_BUILD
  double R_NegInf = -1.0 * std::numeric_limits<double>::infinity();
#endif
  dblvec lower;
  if(beta){
    if(lower_bound.size()==0)
    {
      for(int i = 0; i< P(); i++)lower.push_back(R_NegInf);
    } else {
      lower = lower_bound;
    }
  } 
  if(theta)
  {
    bool all_log = model.covariance.all_log_re();
    if(model.covariance.any_log_re() && !all_log) throw std::runtime_error("Requires all covariance functions to be on log scale");
    if(lower_bound_theta.size()==0)
    {
      if(all_log){
        for(int i=0; i< model.covariance.npar(); i++)lower.push_back(R_NegInf);
      } else {
        for(int i=0; i< model.covariance.npar(); i++)lower.push_back(1e-6);
      }
    } else {
      for(const auto& par: lower_bound_theta)lower.push_back(par);
    }
  }
  if(var && (model.family.family==Fam::gaussian||model.family.family==Fam::gamma||model.family.family==Fam::beta))
  {
    lower.push_back(0.0);
  }
  if(u)
  {
    for(int i = 0; i< Q(); i++) lower.push_back(R_NegInf);
  }
  return lower;
}

template<typename modeltype>
inline dblvec glmmr::ModelOptim<modeltype>::get_upper_values(bool beta, bool theta, bool var, bool u){
  dblvec upper;
#ifndef R_BUILD
  double R_PosInf = std::numeric_limits<double>::infinity();
#endif
  if(beta)
  {
    if(upper_bound.size()==0){
      for(int i = 0; i< P(); i++)upper.push_back(R_PosInf);
    } else {
      upper = upper_bound;
    }
  } 
  if(theta)
  {
    if(upper_bound_theta.size()==0){
      for(int i=0; i< model.covariance.npar(); i++)upper.push_back(R_PosInf);
    } else {
      for(const auto& par: upper_bound_theta)upper.push_back(par);
    }
  }
  if(var && (model.family.family==Fam::gaussian||model.family.family==Fam::gamma||model.family.family==Fam::beta))
  {
    upper.push_back(R_PosInf);
  }
  if(u)
  {
    for(int i = 0; i < Q(); i++) upper.push_back(R_PosInf);
  }
  return upper;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::nr_beta(){
  if(re.u_.cols() != ll_current.rows()) ll_current.resize(re.u_.cols(),NoChange);
  // save the old likelihood values
  previous_ll_values.first = current_ll_values.first;
  previous_ll_var.first = current_ll_var.first;
  if(trace > 1)Rcpp::Rcout << "\nESS: " << 1.0 / re.u_weight_.square().sum();
  
  int niter = re.u(false).cols();
  MatrixXd zd = matrix.linpred();
  MatrixXd X = model.linear_predictor.X();
  
  VectorXd xb = X * model.linear_predictor.parameter_vector();
  double xb_mean = xb.mean();
  if(zd.cols() > 1){
    for(int i = 0; i < zd.cols(); i++){
      double zu_mean = zd.col(i).mean() - xb_mean;
      zd.col(i).array() -= zu_mean;
    }
  }
  
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

  MatrixXd XtWXm = MatrixXd::Zero(P(), P());
  MatrixXd W = glmmr::maths::dhdmu(zd, model.family);
  W = (W.array().colwise() * nvar_par).inverse();
  W.array().colwise() *= model.data.weights;

  zd = maths::mod_inv_func(zd, model.family.link);
  if(model.family.family == Fam::binomial) zd.array().colwise() *= model.data.variance;
  VectorXd resid = VectorXd::Zero(model.n());
  
  if(!model.family.canonical()){
    MatrixXd zdresid = MatrixXd::Zero(model.n(), zd.cols());
    zdresid.colwise() += model.data.y;
    zdresid -= zd;
    MatrixXd detmu = maths::detadmu(zd, model.family.link);
    zdresid.array() *= detmu.array();
    for(int i = 0; i < niter; ++i){
      resid += re.u_weight_(i) * (W.col(i).array() * zdresid.col(i).array()).matrix();
    }
  } else {
    for(int i = 0; i < niter; ++i){
      resid += re.u_weight_(i) * (model.data.y - zd.col(i));
    }
  }

  #pragma omp parallel
  {
    MatrixXd XtWXm_private = MatrixXd::Zero(P(), P());
  #pragma omp for nowait
    for(int i = 0; i < niter; ++i){
      XtWXm_private.noalias() += re.u_weight_(i) * X.transpose() * (X.array().colwise() * W.col(i).array()).matrix();
    }
  #pragma omp critical
    XtWXm += XtWXm_private;
  }

  //XtWXm *= (1.0 / niter);
  Eigen::LLT<MatrixXd> llt(XtWXm);
  
  gradients.head(X.cols()) = X.transpose() * resid;
  VectorXd bincr = llt.solve(gradients.head(X.cols()).matrix());
  update_beta(model.linear_predictor.parameter_vector() + bincr);
  current_ll_values.first = log_likelihood();
  current_ll_var.first = (ll_current.col(0) - ll_current.col(0).mean()).square().sum() / (ll_current.col(0).size() - 1);
}

template<>
inline void glmmr::ModelOptim<bits_hsgp>::nr_beta_gaussian(){
  int n = model.data.y.size();
  int Mspec = model.covariance.Q();
  
  double sigma2 = model.data.var_par;
  double sigma2_inv = 1.0 / sigma2;
  double sigma4_inv = sigma2_inv * sigma2_inv;
  
  MatrixXd ZPhi = model.covariance.ZPhi();   // n × Mspec
  MatrixXd X = model.linear_predictor.X();
  VectorXd y = model.data.y.matrix();
  
  ArrayXd inv_lambda = 1.0 / model.covariance.LambdaSPD();
  
  // Mspec-dimensional quantities
  MatrixXd G = ZPhi.transpose() * ZPhi;        // Mspec × Mspec
  MatrixXd C = ZPhi.transpose() * X;            // Mspec × p
  VectorXd c = ZPhi.transpose() * y;            // Mspec × 1
  
  // M = diag(1/Lambda) + sigma^{-2} G
  MatrixXd Mmat = sigma2_inv * G;
  Mmat.diagonal() += inv_lambda.matrix();
  LLT<MatrixXd> llt_M(Mmat);
  
  MatrixXd M_inv_C = llt_M.solve(C);
  VectorXd M_inv_c = llt_M.solve(c);
  
  MatrixXd XtX = X.transpose() * X;
  VectorXd Xty = X.transpose() * y;
  
  // Woodbury: X^T V^{-1} X and X^T V^{-1} y
  MatrixXd XtVinvX = sigma2_inv * XtX - sigma4_inv * C.transpose() * M_inv_C;
  VectorXd XtVinvy = sigma2_inv * Xty - sigma4_inv * C.transpose() * M_inv_c;
  
  LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  VectorXd beta_new = llt_XtVinvX.solve(XtVinvy);
  
  model.linear_predictor.update_parameters(beta_new);
}

template<>
inline void glmmr::ModelOptim<bits_hsgp>::nr_theta_gaussian(){
  int n = model.data.y.size();
  int p = model.linear_predictor.P();
  int Mspec = model.covariance.Q();
  int npars = model.covariance.npar();
  int n_psi = npars + 1;                          // covariance pars + sigma^2
  
  VectorXd score(n_psi);
  MatrixXd fisher(n_psi, n_psi);
  score.setZero();
  fisher.setZero();
  
  double sigma2 = model.data.var_par;
  double sigma2_inv = 1.0 / sigma2;
  double sigma4_inv = sigma2_inv * sigma2_inv;
  double sigma6_inv = sigma4_inv * sigma2_inv;
  double sigma8_inv = sigma4_inv * sigma4_inv;
  
  // ── HSGP-specific: ZPhi, Lambda, and diagonal derivatives ──
  MatrixXd ZPhi = model.covariance.ZPhi();         // n × Mspec
  ArrayXd  Lambda = model.covariance.LambdaSPD();
  ArrayXd  inv_lambda = 1.0 / Lambda;
  
  // dLambda(k,j) = dLambda_k / d theta_j
  ArrayXXd dLambda(Mspec, npars);
  for(int k = 0; k < Mspec; k++){
    dblvec deriv = model.covariance.d_spd_nD(k);
    for(int j = 0; j < npars; j++){
      dLambda(k, j) = deriv[j];
    }
  }
  
  MatrixXd X = model.linear_predictor.X();
  VectorXd r = model.data.y.matrix() - model.linear_predictor.xb();
  
  // ── Mspec-dimensional precomputes ──
  MatrixXd G = ZPhi.transpose() * ZPhi;
  MatrixXd C = ZPhi.transpose() * X;
  VectorXd c_vec = ZPhi.transpose() * r;
  MatrixXd XtX = X.transpose() * X;
  VectorXd Xtr = X.transpose() * r;
  
  // M = diag(1/Lambda) + sigma^{-2} G
  MatrixXd Mmat = sigma2_inv * G;
  Mmat.diagonal() += inv_lambda.matrix();
  LLT<MatrixXd> llt_M(Mmat);
  MatrixXd M_inv = llt_M.solve(MatrixXd::Identity(Mspec, Mspec));
  
  MatrixXd M_inv_G = M_inv * G;
  MatrixXd M_inv_C = M_inv * C;
  VectorXd M_inv_c = M_inv * c_vec;
  MatrixXd M_inv_G_M_inv = M_inv_G * M_inv;
  
  // ── Woodbury products ──
  MatrixXd ZtVinvZ = sigma2_inv * G - sigma4_inv * G * M_inv_G;
  MatrixXd ZtVinvX = sigma2_inv * C - sigma4_inv * G * M_inv_C;
  VectorXd ZtVinvr = sigma2_inv * c_vec - sigma4_inv * G * M_inv_c;
  
  MatrixXd XtVinvX = sigma2_inv * XtX - sigma4_inv * C.transpose() * M_inv_C;
  LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  MatrixXd XtVinvX_inv = llt_XtVinvX.solve(MatrixXd::Identity(p, p));
  
  VectorXd XtVinvr = sigma2_inv * Xtr - sigma4_inv * C.transpose() * M_inv_c;
  VectorXd Vinvr   = sigma2_inv * r   - sigma4_inv * ZPhi * M_inv_c;
  
  double trM_inv_G = (M_inv.array() * G.transpose().array()).sum();
  double trVinv = n * sigma2_inv - sigma4_inv * trM_inv_G;
  
  // ── V^{-2} products ──
  MatrixXd G_M_inv_G = G * M_inv_G;
  MatrixXd G_M_inv_G_M_inv_G = G_M_inv_G * M_inv_G;
  
  MatrixXd ZtV2Z = sigma4_inv * G
  - 2.0 * sigma6_inv * G_M_inv_G
  + sigma8_inv * G_M_inv_G_M_inv_G;
  
  MatrixXd XtV2X = sigma4_inv * XtX
  - 2.0 * sigma6_inv * C.transpose() * M_inv_C
  + sigma8_inv * C.transpose() * M_inv_G_M_inv * C;
  
  MatrixXd ZtV2X = sigma4_inv * C
  - 2.0 * sigma6_inv * G * M_inv_C
  + sigma8_inv * G_M_inv_G * M_inv_C;
  
  double trM_inv_G_M_inv_G = (M_inv_G.array() * M_inv_G.transpose().array()).sum();
  double trV2 = n * sigma4_inv
  - 2.0 * sigma6_inv * trM_inv_G
  + sigma8_inv * trM_inv_G_M_inv_G;
  
  // ── P-related quantities (REML or ML) ──
  MatrixXd ZtPZ, ZtP2Z;
  VectorXd ZtPr;
  double trP, trP2, Pr_sqnorm;
  
  if(control.reml){
    MatrixXd XtVinvX_inv_ZtVinvX_t = llt_XtVinvX.solve(ZtVinvX.transpose());
    ZtPZ = ZtVinvZ - ZtVinvX * XtVinvX_inv_ZtVinvX_t;
    
    VectorXd XtVinvX_inv_XtVinvr = llt_XtVinvX.solve(XtVinvr);
    ZtPr = ZtVinvr - ZtVinvX * XtVinvX_inv_XtVinvr;
    
    VectorXd VinvX_adj = sigma2_inv * X * XtVinvX_inv_XtVinvr
    - sigma4_inv * ZPhi * (M_inv * (C * XtVinvX_inv_XtVinvr));
    VectorXd Pr = Vinvr - VinvX_adj;
    Pr_sqnorm = Pr.squaredNorm();
    
    trP = trVinv - (XtVinvX_inv.array() * XtV2X.transpose().array()).sum();
    
    double term1 = trV2;
    double term2 = 2.0 * (XtV2X.array() * XtVinvX_inv.transpose().array()).sum();
    MatrixXd W_inv_XtV2X = XtVinvX_inv * XtV2X;
    double term3 = (W_inv_XtV2X.array() * W_inv_XtV2X.transpose().array()).sum();
    trP2 = term1 - term2 + term3;
    
    MatrixXd ZtV2X_W_inv = ZtV2X * XtVinvX_inv;
    MatrixXd term2_mat = ZtV2X_W_inv * ZtVinvX.transpose();
    MatrixXd ZtVinvX_W_inv = ZtVinvX * XtVinvX_inv;
    MatrixXd term3_mat = ZtVinvX_W_inv * XtV2X * XtVinvX_inv * ZtVinvX.transpose();
    ZtP2Z = ZtV2Z - term2_mat - term2_mat.transpose() + term3_mat;
  } else {
    ZtPZ = ZtVinvZ;
    ZtPr = ZtVinvr;
    Pr_sqnorm = Vinvr.squaredNorm();
    trP = trVinv;
    trP2 = trV2;
    ZtP2Z = ZtV2Z;
  }
  
  // ── Score for theta: exploit dD_j = diag(dLambda_j) ──
  // tr(P dV_j) = tr(ZtPZ diag(dLam_j)) = diag(ZtPZ)^T dLam_j
  // r^T P dV_j P r = (ZtPr .^2)^T dLam_j
  ArrayXd ZtPZ_diag = ZtPZ.diagonal().array();
  ArrayXd ZtPr_sq   = ZtPr.array().square();
  
  for(int j = 0; j < npars; j++){
    double trace_PdV = (ZtPZ_diag * dLambda.col(j)).sum();
    double quadform  = (ZtPr_sq   * dLambda.col(j)).sum();
    score(j) = -0.5 * trace_PdV + 0.5 * quadform;
  }
  
  // ── Fisher theta-theta ──
  // tr(ZtPZ dD_j ZtPZ dD_l) = dLam_l^T (ZtPZ.^2) dLam_j
  ArrayXXd ZtPZ_sq = ZtPZ.array().square();         // Mspec × Mspec
  for(int j = 0; j < npars; j++){
    // ZtPZ_sq * dLam_j  →  Mspec-vector
    ArrayXd col_j = (ZtPZ_sq.matrix() * dLambda.col(j).matrix()).array();
    for(int l = j; l < npars; l++){
      double h = 0.5 * (col_j * dLambda.col(l)).sum();
      fisher(j, l) = h;
      if(j != l) fisher(l, j) = h;
    }
  }
  
  // ── Score and Fisher for sigma^2 (phi = log sigma^2) ──
  score(npars) = -0.5 * sigma2 * trP + 0.5 * sigma2 * Pr_sqnorm;
  fisher(npars, npars) = 0.5 * sigma2 * sigma2 * trP2;
  
  // ── Cross-Fisher theta × sigma^2 ──
  // tr(ZtP2Z diag(dLam_j)) = diag(ZtP2Z)^T dLam_j
  ArrayXd ZtP2Z_diag = ZtP2Z.diagonal().array();
  for(int j = 0; j < npars; j++){
    double cross = 0.5 * sigma2 * (ZtP2Z_diag * dLambda.col(j)).sum();
    fisher(j, npars) = cross;
    fisher(npars, j) = cross;
  }
  
  // ── Newton step ──
  LLT<MatrixXd> llt_fisher(fisher);
  if(llt_fisher.info() != Eigen::Success){
    fisher.diagonal().array() += 0.01 * fisher.diagonal().array().abs().maxCoeff() + 1e-6;
    llt_fisher.compute(fisher);
  }
  
  VectorXd step = llt_fisher.solve(score);
  
  double max_step = step.array().abs().maxCoeff();
  if(max_step > 1.0) step *= 1.0 / max_step;
  
  // ── Update covariance parameters ──
  bool logpars = model.covariance.all_log_re();
  VectorXd logtheta(npars);
  if(logpars){
    logtheta = Map<VectorXd>(model.covariance.parameters_.data(), npars);
  } else {
    for(int j = 0; j < npars; j++) logtheta(j) = log(model.covariance.parameters_[j]);
  }
  logtheta += step.head(npars);
  
  if(logpars){
    dblvec newpars(logtheta.data(), logtheta.data() + logtheta.size());
    model.covariance.update_parameters(newpars);
  } else {
    model.covariance.update_parameters(logtheta.array().exp());
  }
  
  if(model.covariance.infomat_theta.rows() != n_psi)
    model.covariance.infomat_theta.resize(n_psi, n_psi);
  model.covariance.infomat_theta = fisher;
  
  // ── Update sigma^2 ──
  double phi = std::log(sigma2);
  phi += step(npars);
  update_var_par(std::exp(phi));
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::nr_beta_gaussian(){
  int n = model.data.y.size();
  int q = model.covariance.Q();
  
  double sigma2 = model.data.var_par;
  double sigma2_inv = 1.0 / sigma2;
  double sigma4_inv = sigma2_inv * sigma2_inv;
  
  MatrixXd Z = model.covariance.Z();
  MatrixXd X = model.linear_predictor.X();
  VectorXd y = model.data.y.matrix();
  
  // Precompute
  MatrixXd G = Z.transpose() * Z;
  MatrixXd C = Z.transpose() * X;
  VectorXd c = Z.transpose() * y;
  
  std::vector<MatrixXd> derivs;
  model.covariance.derivatives(derivs, 1);
  MatrixXd& D = derivs[0];
  
  LLT<MatrixXd> llt_D(D);
  MatrixXd D_inv = llt_D.solve(MatrixXd::Identity(q, q));
  MatrixXd M = D_inv + sigma2_inv * G;
  LLT<MatrixXd> llt_M(M);
  MatrixXd M_inv = llt_M.solve(MatrixXd::Identity(q, q));
  
  // X^T V^{-1} X = sigma^{-2}*X^T*X - sigma^{-4}*C^T*M^{-1}*C
  MatrixXd XtX = X.transpose() * X;
  MatrixXd M_inv_C = M_inv * C;
  MatrixXd XtVinvX = sigma2_inv * XtX - sigma4_inv * C.transpose() * M_inv_C;
  
  // X^T V^{-1} y = sigma^{-2}*X^T*y - sigma^{-4}*C^T*M^{-1}*c
  VectorXd Xty = X.transpose() * y;
  VectorXd M_inv_c = M_inv * c;
  VectorXd XtVinvy = sigma2_inv * Xty - sigma4_inv * C.transpose() * M_inv_c;
  
  // beta = (X^T V^{-1} X)^{-1} X^T V^{-1} y
  LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  VectorXd beta_new = llt_XtVinvX.solve(XtVinvy);
  
  model.linear_predictor.update_parameters(beta_new);
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::nr_theta_gaussian(){
  int n = model.data.y.size();
  int p = model.linear_predictor.P();
  int n_theta = model.covariance.parameters_.size();
  int n_psi = n_theta + 1;
  int q = model.covariance.Q();
  
  VectorXd score(n_psi);
  MatrixXd fisher(n_psi, n_psi);
  score.setZero();
  fisher.setZero();
  
  double sigma2 = model.data.var_par;
  double sigma2_inv = 1.0 / sigma2;
  double sigma4_inv = sigma2_inv * sigma2_inv;
  double sigma6_inv = sigma4_inv * sigma2_inv;
  double sigma8_inv = sigma4_inv * sigma4_inv;
  
  // Get D and derivatives (all q x q)
  std::vector<MatrixXd> derivs;
  model.covariance.derivatives(derivs, 1);
  MatrixXd& D = derivs[0];
  
  MatrixXd Z = model.covariance.Z();
  MatrixXd X = model.linear_predictor.X();
  VectorXd r = model.data.y.matrix() - model.linear_predictor.xb();
  
  // Precompute q-dimensional quantities
  MatrixXd G = Z.transpose() * Z;                    // q x q
  MatrixXd C = Z.transpose() * X;                    // q x p
  VectorXd c = Z.transpose() * r;                    // q x 1
  MatrixXd XtX = X.transpose() * X;                  // p x p
  VectorXd Xtr = X.transpose() * r;                  // p x 1
  
  // M = D^{-1} + sigma^{-2} * G  (q x q)
  LLT<MatrixXd> llt_D(D);
  MatrixXd D_inv = llt_D.solve(MatrixXd::Identity(q, q));
  MatrixXd M = D_inv + sigma2_inv * G;
  LLT<MatrixXd> llt_M(M);
  MatrixXd M_inv = llt_M.solve(MatrixXd::Identity(q, q));
  
  // Precompute products we'll need multiple times
  MatrixXd M_inv_G = M_inv * G;                      // q x q
  MatrixXd M_inv_C = M_inv * C;                      // q x p
  VectorXd M_inv_c = M_inv * c;                      // q x 1
  MatrixXd M_inv_G_M_inv = M_inv * G * M_inv;        // q x q
  
  // V^{-1} via Woodbury: V^{-1} = sigma^{-2}*I - sigma^{-4}*Z*M^{-1}*Z^T
  // We never form this explicitly
  
  // Z^T V^{-1} Z = sigma^{-2}*G - sigma^{-4}*G*M^{-1}*G
  MatrixXd ZtVinvZ = sigma2_inv * G - sigma4_inv * G * M_inv_G;
  
  // Z^T V^{-1} X = sigma^{-2}*C - sigma^{-4}*G*M^{-1}*C
  MatrixXd ZtVinvX = sigma2_inv * C - sigma4_inv * G * M_inv_C;
  
  // Z^T V^{-1} r = sigma^{-2}*c - sigma^{-4}*G*M^{-1}*c
  VectorXd ZtVinvr = sigma2_inv * c - sigma4_inv * G * M_inv_c;
  
  // X^T V^{-1} X = sigma^{-2}*X^T*X - sigma^{-4}*C^T*M^{-1}*C
  MatrixXd XtVinvX = sigma2_inv * XtX - sigma4_inv * C.transpose() * M_inv_C;
  LLT<MatrixXd> llt_XtVinvX(XtVinvX);
  MatrixXd XtVinvX_inv = llt_XtVinvX.solve(MatrixXd::Identity(p, p));
  
  // X^T V^{-1} r = sigma^{-2}*X^T*r - sigma^{-4}*C^T*M^{-1}*c
  VectorXd XtVinvr = sigma2_inv * Xtr - sigma4_inv * C.transpose() * M_inv_c;
  
  // V^{-1} r (as n-vector, but compute via q-space)
  VectorXd Vinvr = sigma2_inv * r - sigma4_inv * Z * M_inv_c;
  
  // tr(V^{-1}) = n*sigma^{-2} - sigma^{-4}*tr(M^{-1}*G)
  double trM_inv_G = (M_inv.array() * G.transpose().array()).sum();
  double trVinv = n * sigma2_inv - sigma4_inv * trM_inv_G;
  
  // Z^T V^{-2} Z = sigma^{-4}*G - 2*sigma^{-6}*G*M^{-1}*G + sigma^{-8}*G*M^{-1}*G*M^{-1}*G
  MatrixXd G_M_inv_G = G * M_inv_G;
  MatrixXd G_M_inv_G_M_inv_G = G_M_inv_G * M_inv_G;
  MatrixXd ZtV2Z = sigma4_inv * G 
  - 2.0 * sigma6_inv * G_M_inv_G
  + sigma8_inv * G_M_inv_G_M_inv_G;
  
  // X^T V^{-2} X = sigma^{-4}*X^T*X - 2*sigma^{-6}*C^T*M^{-1}*C + sigma^{-8}*C^T*M^{-1}*G*M^{-1}*C
  MatrixXd XtV2X = sigma4_inv * XtX 
  - 2.0 * sigma6_inv * C.transpose() * M_inv_C
  + sigma8_inv * C.transpose() * M_inv_G_M_inv * C;
  
  // Z^T V^{-2} X = sigma^{-4}*C - 2*sigma^{-6}*G*M^{-1}*C + sigma^{-8}*G*M^{-1}*G*M^{-1}*C
  MatrixXd ZtV2X = sigma4_inv * C
  - 2.0 * sigma6_inv * G * M_inv_C
  + sigma8_inv * G_M_inv_G * M_inv_C;
  
  // tr(V^{-2}) = n*sigma^{-4} - 2*sigma^{-6}*tr(M^{-1}*G) + sigma^{-8}*tr(G*M^{-1}*G*M^{-1})
  double trM_inv_G_M_inv_G = (M_inv_G.array() * M_inv_G.transpose().array()).sum();
  double trV2 = n * sigma4_inv 
  - 2.0 * sigma6_inv * trM_inv_G
  + sigma8_inv * trM_inv_G_M_inv_G;
  
  // Now compute P-related quantities depending on REML or ML
  MatrixXd ZtPZ;
  VectorXd ZtPr;
  double trP;
  double trP2;
  MatrixXd ZtP2Z;
  double Pr_sqnorm;
  
  if(control.reml){
    // P = V^{-1} - V^{-1}*X*(X^T*V^{-1}*X)^{-1}*X^T*V^{-1}
    
    // Z^T P Z = Z^T V^{-1} Z - Z^T V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1} Z
    MatrixXd XtVinvX_inv_ZtVinvX_t = llt_XtVinvX.solve(ZtVinvX.transpose());
    ZtPZ = ZtVinvZ - ZtVinvX * XtVinvX_inv_ZtVinvX_t;
    
    // Z^T P r = Z^T V^{-1} r - Z^T V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1} r
    VectorXd XtVinvX_inv_XtVinvr = llt_XtVinvX.solve(XtVinvr);
    ZtPr = ZtVinvr - ZtVinvX * XtVinvX_inv_XtVinvr;
    
    // Pr = V^{-1}r - V^{-1}X*(X^TV^{-1}X)^{-1}*X^TV^{-1}r
    VectorXd VinvX_adj = sigma2_inv * X * XtVinvX_inv_XtVinvr 
    - sigma4_inv * Z * (M_inv * (C * XtVinvX_inv_XtVinvr));
    VectorXd Pr = Vinvr - VinvX_adj;
    Pr_sqnorm = Pr.squaredNorm();
    
    // tr(P) = tr(V^{-1}) - tr((X^T V^{-1} X)^{-1} X^T V^{-2} X)
    trP = trVinv - (XtVinvX_inv.array() * XtV2X.transpose().array()).sum();
    
    // tr(P^2) = tr(V^{-2}) - 2*tr(V^{-2}X W^{-1} X^T V^{-1}) + tr(V^{-1}X W^{-1} X^T V^{-2} X W^{-1} X^T V^{-1})
    // where W = X^T V^{-1} X
    
    // Term 1: tr(V^{-2})
    double term1 = trV2;
    
    // Term 2: 2*tr(V^{-2}X W^{-1} X^T V^{-1}) = 2*tr(W^{-1} X^T V^{-1} V^{-2} X)
    //       = 2*tr(W^{-1} X^T V^{-3} X)... complicated
    // Simpler: = 2*tr((X^T V^{-2} X) W^{-1})
    double term2 = 2.0 * (XtV2X.array() * XtVinvX_inv.transpose().array()).sum();
    
    // Term 3: tr(V^{-1}X W^{-1} X^T V^{-2} X W^{-1} X^T V^{-1})
    //       = tr(W^{-1} X^T V^{-2} X W^{-1} X^T V^{-2} X)
    MatrixXd W_inv_XtV2X = XtVinvX_inv * XtV2X;
    double term3 = (W_inv_XtV2X.array() * W_inv_XtV2X.transpose().array()).sum();
    
    trP2 = term1 - term2 + term3;
    
    // Z^T P^2 Z = Z^T V^{-2} Z - 2*Z^T V^{-2} X W^{-1} X^T V^{-1} Z 
    //           + Z^T V^{-1} X W^{-1} X^T V^{-2} X W^{-1} X^T V^{-1} Z
    
    // Term 1: Z^T V^{-2} Z
    // Term 2: 2 * Z^T V^{-2} X W^{-1} X^T V^{-1} Z
    MatrixXd ZtV2X_W_inv = ZtV2X * XtVinvX_inv;
    MatrixXd term2_mat = ZtV2X_W_inv * ZtVinvX.transpose();
    
    // Term 3: Z^T V^{-1} X W^{-1} X^T V^{-2} X W^{-1} X^T V^{-1} Z
    MatrixXd ZtVinvX_W_inv = ZtVinvX * XtVinvX_inv;
    MatrixXd term3_mat = ZtVinvX_W_inv * XtV2X * XtVinvX_inv * ZtVinvX.transpose();
    
    ZtP2Z = ZtV2Z - term2_mat - term2_mat.transpose() + term3_mat;
    
  } else {
    // ML: P = V^{-1}
    ZtPZ = ZtVinvZ;
    ZtPr = ZtVinvr;
    Pr_sqnorm = Vinvr.squaredNorm();
    trP = trVinv;
    trP2 = trV2;
    ZtP2Z = ZtV2Z;
  }
  
  // Score and Fisher for theta (random effects parameters)
  std::vector<MatrixXd> ZtPZ_dD(n_theta);
  
  for(int j = 0; j < n_theta; j++){
    MatrixXd& dD_j = derivs[j + 1];
    
    // tr(P * dV_j) = tr(ZtPZ * dD_j)
    double trace_PdV = (ZtPZ.array() * dD_j.transpose().array()).sum();
    
    // r^T P dV_j P r = ZtPr^T * dD_j * ZtPr
    double quadform = ZtPr.dot(dD_j * ZtPr);
    
    score(j) = -0.5 * trace_PdV + 0.5 * quadform;
    
    ZtPZ_dD[j] = ZtPZ * dD_j;
  }
  
  // Fisher theta-theta
  for(int j = 0; j < n_theta; j++){
    for(int k = j; k < n_theta; k++){
      fisher(j, k) = 0.5 * (ZtPZ_dD[j].array() * ZtPZ_dD[k].transpose().array()).sum();
      fisher(k, j) = fisher(j, k);
    }
  }
  
  // Score for sigma2 (phi = log(sigma2))
  score(n_theta) = -0.5 * sigma2 * trP + 0.5 * sigma2 * Pr_sqnorm;
  
  // Fisher sigma2-sigma2
  fisher(n_theta, n_theta) = 0.5 * sigma2 * sigma2 * trP2;
  
  // Fisher theta-sigma2
  for(int j = 0; j < n_theta; j++){
    MatrixXd& dD_j = derivs[j + 1];
    fisher(j, n_theta) = 0.5 * sigma2 * (ZtP2Z.array() * dD_j.transpose().array()).sum();
    fisher(n_theta, j) = fisher(j, n_theta);
  }
  
  // Newton step
  LLT<MatrixXd> llt_fisher(fisher);
  if(llt_fisher.info() != Eigen::Success){
    // Fisher not positive definite - add small ridge
    fisher.diagonal().array() += 0.01 * fisher.diagonal().array().abs().maxCoeff() + 1e-6;
    llt_fisher.compute(fisher);
  }
  
  VectorXd step = llt_fisher.solve(score);
  
  // Damping if step is too large
  double max_step = step.array().abs().maxCoeff();
  if(max_step > 1.0){
    step *= 1.0 / max_step;
  }
  
  // Update parameters
  VectorXd theta_current = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(model.covariance.parameters_.data(),model.covariance.parameters_.size());
  VectorXd theta_new = theta_current + step.head(n_theta);
  model.covariance.update_parameters(theta_new);
  
  if(model.covariance.infomat_theta.rows() != fisher.rows()) model.covariance.infomat_theta.resize(fisher.rows(), fisher.cols());
  model.covariance.infomat_theta = fisher;
  
  double phi = std::log(sigma2);
  // Update sigma2
  phi += step(n_theta);
  update_var_par(std::exp(phi));
}

template<typename modeltype>
inline bool glmmr::ModelOptim<modeltype>::check_convergence(const double tol, const int hist, const int k, const int k0){
  gradient_history.push_back(current_ll_values.first + current_ll_values.second);
  if(gradient_history.size() > hist) gradient_history.pop_front();
  double diffg = 1;
  double vardiffg = 0.1;
  double z = 10;
  int iter = 0;
  if(gradient_history.size()>1){
    diffg = (gradient_history.back() - gradient_history.front())/(gradient_history.size() - 1);
    if(gradient_history.size()>2){
      vardiffg = 0;
      double a = 0;
      for(const double x: gradient_history){
        if(iter > 0) {
          vardiffg += pow((x - a) - diffg, 2);
        } 
        a = x;
        iter++;
      }
      vardiffg *= 1.0/ (gradient_history.size() - 2);
      z = diffg * sqrt(gradient_history.size()) / sqrt(vardiffg);
      converge_z.push_back(z);
    }
  }
  if(trace > 0)Rcpp::Rcout << "\nMean: " << diffg << " sd: " << sqrt(vardiffg) << "\nZ (diff): " << z;
  double prior0 = 1.0 - exp(-(k*k/(k0*k0)));// squared weibull
  double p = maths::gaussian_cdf(z);
  double bf = (1-p)*prior0/(p*(1-prior0));
  if(gradient_history.size()>2)converge_bf.push_back(bf);
  if(trace > 0)Rcpp::Rcout << "\nBF: " << bf << " prior: " << prior0;
  double meang = 0;
  for(const double x: gradient_history) meang += x;
  meang *= 1.0/gradient_history.size();
  double diff = quantile == 0 ? 10 : meang - quantile;
  
  if(trace > 0)Rcpp::Rcout << "\nGradients: " << gradients.transpose() << " | ||G|| = " << gradients.matrix().norm();
  if(trace > 0)Rcpp::Rcout << "\nLog-likelihood running mean: " << meang << " (old " << quantile << ") diff: " << diff;
  quantile = meang;
  return bf > tol;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::nr_theta(){
  if(re.scaled_u_.cols() != re.u_.cols())re.scaled_u_.resize(NoChange,re.u_.cols());
  if(ll_current.rows() != re.u_.cols())ll_current.resize(re.u_.cols(),NoChange);
  previous_ll_values.second = current_ll_values.second;
  previous_ll_var.second = current_ll_var.second;
  ArrayXd tmp(ll_current.rows());
  model.covariance.nr_step(re.scaled_u_, re.u_solve_, tmp, gradients, re.u_weight_);
  bool weighted = re.u_weight_.maxCoeff() - re.u_weight_.minCoeff() > 1e-10;
  re.update_zu(weighted);
  ll_current.col(1) = tmp;
  current_ll_values.second = ll_current.col(1).mean();
  current_ll_var.second = (ll_current.col(1) - ll_current.col(1).mean()).square().sum() / (ll_current.col(1).size() - 1);
  calculate_var_par();
}

template<>
inline void glmmr::ModelOptim<bits_hsgp>::nr_theta(){
  if(re.scaled_u_.cols() != re.u_.cols()) re.scaled_u_.resize(NoChange, re.u_.cols());
  previous_ll_values.second = current_ll_values.second;
  previous_ll_var.second = current_ll_var.second;
  
  const int n_iter = re.u_.cols();
  const int M = model.covariance.Q();
  const int npars = model.covariance.npar();
  ArrayXd eta(model.n());
  VectorXd W_(eta.size());
  ArrayXd resid(eta.size());
  
  MatrixXd Phi = model.covariance.PhiSPD(false, false);
  
  ArrayXd Lambda = model.covariance.LambdaSPD();
  ArrayXd inv_lambda = 1.0 / Lambda;
  ArrayXd inv_lambda2 = inv_lambda.square();
  ArrayXd sqrt_lambda = Lambda.sqrt();
  
  ArrayXXd lambda_deriv(M, npars);
  ArrayXXd dsqrt_lambda(M, npars);
  for(int i = 0; i < M; i++){
    dblvec deriv = model.covariance.d_spd_nD(i);
    for(int j = 0; j < npars; j++){
      lambda_deriv(i, j) = deriv[j];
      dsqrt_lambda(i, j) = deriv[j] / (2.0 * sqrt_lambda(i));
    }
  }
  
  // Precompute Phi * diag(d sqrt(Lambda)/d theta_j) for observation-space Hessian
  // std::vector<MatrixXd> Phi_dj(npars);
  // for(int j = 0; j < npars; j++){
  //   Phi_dj[j] = Phi * dsqrt_lambda.col(j).matrix().asDiagonal();
  // }
  
  std::vector<MatrixXd> Phi_dj(npars);
  for(int j = 0; j < npars; j++){
    ArrayXd scale = 0.5 * lambda_deriv.col(j) * inv_lambda;
    Phi_dj[j] = Phi * scale.matrix().asDiagonal();
  }
  
  // --- Prior gradient: deterministic + MC ---
  VectorXd grad = VectorXd::Zero(npars);
  for(int j = 0; j < npars; j++){
    grad(j) = -0.5 * (lambda_deriv.col(j) * inv_lambda).sum();
  }
  if(n_iter == 1){
    ArrayXd second_moment = re.u_.col(0).array().square() + re.u_var_diag_.array();
    for(int j = 0; j < npars; j++){
      grad(j) += 0.5 * (second_moment * lambda_deriv.col(j) * inv_lambda2).sum();
    }
  } else {
    for(int i = 0; i < n_iter; i++){
      double w_i = re.u_weight_(i);
      // u_ is in u-space, so quadratic form uses u^2/Lambda^2
      ArrayXd u2 = re.u_.col(i).array().square();
      for(int j = 0; j < npars; j++){
        grad(j) += 0.5 * w_i * (u2 * lambda_deriv.col(j) * inv_lambda2).sum();
      }
    }
  }
  
  // --- Combined Hessian: prior Fisher + observation-space Fisher ---
  // Prior Fisher: ½ sum_k (dLambda_k/dtheta_j)(dLambda_k/dtheta_l) / Lambda_k^2
  MatrixXd Hess = MatrixXd::Zero(npars, npars);
  for(int j = 0; j < npars; j++){
    for(int l = j; l < npars; l++){
      double h = 0.5 * (lambda_deriv.col(j) * lambda_deriv.col(l) * inv_lambda2).sum();
      Hess(j, l) = h;
      if(j != l) Hess(l, j) = h;
    }
  }
  
  ArrayXd W_avg = ArrayXd::Zero(model.n());
  
  // Observation-space Fisher: sum_i w_i (d eta/d theta_j)^T W (d eta/d theta_l)
  MatrixXd zd = matrix.linpred();
  for(int i = 0; i < n_iter; i++){
    eta = zd.col(i).array();
    
    switch(model.family.family){
    case Fam::gaussian:
      if(model.family.link == Link::identity){
        W_ = (model.data.variance.inverse() * model.data.weights).matrix();
      } else {
        throw std::runtime_error("NR2 only available with canonical link");
      }
      break;
    case Fam::binomial: case Fam::bernoulli:
      if(model.family.link == Link::logit){
        ArrayXd logitp = (eta.exp().inverse() + 1.0).inverse();
        W_ = (model.data.variance * logitp * (1 - logitp)).matrix();
      } else {
        throw std::runtime_error("NR2 only available with canonical link");
      }
      break;
    case Fam::poisson:
      if(model.family.link == Link::loglink){
        W_ = eta.exp().matrix();
      } else {
        throw std::runtime_error("NR2 only available with canonical link");
      }
      break;
    default:
      throw std::runtime_error("NR2 only available with Gaussian, Poisson, and Binomial");
    }
    
    double w_i = re.u_weight_(i);
    ArrayXd w_arr = W_.array();
    W_avg += w_i * W_.array();
    
    for(int j = 0; j < npars; j++){
      ArrayXd djprod = (Phi_dj[j] * re.u_.col(i)).array();
      for(int l = j; l < npars; l++){
        ArrayXd dlprod = (Phi_dj[l] * re.u_.col(i)).array();
        double h_obs = (djprod * dlprod * w_arr).sum();
        Hess(j, l) += w_i * h_obs;
        if(j != l) Hess(l, j) += w_i * h_obs;
      }
    }
  }
  
  
  // --- REML correction to gradient ---
  if(control.reml){
    int P = model.linear_predictor.P();
    MatrixXd X = model.linear_predictor.X();  // n × P
    MatrixXd PhiW = Phi.transpose() * W_avg.matrix().asDiagonal();   // M × n
    MatrixXd C_mat = inv_lambda.matrix().asDiagonal().toDenseMatrix();
    C_mat.noalias() += PhiW * Phi;
    MatrixXd WX = W_avg.matrix().asDiagonal() * X;
    MatrixXd PhiWX = Phi.transpose() * WX;
    Eigen::LLT<MatrixXd> llt_C(C_mat);
    if(llt_C.info() == Eigen::Success){
      MatrixXd C_inv_PhiWX = llt_C.solve(PhiWX);           // M × P
      MatrixXd R = inv_lambda.matrix().asDiagonal() * C_inv_PhiWX;
      MatrixXd XtVinvX = X.transpose() * WX;
      XtVinvX.noalias() -= PhiWX.transpose() * C_inv_PhiWX;
      
      Eigen::LLT<MatrixXd> llt_XVX(XtVinvX);
      if(llt_XVX.info() == Eigen::Success){
        MatrixXd S = llt_XVX.solve(MatrixXd::Identity(P, P));  // P × P
        MatrixXd T = R * S;                                     // M × P
        ArrayXd TR_diag = (T.array() * R.array()).rowwise().sum();  // M
        
        for(int j = 0; j < npars; j++){
          grad(j) += 0.5 * (lambda_deriv.col(j) * TR_diag).sum();
        }
      }
    }
  }
  
  // --- Solve with fallback ---
  model.covariance.infomat_theta = Hess;
  
  bool logpars = model.covariance.all_log_re();
  VectorXd logtheta(npars);
  if(logpars){
    logtheta = Map<VectorXd>(model.covariance.parameters_.data(), model.covariance.parameters_.size());
  } else {
    for(int j = 0; j < npars; j++) logtheta(j) = log(model.covariance.parameters_[j]);
  }
  
  VectorXd step;
  double lambda_damp = 1e-4 * Hess.diagonal().array().abs().maxCoeff();
  MatrixXd Hess_reg = Hess;
  Hess_reg.diagonal().array() += lambda_damp;
  Eigen::LLT<MatrixXd> llt_H(Hess_reg);
  if(llt_H.info() == Eigen::Success){
    step = llt_H.solve(grad);
  } else {
    step = grad.array() / Hess.diagonal().array().abs().max(1e-6);
  }
  
  double max_step = 1.0;
  double step_norm = step.array().abs().maxCoeff();
  if(n_iter > 1 && step_norm > max_step) step *= max_step / step_norm;
  
  logtheta += step;
  
  if(logpars){
    dblvec newpars(logtheta.data(), logtheta.data() + logtheta.size());
    model.covariance.update_parameters(newpars);
  } else {
    model.covariance.update_parameters(logtheta.array().exp());
  }
  gradients.tail(grad.size()) = grad.array();
  bool weighted = re.u_weight_.maxCoeff() - re.u_weight_.minCoeff() > 1e-10;
  re.update_zu(weighted);
  current_ll_values.second = log_likelihood(false);
  current_ll_var.second = (ll_current.col(1) - ll_current.col(1).mean()).square().sum() / (ll_current.col(1).size() - 1);
  calculate_var_par();
}

template<>
inline void glmmr::ModelOptim<bits_spde>::nr_theta(){
  if(re.scaled_u_.cols() != re.u_.cols()) re.scaled_u_.resize(NoChange, re.u_.cols());
  previous_ll_values.second = current_ll_values.second;
  previous_ll_var.second    = current_ll_var.second;
  
  auto& cov = model.covariance;
  if(!cov.spde_loaded){
    throw std::runtime_error("SPDE nr_theta: spde_data() has not been loaded.");
  }
  if(!cov.chol_Q_current) cov.refactor_Q();
  const int n_iter = re.u_.cols();
  const int M      = cov.Q();           // n_v
  const int npars  = cov.npar();        // 2 (σ², λ)
  auto [tr_grad, tr_hess] = cov.traces_for_lambda(50);
  VectorXd trace_term(npars);
  trace_term(0) = -static_cast<double>(M);
  trace_term(1) = tr_grad;//cov.trace_Qinv_dQ_log_lambda_hutch();   // triggers Takahashi if stale
  MatrixXd scores(npars, n_iter);       // column k = score vector for sample k
  VectorXd grad = VectorXd::Zero(npars);
  
  // MC path — weighted average over posterior samples.
  for(int k = 0; k < n_iter; ++k){
    const VectorXd& u_k = re.u_.col(k);
    const double w_k    = re.u_weight_(k);
    
    double qf_Q = cov.quad_form_Q(u_k);
    double qf_l = cov.quad_form_dQ_log_lambda(u_k);
    
    scores(0, k) = -0.5 * static_cast<double>(M) + 0.5 * qf_Q;
    scores(1, k) =  0.5 * trace_term(1)          - 0.5 * qf_l;
    
    grad += w_k * scores.col(k);
  }
  MatrixXd Hess(npars, npars);
  Hess(0, 0) = 0.5 * static_cast<double>(M);
  Hess(0, 1) = -0.5 * trace_term(1);
  Hess(1, 0) =  Hess(0, 1);
  Hess(1, 1) = 0.5 * tr_hess;//cov.trace_Qinv_dQ_Qinv_dQ_lambda();
  cov.infomat_theta = Hess;
  
  // --- NR step on log-scale parameters, with same damping/clamping as HSGP ---
  bool logpars = cov.all_log_re();
  VectorXd logtheta(npars);
  if(logpars){
    logtheta = Map<VectorXd>(cov.parameters_.data(), cov.parameters_.size());
  } else {
    for(int j = 0; j < npars; ++j) logtheta(j) = std::log(cov.parameters_[j]);
  }
  
  double lambda_damp = 1e-4 * Hess.diagonal().array().abs().maxCoeff();
  MatrixXd Hess_reg  = Hess;
  Hess_reg.diagonal().array() += lambda_damp;
  
  VectorXd step;
  Eigen::LLT<MatrixXd> llt_H(Hess_reg);
  if(llt_H.info() == Eigen::Success){
    step = llt_H.solve(grad);
  } else {
    step = grad.array() / Hess.diagonal().array().abs().max(1e-6);
  }
  
  const double max_step  = 1.0;
  const double step_norm = step.array().abs().maxCoeff();
  if(n_iter > 1 && step_norm > max_step) step *= max_step / step_norm;
  
  logtheta += step;
  
  if(logpars){
    dblvec newpars(logtheta.data(), logtheta.data() + logtheta.size());
    cov.update_parameters(newpars);
  } else {
    cov.update_parameters(logtheta.array().exp());
  }
  
  gradients.tail(grad.size()) = grad.array();
  bool weighted = re.u_weight_.maxCoeff() - re.u_weight_.minCoeff() > 1e-10;
  re.update_zu(weighted);
  current_ll_values.second = log_likelihood(false);
  current_ll_var.second    = (ll_current.col(1) - ll_current.col(1).mean())
                .square().sum() / (ll_current.col(1).size() - 1);
  
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_var_par(const double& v){
  model.data.var_par = v;
  model.data.variance.setConstant(v);
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::update_var_par(const ArrayXd& v){
  model.data.variance = v;
}

template<typename modeltype>
inline void glmmr::ModelOptim<modeltype>::calculate_var_par(){
  if(model.family.family==Fam::gaussian || model.family.family==Fam::quantile_scaled){
    // revise this for beta and Gamma re residuals
    if(control.reml){
      // using Diffey 2017
      // generate_czz();
      VectorXd resid = (model.data.y - re.zu_.col(0));
      MatrixXd X = model.linear_predictor.X();
      MatrixXd XW = X;
      bool weighted = (model.data.weights != 1).any();
      if (weighted) XW.applyOnTheLeft(model.data.weights.matrix().asDiagonal());
      MatrixXd XWX = X.transpose() * XW;
      XWX = XWX.llt().solve(MatrixXd::Identity(XWX.rows(),XWX.cols()));
      MatrixXd U = -1.0* XW * XWX * XW.transpose();
      if(weighted){
        U = U + model.data.weights.matrix().asDiagonal().toDenseMatrix();
      } else {
        U += MatrixXd::Identity(U.rows(),U.cols());
      }
      MatrixXd ZUZC = model.covariance.Z().transpose() * U * model.covariance.Z();
      ZUZC *= CZZ;
      double new_var_par = (1.0 /( model.n() - X.cols())) * (resid.transpose() * U * resid + ZUZC.trace());
      update_var_par(new_var_par);
    } else {
      int niter = re.u(false).cols();
      ArrayXd sigmas(niter);
      sigmas.setZero();
      MatrixXd zd = matrix.linpred();
      MatrixXd zdu = glmmr::maths::mod_inv_func(zd, model.family.link);
//#pragma omp parallel for if(niter > 50)
      for(int i = 0; i < niter; ++i){
        ArrayXd resid = (model.data.y - zdu.col(i));
        resid *= model.data.weights.sqrt();
        sigmas(i) = (resid - resid.mean()).square().sum()/(resid.size()-1.0);
      }
      update_var_par(sigmas.mean());
    }
  }
}

template<typename modeltype>
inline double glmmr::ModelOptim<modeltype>::aic(){
  MatrixXd Lu = re.u();
  int dof = P() + model.covariance.npar();
  double logl = 0;
#pragma omp parallel for reduction (+:logl) if(Lu.cols() > 50)
  for(int i = 0; i < Lu.cols(); i++){
    logl += model.covariance.log_likelihood(Lu.col(i));
  }
  double ll = log_likelihood();
  
  return (-2*( ll + logl ) + 2*dof); 
}

template<typename modeltype>
inline ArrayXd glmmr::ModelOptim<modeltype>::optimum_weights(double N, 
                                                             VectorXd C,
                                                             double tol,
                                                             int max_iter){
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
  if(C.size()!=P())throw std::runtime_error("C is wrong size");
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
    //sparse ZLs = submat_sparse(model.covariance.ZL_sparse(),sb.RowIndexes);
    ArrayXi rows = Map<ArrayXi,Unaligned>(sb.RowIndexes.data(),sb.RowIndexes.size());
    MatrixXd ZLs = model.covariance.ZL();
    MatrixXd ZL = glmmr::Eigen_ext::submat(ZLs,rows,ArrayXi::LinSpaced(Q(),0,Q()-1));//sparse_to_dense(ZLs,false);
    MatrixXd S = ZL * ZL.transpose();
    ZDZ.push_back(S);
    Sigmas.push_back(S);
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
      for(int i = 0 ; i < SB.size(); i++){
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
    for(int i = 0 ; i < static_cast<int>(SB.size()); i++){
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
          for(int k = 0; k < Xs.size(); k++){
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
      for(int k = 0; k < static_cast<int>(SB.size()); k++){
        M += Xs[k].transpose() * Sigmas[k] * Xs[k];
      }
    }
    M = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
    VectorXd Mc = M*Cvec;
    weightsnew.setZero();
    for(int i = 0 ; i < SB.size(); i++){
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
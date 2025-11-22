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
    bool    trisect_once = false; 
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
  virtual void    nr_beta();
  virtual void    nr_theta();
  virtual void    update_var_par(const double& v);
  virtual void    update_var_par(const ArrayXd& v);
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            ml_beta();
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            ml_theta();
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            ml_all();
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
  }
  op.minimise();
  int eval_size = control.saem ? re.mcmc_block_size : ll_current.rows();
  current_ll_values.second = ll_current.col(1).tail(eval_size).mean();
  current_ll_var.second = (ll_current.col(1).tail(eval_size) - ll_current.col(1).tail(eval_size).mean()).square().sum() / (eval_size - 1);
  calculate_var_par();
}

template<typename modeltype>
template<class algo, typename>
inline void glmmr::ModelOptim<modeltype>::ml_all(){
  if(model.covariance.parameters_.size()==0)throw std::runtime_error("no covariance parameters, cannot calculate log likelihood");
  dblvec start = get_start_values(true,true,false);  
  dblvec lower = get_lower_values(true,true,false);
  dblvec upper = get_upper_values(true,true,false);
  // store previous log likelihood values for convergence calculations
  previous_ll_values.second = current_ll_values.second;
  previous_ll_var.second = current_ll_var.second;
  previous_ll_values.first = previous_ll_values.second;
  previous_ll_var.first = previous_ll_var.second;
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
    op.template fn<&glmmr::ModelOptim<bits>::log_likelihood_all, glmmr::ModelOptim<bits> >(this);
  } else if constexpr (std::is_same_v<modeltype,bits_nngp>) {
    op.template fn<&glmmr::ModelOptim<bits_nngp>::log_likelihood_all, glmmr::ModelOptim<bits_nngp> >(this);
  } else if constexpr (std::is_same_v<modeltype,bits_hsgp>){
    op.template fn<&glmmr::ModelOptim<bits_hsgp>::log_likelihood_all, glmmr::ModelOptim<bits_hsgp> >(this);
  }
  op.minimise();
  int eval_size = control.saem ? re.mcmc_block_size : ll_current.rows();
  current_ll_values.second = ll_current.col(1).tail(eval_size).mean();
  current_ll_var.second = (ll_current.col(1).tail(eval_size) - ll_current.col(1).tail(eval_size).mean()).square().sum() / (eval_size - 1);
  current_ll_values.first = current_ll_values.second;
  current_ll_var.first = current_ll_var.second;
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
    ll = ll_current.col(1).mean();
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
//#pragma omp parallel for if(re.zu_.cols() > 50)
      for(int j= 0; j< re.zu_.cols() ; j++){
        ll_current(j,llcol ) = glmmr::maths::log_likelihood(model.data.y.array(),xb + re.zu_.col(j).array(),
                   model.data.variance * model.data.weights.inverse(),
                   model.family);
        
        // for(int i = 0; i<model.n(); i++){
        //   ll_current(j,llcol ) += glmmr::maths::log_likelihood(model.data.y(i),xb(i) + re.zu_(i,j),
        //                                      model.data.variance(i)/model.data.weights(i),
        //                                      model.family);
        // }
      }
    } else {
//#pragma omp parallel for if(re.zu_.cols() > 50)
      for(int j=0; j< re.zu_.cols() ; j++){
        for(int i = 0; i<model.n(); i++){
          ll_current(j,llcol) += model.data.weights(i)*glmmr::maths::log_likelihood(model.data.y(i),xb(i) + re.zu_(i,j),
                                   model.data.variance(i),model.family);
        }
      }
      ll_current.col(llcol) *= model.data.weights.sum()/model.n();
    }
  } else {
//#pragma omp parallel for if(re.zu_.cols() > 50)
    for(int j= 0; j< re.zu_.cols() ; j++){
      ll_current(j,llcol) = glmmr::maths::log_likelihood(model.data.y.array(),xb + re.zu_.col(j).array(),
                 model.data.variance,model.family);
      // for(int i = 0; i<model.n(); i++){
      //   ll_current(j,llcol) += glmmr::maths::log_likelihood(model.data.y(i),xb(i) + re.zu_(i,j),
      //                                      model.data.variance(i),model.family);
      // }
    }
  }
  return ll_current.col(llcol).mean();
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
  matrix.W.update();
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
  
  int niter = re.u(false).cols();
  MatrixXd zd = matrix.linpred();
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

  MatrixXd XtWXm = MatrixXd::Zero(P(), P());
  MatrixXd W = glmmr::maths::dhdmu(zd, model.family);
  W = (W.array().colwise() * nvar_par).inverse();
  W.array().colwise() *= model.data.weights;

  zd = maths::mod_inv_func(zd, model.family.link);
  if(model.family.family == Fam::binomial) zd.array().colwise() *= model.data.variance;
  VectorXd resid(model.n());
  
  if(!model.family.canonical()){
    MatrixXd zdresid = MatrixXd::Zero(model.n(), zd.cols());
    zdresid.colwise() += model.data.y;
    zdresid -= zd;
    MatrixXd detmu = maths::detadmu(zd, model.family.link);
    zdresid.array() *= detmu.array();
    resid.setZero();
    for(int i = 0; i < niter; ++i){
      resid += (W.col(i).array() * zdresid.col(i).array()).matrix();
    }
    resid *= (1.0 / niter);
  } else {
    resid = model.data.y - zd.rowwise().mean();
  }

  #pragma omp parallel
  {
    MatrixXd XtWXm_private = MatrixXd::Zero(P(), P());
  #pragma omp for nowait
    for(int i = 0; i < niter; ++i){
      XtWXm_private.noalias() += X.transpose() * (X.array().colwise() * W.col(i).array()).matrix();
    }
  #pragma omp critical
    XtWXm += XtWXm_private;
  }

  XtWXm *= (1.0 / niter);
  Eigen::LLT<MatrixXd> llt(XtWXm);
  
  gradients.head(X.cols()) = X.transpose() * resid;
  VectorXd bincr = llt.solve(gradients.head(X.cols()).matrix());
  update_beta(model.linear_predictor.parameter_vector() + bincr);
  current_ll_values.first = log_likelihood();
  current_ll_var.first = (ll_current.col(0) - ll_current.col(0).mean()).square().sum() / (ll_current.col(0).size() - 1);
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
  //Rcpp::Rcout << "\nmeang: " << meang << " varg " << varg << " sdg: " << sdg << " cv: " << sdg/meang;
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
  ArrayXd  tmp(ll_current.rows());
  
  model.covariance.nr_step(re.scaled_u_, tmp, gradients);
  re.update_zu();
  ll_current.col(1) = tmp;
  
  current_ll_values.second = ll_current.col(1).mean();
  current_ll_var.second = (ll_current.col(1) - ll_current.col(1).mean()).square().sum() / (ll_current.col(1).size() - 1);
  calculate_var_par();
}

template<>
inline void glmmr::ModelOptim<bits_hsgp>::nr_theta(){
  if(control.reml)throw std::runtime_error("Newton-Raphson not compatible with REML for covariance parameters");
  if(re.scaled_u_.cols() != re.u_.cols())re.scaled_u_.resize(NoChange,re.u_.cols());
  previous_ll_values.second = current_ll_values.second;
  previous_ll_var.second = current_ll_var.second;
  
  // Pre-cache frequently accessed members
  const int n_iter = re.u_.cols();
  const int resid_size = model.data.y.size();
  const double inv_n_iter = 1.0 / static_cast<double>(n_iter);
  ArrayXd eta(model.n());
  VectorXd W_(eta.size());
  ArrayXd resid(eta.size());
  
  MatrixXd Phi = model.covariance.PhiSPD(false, false);
  VectorXd grad = VectorXd::Zero(2);
  MatrixXd hess = MatrixXd::Zero(2, 2);
  
  // Pre-compute lambda derivatives
  ArrayXXd lambda_deriv(model.covariance.Q(), 2);
  for(int i = 0; i < model.covariance.Q(); i++){
    dblvec deriv = model.covariance.d_spd_nD(i);
    for(int j = 0; j < 2; j++) lambda_deriv(i, j) = deriv[j];
  }
  
  MatrixXd zd = matrix.linpred();
  
  MatrixXd Phi_d0 = Phi * lambda_deriv.matrix().col(0).asDiagonal();
  MatrixXd Phi_d1 = Phi * lambda_deriv.matrix().col(1).asDiagonal();
  
  for(int i = 0; i < n_iter; i++){
    eta = zd.col(i).array();
    switch(model.family.family){
    case Fam::gaussian: 
      if(model.family.link == Link::identity){
        W_ = (model.data.variance.inverse() *  model.data.weights).matrix();
        resid = model.data.y.array() - eta;
      } else {
        throw std::runtime_error("NR2 only available with canonical link");
      }
      break;
    case Fam::binomial: case Fam::bernoulli:
      if(model.family.link == Link::logit){
        ArrayXd logitp = (eta.exp().inverse() + 1.0).inverse();
        resid = model.data.y.array() - model.data.variance * logitp;
        W_ = (model.data.variance * logitp * (1- logitp)).matrix();
        
      } else {
        throw std::runtime_error("NR2 posterior only available with canonical link");
      }
      break;
    case Fam::poisson:
      if(model.family.link == Link::loglink){
        resid = model.data.y.array() - eta.exp();
        W_ = eta.exp().matrix();
      } else {
        throw std::runtime_error("NR2 posterior only available with canonical link");
      }
      break;
    default:
      throw std::runtime_error("NR2 posterior only available with Gaussian, Poisson, and Binomial");
    break;
    }
    
    // ArrayXd resid = (model.data.y.matrix() - zdu).array();
    ArrayXd w_arr = W_.array();
    
    // Vectorized products
    ArrayXd d0prod = (Phi_d0 * re.u_.col(i)).array();
    ArrayXd d1prod = (Phi_d1 * re.u_.col(i)).array();
    // ArrayXd d2prod = (Phi_d200 * re.u_.col(i)).array();
    // ArrayXd d3prod = (Phi_d211 * re.u_.col(i)).array();
    // ArrayXd d4prod = (Phi_d201 * re.u_.col(i)).array();
    
    // Vectorized accumulation - replaces entire inner loop!
    ArrayXd resid_w = resid * w_arr;
    
    grad(0) += (d0prod * resid_w).sum();
    grad(1) += (d1prod * resid_w).sum();
    
    hess(0, 0) += (d0prod * d0prod * w_arr).sum();// - d2prod * resid_w).sum();
    hess(1, 1) += (d1prod * d1prod * w_arr).sum();//  - d4prod * resid_w).sum();
    
    double addv = (d0prod * d1prod * w_arr).sum();//  - d3prod * resid_w).sum();
    hess(0, 1) += addv;
    hess(1, 0) += addv;
  }
  
  // Rcpp::Rcout << "\nHess: \n" << hess;
  // Rcpp::Rcout << "\nGrad: " << grad.transpose();
  
  // Apply scaling
  hess *= inv_n_iter;
  grad *= inv_n_iter;
  
  hess = hess.llt().solve(MatrixXd::Identity(2, 2));
  
  VectorXd logpars(2);
  logpars(0) = log(model.covariance.parameters_[0]);
  logpars(1) = log(model.covariance.parameters_[1]);
  // Rcpp::Rcout << "\nOld pars: " << logpars.transpose();
  logpars += hess * grad;
  
  //Rcpp::Rcout << "\nNew pars: " << logpars.transpose();
  model.covariance.update_parameters(logpars.array().exp());
  current_ll_values.second = log_likelihood(false);
  current_ll_var.second = (ll_current.col(1) - ll_current.col(1).mean()).square().sum() / (ll_current.col(1).size() - 1);
  calculate_var_par();
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
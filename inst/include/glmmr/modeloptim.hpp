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
  
  // constructor
  ModelOptim(modeltype& model_, glmmr::ModelMatrix<modeltype>& matrix_,glmmr::RandomEffects<modeltype>& re_) ;

  // control parameters for the optimisers - direct will be removed as its useless.
  struct OptimControl {
    int     npt = 0;
    double  rhobeg = 0;
    double  rhoend = 0;
    bool    direct = false;
    double  direct_range_beta = 3.0;
    int     max_iter_direct = 100; 
    double  epsilon = 1e-4; 
    bool    select_one = true; 
    bool    trisect_once = false; 
    int     max_eval = 0; 
    bool    mrdirect = false;
    double  g_epsilon = 1e-8;
    int     past = 3;
    double  delta = 1e-8;
    int     max_linesearch = 64;
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
  virtual void    laplace_nr_beta_u();
  virtual void    update_var_par(const double& v);
  virtual void    update_var_par(const ArrayXd& v);
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            ml_beta();
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            ml_theta();
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            ml_all();
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            laplace_ml_beta_u();
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            laplace_ml_theta();
  template<class algo, typename = std::enable_if_t<std::is_base_of<optim_algo, algo>::value> >
  void            laplace_ml_beta_theta();
  virtual double  aic();
  virtual ArrayXd optimum_weights(double N, VectorXd C, double tol = 1e-5, int max_iter = 501);
  void            set_bobyqa_control(int npt_, double rhobeg_, double rhoend_);
  void            set_direct_control(bool direct = false, double direct_range_beta = 3.0, int max_iter = 100, double epsilon = 1e-4, bool select_one = true, bool trisect_once = false, 
                                      int max_eval = 0, bool mrdirect = false);
  void            set_lbfgs_control(double g_epsilon = 1e-8, int past = 3, double delta = 1e-8, int max_linesearch = 64);
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
  double          log_likelihood_beta_with_gradient(const VectorXd &beta, VectorXd& g);
  double          log_likelihood_theta(const dblvec &theta);
  double          log_likelihood_theta_with_gradient(const VectorXd& theta, VectorXd& g);
  double          log_likelihood_all(const dblvec &par);
  double          log_likelihood_laplace_beta_u(const dblvec &par);
  double          log_likelihood_laplace_beta_u_with_gradient(const VectorXd& theta, VectorXd& g);
  double          log_likelihood_laplace_theta(const dblvec &par);
  double          log_likelihood_laplace_beta_theta(const dblvec &par);
  
protected:
// objects
  dblvec    lower_bound;
  dblvec    upper_bound; // bounds for beta
  dblvec    lower_bound_theta;
  dblvec    upper_bound_theta; // bounds for beta
  bool      beta_bounded = false;
  double    quantile = 0.5;
  
  // functions
  void            calculate_var_par();
  dblvec          get_start_values(bool beta, bool theta, bool var = true);
  dblvec          get_lower_values(bool beta, bool theta, bool var = true, bool u = false);
  dblvec          get_upper_values(bool beta, bool theta, bool var = true, bool u = false);
  void            set_direct_control(directd& op);
  void            set_bobyqa_control(bobyqad& op);
  void            set_newuoa_control(newuoad& op);
  void            set_lbfgs_control(lbfgsd& op);
  
private:
  // used for REML
  void        generate_czz();
  MatrixXd    CZZ = MatrixXd::Zero(1,1);
};

}



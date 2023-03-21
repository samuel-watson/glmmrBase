#ifndef MODEL_HPP
#define MODEL_HPP

#define _USE_MATH_DEFINES

#include <boost/math/special_functions/digamma.hpp>
#include <rbobyqa.h>
#include "general.h"
#include "maths.h"
#include "openmpheader.h"
#include "covariance.hpp"
#include "linearpredictor.hpp"
#include "sparse.h"
#include <random>


// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

namespace glmmr {

using namespace rminqa;
using namespace Eigen;

class Model {
public:
  glmmr::Formula formula_;
  glmmr::Covariance covariance_;
  glmmr::LinearPredictor linpred_;
  double var_par_;
  std::string family_; 
  std::string link_;
  VectorXd offset_;
  const VectorXd y_;
  int n_;
  int Q_;
  int P_;
  int flink;
  bool attenuate_;
  MatrixXd Z_;
  
  Model(
    const VectorXd &y,
    const std::string& formula,
    const ArrayXXd& data,
    const strvec& colnames,
    std::string family, 
    std::string link
  ) : formula_(formula),
      covariance_(formula_,data,colnames),
      linpred_(formula_,data,colnames),
      var_par_(1.0),
      family_(family),
      link_(link),
      offset_(VectorXd::Zero(data.rows())),
      y_(y),
      n_(data.rows()),
      Q_(covariance_.Q()),
      P_(linpred_.P()),
      flink(glmmr::maths::model_to_int.at(family_+link_)),
      attenuate_(false),
      size_m_array(1),
      size_q_array(ArrayXd::Zero(Q_)),
      size_n_array(ArrayXd::Zero(n_)),
      size_p_array(ArrayXd::Zero(P_)),
      ZL_(n_,Q_),
      u_(MatrixXd::Zero(Q_,1)),
      zu_(MatrixXd::Zero(n_,1)),
      W_(n_),
      trace_(0),
      u0_(Q_),
      up_(Q_),
      r_(Q_),
      grad_(Q_){
        for(int i = 0; i< P_; i++){
          lower_b_.push_back(R_NegInf);
          upper_b_.push_back(R_PosInf);
        }
        for(int i = 0; i< Q_; i++){
          lower_t_.push_back(1e-6);
          upper_t_.push_back(R_PosInf);
        }
      };
  
  void set_offset(const VectorXd& offset);
  
  void update_beta(const VectorXd &beta);
  
  void update_beta(const dblvec &beta);
  
  void update_beta_extern(const dblvec &beta);
  
  void update_theta(const VectorXd &theta);
  
  void update_theta(const dblvec &theta);
  
  void update_theta_extern(const dblvec &theta);

  void update_u(const MatrixXd &u);
  
  void update_W(int i = 0);
  
  double log_prob(const VectorXd &v);
  
  VectorXd log_gradient(const VectorXd &v,
                              bool beta = false);
 
  MatrixXd linpred(){
    return (zu_.colwise()+(linpred_.xb()+offset_));
  }
  
  VectorXd xb(){
    return linpred_.xb()+offset_;
  }
  
  double log_likelihood();
  
  void ml_theta();
  
  void ml_beta();
  
  void ml_all();
  
  void laplace_ml_beta_u();
  
  void laplace_ml_theta();
  
  void laplace_ml_beta_theta();
  
  void nr_beta();
  
  void laplace_nr_beta_u();
  
  MatrixXd laplace_hessian(double tol = 1e-4);
  
  MatrixXd hessian(double tol = 1e-4);
  
  MatrixXd u(){
    return covariance_.Lu(u_);
  }
  
  MatrixXd Zu(){
    return zu_;
  }
  
  void mcmc_sample(int warmup,
                   int samples,
                   int adapt = 100){
    sample(warmup,samples,adapt);
    if(u_.cols()!=zu_.cols())zu_.resize(Q_,u_.cols());
    zu_ = ZL_*u_;
  }
  
  void mcmc_set_lambda(double lambda){
    lambda_ = lambda;
  }
  
  void mcmc_set_max_steps(int max_steps){
    max_steps_ = max_steps;
  }
  
  void mcmc_set_refresh(int refresh){
    refresh_ = refresh;
  }
  
  void mcmc_set_target_accept(double target){
    target_accept_ = target;
  }
  
  void set_trace(int trace){
    trace_ = trace;
  }
  
  double aic();
  
  void make_covariance_sparse(){
    covariance_.set_sparse(true);
  }
  
  void make_covariance_dense(){
    covariance_.set_sparse(false);
  }
  
private:
  ArrayXd size_m_array;
  ArrayXd size_q_array;
  ArrayXd size_n_array;
  ArrayXd size_p_array;
  sparse ZL_;
  MatrixXd u_;
  MatrixXd zu_;
  VectorXd W_;
  std::vector<double> lower_b_;
  std::vector<double> upper_b_;
  std::vector<double> lower_t_;
  std::vector<double> upper_t_;
  int trace_;
  VectorXd u0_;
  VectorXd up_;
  VectorXd r_;
  VectorXd grad_;
  int refresh_=500;
  double lambda_=0.01;
  int max_steps_ = 100;
  int accept_;
  double e_ = 0.001;
  double ebar_ = 1.0;
  double H_ = 0;
  int steps_;
  double target_accept_ = 0.9;
  bool verbose_ = true;
  
  dblvec get_start_values(bool beta, bool theta, bool var = true);
  
  dblvec get_lower_values(bool beta, bool theta, bool var = true);
  
  dblvec get_upper_values(bool beta, bool theta, bool var = true);
  
  VectorXd new_proposal(const VectorXd& u0_, bool adapt, 
                        int iter, double rand);
  
  void sample(int warmup,
                  int nsamp,
                  int adapt = 100);
  
  class D_likelihood : public Functor<dblvec> {
    Model& M_;
    const MatrixXd& Lu_;
    double logl;
  public:
    D_likelihood(Model& M,
                 const MatrixXd& Lu) :
    M_(M),
    Lu_(Lu),
    logl(0.0) {};
    double operator()(const dblvec &par);
  };
  
  class L_likelihood : public Functor<dblvec> {
    Model& M_;
    double ll;
  public:
    L_likelihood(Model& M) :  
    M_(M), ll(0.0) {};
    double operator()(const dblvec &par);
  };
  
  class F_likelihood : public Functor<dblvec> {
    Model& M_;
    int G;
    bool importance_;
    double ll;
    double logl;
    double denomD;
  public:
    F_likelihood(Model& M, bool importance = false) : 
    M_(M),
    G(M_.covariance_.npar()), 
    importance_(importance), 
    ll(0.0), logl(0.0),
    denomD(0.0) {
      for(int i = 0; i < M_.u_.cols(); i++){
        denomD += M_.covariance_.log_likelihood(M_.u_.col(i));
      }
      denomD *= 1/M_.u_.cols();
    }
    double operator()(const dblvec &par);
  };
  
  
  class LA_likelihood : public Functor<dblvec> {
    Model& M_;
    MatrixXd v;
    MatrixXd LZWZL;
    double LZWdet;
    double logl;
    double ll;
  public:
    LA_likelihood(Model& M) :
    M_(M),
    v(M_.Q_,1),
    LZWZL(MatrixXd::Zero(M_.Q_,M_.Q_)),
    LZWdet(0.0),
    logl(0.0),ll(0.0){
      M_.update_W();
      LZWZL = M_.covariance_.LZWZL(M_.W_);
      LZWdet = glmmr::maths::logdet(LZWZL);
    };
    double operator()(const dblvec &par);
  };
  
  class LA_likelihood_cov : public Functor<dblvec> {
    Model& M_;
    MatrixXd LZWZL;
    double LZWdet;
    double logl;
    double ll;
  public:
    LA_likelihood_cov(Model& M) :
    M_(M),
    LZWZL(MatrixXd::Zero(M_.Q_,M_.Q_)),
    LZWdet(0.0), logl(0.0), ll(0.0) {} 
    
    double operator()(const dblvec &par);
  };
  
  class LA_likelihood_btheta : public Functor<dblvec> {
    Model& M_;
    MatrixXd LZWZL;
    double LZWdet;
    double logl;
    double ll;
  public:
    LA_likelihood_btheta(Model& M) :
    M_(M),
    LZWZL(MatrixXd::Zero(M_.Q_,M_.Q_)),
    LZWdet(0.0), logl(0.0), ll(0.0) {} 
    
    double operator()(const dblvec &par);
  };
  
};

}


#include "likelihood.ipp"
#include "mhmcmc.ipp"
#include "model.ipp"

#endif
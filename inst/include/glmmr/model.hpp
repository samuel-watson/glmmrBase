#ifndef MODEL_HPP
#define MODEL_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <rbobyqa.h>
#include "general.h"
#include "maths.h"
#include "openmpheader.h"
#include "covariance.hpp"
#include "linearpredictor.hpp"
#include <random>

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

namespace glmmr {

using namespace rminqa;

class Model {
public:
  glmmr::Formula formula_;
  glmmr::Covariance covariance_;
  glmmr::LinearPredictor linpred_;
  double var_par_;
  std::string family_; 
  std::string link_;
  Eigen::VectorXd offset_;
  const Eigen::VectorXd y_;
  int n_;
  int Q_;
  int P_;
  int flink;
  bool useLflag;
  bool attenuate_;
  Eigen::MatrixXd Z_;
  
  Model(
    const Eigen::VectorXd &y,
    const std::string& formula,
    const Eigen::ArrayXXd& data,
    const strvec& colnames,
    std::string family, 
    std::string link
  ) : formula_(formula),
      covariance_(formula_,data,colnames),
      linpred_(formula_,data,colnames),
      var_par_(1.0),
      family_(family),
      link_(link),
      offset_(Eigen::VectorXd::Zero(data.rows())),
      y_(y),
      n_(data.rows()),
      Q_(covariance_.Q()),
      P_(linpred_.P()),
      flink(glmmr::maths::string_to_case.at(family_+link_)),
      useLflag(false),
      attenuate_(false),
      Z_(covariance_.Z()),
      size_m_array(1),
      size_q_array(Eigen::ArrayXd::Zero(Q_)),
      size_n_array(Eigen::ArrayXd::Zero(n_)),
      size_p_array(Eigen::ArrayXd::Zero(P_)),
      ZL_(Z_.rows(),Z_.cols()),
      u_(Eigen::MatrixXd::Zero(Q_,1)),
      zu_(Z_*u_),
      L_(Q_,Q_),
      W_(n_),
      trace_(0),
      mcmc_(*this){
        for(int i = 0; i< P_; i++){
          lower_b_.push_back(R_NegInf);
          upper_b_.push_back(R_PosInf);
        }
        for(int i = 0; i< Q_; i++){
          lower_t_.push_back(1e-6);
          upper_t_.push_back(R_PosInf);
        }
      };
  
  void set_offset(const Eigen::VectorXd& offset);
  
  void update_beta(const Eigen::VectorXd &beta);
  
  void update_beta(const dblvec &beta);
  
  void update_beta_extern(const dblvec &beta);
  
  void update_theta(const Eigen::VectorXd &theta);
  
  void update_theta(const dblvec &theta);
  
  void update_theta_extern(const dblvec &theta);

  void update_u(const Eigen::MatrixXd &u);
  
  void use_L_in_calculations(bool useL = true){
    useLflag = useL;
  }
  
  void update_W(int i = 0);
  
  double log_prob(const Eigen::VectorXd &v);
  
  Eigen::VectorXd log_gradient(const Eigen::VectorXd &v,
                           bool usezl = true,
                           bool beta = false);
 
  Eigen::MatrixXd linpred(){
    return (zu_.colwise()+(linpred_.xb()+offset_));
  }
  
  Eigen::VectorXd xb(){
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
  
  Eigen::MatrixXd laplace_hessian(double tol = 1e-4);
  
  Eigen::MatrixXd hessian(double tol = 1e-4);
  
  Eigen::MatrixXd u(){
    return u_;
  }
  
  Eigen::MatrixXd L(){
    return L_;
  }
  
  Eigen::MatrixXd ZL(){
    return ZL_;
  }
  
  Eigen::MatrixXd Zu(){
    return zu_;
  }
  
  void mcmc_sample(int warmup,
                   int samples,
                   int adapt = 100){
    Eigen::MatrixXd re_samples = mcmc_.sample(warmup,samples,adapt);
    update_u(re_samples);
  }
  
  void mcmc_set_lambda(double lambda){
    mcmc_.lambda_ = lambda;
  }
  
  void mcmc_set_max_steps(int max_steps){
    mcmc_.max_steps_ = max_steps;
  }
  
  void mcmc_set_refresh(int refresh){
    mcmc_.refresh_ = refresh;
  }
  
  void set_trace(int trace){
    trace_ = trace;
  }
  
  double aic();
  
private:
  Eigen::ArrayXd size_m_array;
  Eigen::ArrayXd size_q_array;
  Eigen::ArrayXd size_n_array;
  Eigen::ArrayXd size_p_array;
  Eigen::MatrixXd ZL_;
  Eigen::MatrixXd u_;
  Eigen::MatrixXd zu_;
  Eigen::MatrixXd L_;
  Eigen::VectorXd W_;
  std::vector<double> lower_b_;
  std::vector<double> upper_b_;
  std::vector<double> lower_t_;
  std::vector<double> upper_t_;
  int trace_;
  
  
  dblvec get_start_values(bool beta, bool theta, bool var = true);
  
  dblvec get_lower_values(bool beta, bool theta, bool var = true);
  
  dblvec get_upper_values(bool beta, bool theta, bool var = true);
  
  class D_likelihood : public Functor<dblvec> {
    Model& M_;
    double logl;
  public:
    D_likelihood(Model& M) :
    M_(M),
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
    Eigen::MatrixXd v;
    Eigen::MatrixXd LZWZL;
    double LZWdet;
    double logl;
    double ll;
  public:
    LA_likelihood(Model& M) :
    M_(M),
    v(M_.Q_,1),
    LZWZL(Eigen::MatrixXd::Zero(M.ZL_.cols(),M.ZL_.cols())),
    LZWdet(0.0),
    logl(0.0),ll(0.0){
      M_.update_W();
      LZWZL = M_.ZL_.transpose() * M_.W_.asDiagonal() * M_.ZL_;
      LZWZL.noalias() += Eigen::MatrixXd::Identity(LZWZL.rows(),LZWZL.cols());
      LZWdet = glmmr::maths::logdet(LZWZL);
    };
    
    double operator()(const dblvec &par);
  };
  
  class LA_likelihood_cov : public Functor<dblvec> {
    Model& M_;
    Eigen::MatrixXd LZWZL;
    double LZWdet;
    double logl;
    double ll;
  public:
    LA_likelihood_cov(Model& M) :
    M_(M),
    LZWZL(Eigen::MatrixXd::Zero(M.ZL_.cols(),M.ZL_.cols())),
    LZWdet(0.0), logl(0.0), ll(0.0) {} 
    
    double operator()(const dblvec &par);
  };
  
  class LA_likelihood_btheta : public Functor<dblvec> {
    Model& M_;
    Eigen::MatrixXd LZWZL;
    double LZWdet;
    double logl;
    double ll;
  public:
    LA_likelihood_btheta(Model& M) :
    M_(M),
    LZWZL(Eigen::MatrixXd::Zero(M_.ZL_.cols(),M_.ZL_.cols())),
    LZWdet(0.0), logl(0.0), ll(0.0) {} 
    
    double operator()(const dblvec &par);
  };
  
  class HMC {
  public:
    Model& M_;
    Eigen::VectorXd u0_;
    Eigen::VectorXd up_;
    Eigen::VectorXd r_;
    Eigen::VectorXd grad_;
    int refresh_=500;
    double lambda_=0.01;
    int max_steps_ = 100;
    std::minstd_rand gen_;
    std::uniform_real_distribution<double> dist_;
    int accept_;
    double e_;
    double ebar_;
    int steps_;
    double H_;
    double target_accept_ = 0.9;
    bool verbose_ = true;
    
    HMC(Model& M) :
      M_(M),
      u0_(M_.Q_),
      up_(M_.Q_),
      r_(M_.Q_),
      grad_(M_.Q_)
    {
      initialise_u();
    }
    
    void initialise_u();
    
    void new_proposal(bool adapt = false, int iter = 1);
    
    Eigen::ArrayXXd sample(int warmup,
                           int nsamp,
                           int adapt = 100);
    
  };
  
  HMC mcmc_;
  
};

}



#include "likelihood.ipp"
#include "mhmcmc.ipp"
#include "model.ipp"

#endif
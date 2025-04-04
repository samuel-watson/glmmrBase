#pragma once

#include "general.h"
#include "modelbits.hpp"
#include "modelmatrix.hpp"
#include "randomeffects.hpp"
#include "openmpheader.h"
#include "maths.h"

namespace glmmr {

using namespace Eigen;

template<typename modeltype>
class ModelMCMC{
public:
  modeltype&                        model;
  glmmr::ModelMatrix<modeltype>&    matrix;
  glmmr::RandomEffects<modeltype>&  re;
  bool                              verbose = true;
  int                               trace = 1;
  
  ModelMCMC(modeltype& model_,glmmr::ModelMatrix<modeltype>& matrix_,glmmr::RandomEffects<modeltype>& re_);
  
  double    log_prob(const VectorXd &v);
  void      mcmc_sample(int warmup_,int samples_,int adapt_ = 100);
  void      mcmc_set_lambda(double lambda);
  void      mcmc_set_max_steps(int max_steps);
  void      mcmc_set_refresh(int refresh);
  void      mcmc_set_target_accept(double target);
  
protected:
  VectorXd      u0;
  VectorXd      up;
  VectorXd      r;
  VectorXd      grad;
  int           refresh=500;
  double        lambda=0.01;
  int           max_steps = 100;
  int           accept;
  double        e = 0.001;
  double        ebar = 1.0;
  double        H= 0;
  int           steps;
  double        target_accept = 0.9;
  VectorXd      new_proposal(const VectorXd& u0_, bool adapt,  int iter, double rand);
  void          sample(int warmup_,int nsamp_,int adapt_ = 100);
  
};

}

template<typename modeltype>
inline glmmr::ModelMCMC<modeltype>::ModelMCMC(modeltype& model_, 
          glmmr::ModelMatrix<modeltype>& matrix_,
          glmmr::RandomEffects<modeltype>& re_) : 
  model(model_), 
  matrix(matrix_), 
  re(re_),
  u0(model.covariance.Q()),
  up(model.covariance.Q()),
  r(model.covariance.Q()),
  grad(model.covariance.Q()){};

template<typename modeltype>
inline double glmmr::ModelMCMC<modeltype>::log_prob(const VectorXd &v){
  VectorXd zu = SparseOperators::operator*(model.covariance.ZL_sparse(), v);
  VectorXd mu = model.xb().matrix() + zu;
  double lp1 = 0;
  double lp2 = 0;
  if(model.weighted){
    if(model.family.family==Fam::gaussian){
#pragma omp parallel for reduction (+:lp1) 
      for(int i = 0; i<model.n(); i++){
        lp1 += glmmr::maths::log_likelihood(model.data.y(i),mu(i),model.data.variance(i)/model.data.weights(i),
                                            model.family);
      }
    } else {
#pragma omp parallel for reduction (+:lp1) 
      for(int i = 0; i<model.n(); i++){
        lp1 += model.data.weights(i)*glmmr::maths::log_likelihood(model.data.y(i),mu(i),model.data.variance(i),
                                  model.family);
      }
      lp1 *= model.data.weights.sum()/model.n();
    }
  } else {
#pragma omp parallel for reduction (+:lp1)
    for(int i = 0; i<model.n(); i++){
      lp1 += glmmr::maths::log_likelihood(model.data.y(i),mu(i),model.data.variance(i),
                                          model.family);
    }
  }
#pragma omp parallel for reduction (+:lp2)
  for(int i = 0; i < v.size(); i++){
    lp2 += -0.5*v(i)*v(i); 
  }
  return lp1+lp2-0.5*v.size()*log(2*M_PI);
}

template<typename modeltype>
inline VectorXd glmmr::ModelMCMC<modeltype>::new_proposal(const VectorXd& u0_,
                                              bool adapt_, 
                                              int iter_,
                                              double runif_){
  
    VectorXd r(model.covariance.Q()); 
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution d{ 0.0, 1.0 };
    auto random_norm = [&d, &gen] { return d(gen); };
    for (int j = 0; j < r.size(); j++) r(j) = random_norm();
  
  VectorXd grad = matrix.log_gradient(u0_,false);
  double lpr = 0.5*r.transpose()*r;
  VectorXd up = u0_;
  steps = std::max(1,(int)std::round(lambda/e));
  steps = std::min(steps, max_steps);
  // leapfrog integrator
  for(int i=0; i< steps; i++){
    r += (e/2)*grad;
    up += e * r;
    grad = matrix.log_gradient(up,false);
    r += (e/2)*grad;
  }
  double lprt = 0.5*r.transpose()*r;
  double l1 = log_prob(u0_);
  double l2 = log_prob(up);
  double prob = std::min(1.0,exp(-l1 + lpr + l2 - lprt));
  bool accepttf = runif_ < prob;
#ifdef R_BUILD
  if(trace==2){
    int printSize = u0_.size() < 10 ? u0_.size() : 10;
    Rcpp::Rcout << "\nIter: " << iter_ << " l1 " << l1 << " h1 " << lpr << " l2 " << l2 << " h2 " << lprt;
    Rcpp::Rcout << "\nCurrent value: " << u0_.transpose().head(printSize);
    Rcpp::Rcout << "\nvelocity: " << r.transpose().head(printSize);
    Rcpp::Rcout << "\nProposal: " << up.transpose().head(printSize);
    Rcpp::Rcout << "\nAccept prob: " << prob << " step size: " << e << " mean: " << ebar << " steps: " << steps;
    if(accepttf){
      Rcpp::Rcout << " ACCEPT \n";
    } else {
      Rcpp::Rcout << " REJECT \n";
    }
  }
#endif
  if(adapt_){
    double f1 = 1.0/(iter_ + 10);
    H = (1-f1)*H + f1*(target_accept - prob);
    double loge = -4.60517 - (sqrt((double)iter_ / 0.05))*H;
    double powm = std::pow(iter_,-0.75);
    double logbare = powm*loge + (1-powm)*log(ebar);
    e = exp(loge);
    ebar = exp(logbare);
  } else {
    e = ebar;
  }
  if(accepttf){
    accept++;
    return up;
  } else {
    return u0_;
  }
}

template<typename modeltype>
inline void glmmr::ModelMCMC<modeltype>::sample(int warmup_,
                                 int nsamp_,
                                 int adapt_){
    VectorXd unew(model.covariance.Q());
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution d{ 0.0, 1.0 };
    auto random_norm = [&d, &gen] { return d(gen); };
    for (int j = 0; j < unew.size(); j++) unew(j) = random_norm();
  accept = 0;
  std::minstd_rand gen2(std::random_device{}());
  std::uniform_real_distribution<double> dist2(0.0, 1.0);
  int totalsamps = nsamp_ + warmup_;
  int i;
  double prob;
  prob = dist2(gen2);
  // warmups
  for(i = 0; i < warmup_; i++){
    prob = dist2(gen2);
    if(i < adapt_){
      unew = new_proposal(unew,true,i+1,prob);
    } else {
      unew = new_proposal(unew,false,i+1,prob);
    }
    if(verbose && i%refresh== 0){
      //Rcpp::Rcout << "\nWarmup: Iter " << i << " of " << totalsamps;
    }
  }
  re.u_.col(0) = unew;
  //sampling
  for(i = 0; i < nsamp_-1; i++){
    prob = dist2(gen2);
    re.u_.col(i+1) = new_proposal(re.u_.col(i),false,i+1,prob);
#ifdef R_BUILD
    if(verbose && i%refresh== 0){
      Rcpp::Rcout << "\nSampling: Iter " << i + warmup_ << " of " << totalsamps;
    }
#endif
  }
#ifdef R_BUILD
  if(trace>0)Rcpp::Rcout << "\nAccept rate: " << (double)accept/(warmup_+nsamp_) << " steps: " << steps << " step size: " << e;
  if(verbose)Rcpp::Rcout << "\n" << std::string(40, '-');
#endif
}

template<typename modeltype>
inline void glmmr::ModelMCMC<modeltype>::mcmc_sample(int warmup_,
                                          int samples_,
                                          int adapt_){
  if(re.u_.cols()!=samples_)re.u_.resize(NoChange,samples_);
  if(re.zu_.cols()!=samples_)re.zu_.resize(NoChange,samples_);
  re.u_.setZero();
  sample(warmup_,samples_,adapt_);
  re.zu_ = model.covariance.ZLu(re.u_);
}

template<typename modeltype>
inline void glmmr::ModelMCMC<modeltype>::mcmc_set_lambda(double lambda_){
  lambda = lambda_;
}

template<typename modeltype>
inline void glmmr::ModelMCMC<modeltype>::mcmc_set_max_steps(int max_steps_){
  max_steps = max_steps_;
}

template<typename modeltype>
inline void glmmr::ModelMCMC<modeltype>::mcmc_set_refresh(int refresh_){
  refresh = refresh_;
}

template<typename modeltype>
inline void glmmr::ModelMCMC<modeltype>::mcmc_set_target_accept(double target_){
  target_accept = target_;
}

#pragma once

//#include <boost/math/distributions/normal.hpp>
// #include <boost/math/distributions/students_t.hpp>
#include <random>
#include "general.h"
#include "algo.h"
#include "family.hpp"

// [[Rcpp::depends(RcppEigen)]]

struct VectorMatrix {
public:
  VectorXd vec;
  MatrixXd mat;
  VectorMatrix(int n): vec(n), mat(n,n) {};
  VectorMatrix(const VectorMatrix& x) : vec(x.vec), mat(x.mat) {};
  VectorMatrix& operator=(VectorMatrix x){
    vec = x.vec;
    mat = x.mat;
    return *this;
  };
};

struct MatrixMatrix {
public:
  MatrixXd mat1;
  MatrixXd mat2;
  double a = 0;
  double b = 0;
  MatrixMatrix(int n1, int m1, int n2, int m2): mat1(n1,m1), mat2(n2,m2) {};
  MatrixMatrix(const MatrixMatrix& x) : mat1(x.mat1), mat2(x.mat2) {};
  MatrixMatrix& operator=(MatrixMatrix x){
    mat1 = x.mat1;
    mat2 = x.mat2;
    a = x.a;
    b = x.b;
    return *this;
  };
};

namespace glmmr {
/*
 template<class T>
 inline T randomGaussian(T generator,
 VectorXd& res)
 {
 for (size_t i = 0; i < res.size(); ++i)
 res(i) = generator();
 // Note the generator is returned back
 return  generator;
 }*/

namespace maths {

inline double gaussian_cdf(double value)
{
  return 0.5 * erfc(-value * 0.707106781186547524401);
}


inline Eigen::VectorXd gaussian_cdf_vec(const Eigen::VectorXd& v) {
  Eigen::VectorXd res(v.size());
  for (int i = 0; i < v.size(); ++i)
    res(i) = gaussian_cdf(v(i));
  return res;
}

inline Eigen::MatrixXd gaussian_cdf_vec(const Eigen::MatrixXd& v) {
  Eigen::MatrixXd res(v.rows(), v.cols());
  for (int i = 0; i < v.rows(); ++i) {
    for (int j = 0; j < v.cols(); j++) {
      res(i,j) = gaussian_cdf(v(i));
    }
  }
  return res;
}

inline double gaussian_pdf(double x)
{
  static const double inv_sqrt_2pi = 0.3989422804014327;
  return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

inline Eigen::VectorXd gaussian_pdf_vec(const Eigen::VectorXd& v)
{
  Eigen::VectorXd res(v.size());
  for (int i = 0; i < v.size(); ++i)
    res(i) = gaussian_pdf(v(i));
  return res;
}

inline Eigen::MatrixXd gaussian_pdf_vec(const Eigen::MatrixXd& v)
{
  Eigen::MatrixXd res(v.rows(), v.cols());
  for (int i = 0; i < v.rows(); ++i) {
    for (int j = 0; j < v.cols(); j++) {
      res(i, j) = gaussian_pdf(v(i));
    }
  }
  return res;
}

inline Eigen::MatrixXd mod_inv_func(const Eigen::MatrixXd& muin,
                                    Link link)
{
  Eigen::MatrixXd mu(muin);
  switch (link) {
  case Link::logit:
    mu = (mu.array().exp() / (1.0 + mu.array().exp())).matrix();
    break;
  case Link::loglink:
    mu = mu.array().exp().matrix();
    break;
  case Link::probit:
    mu.unaryExpr(&gaussian_cdf);
    //mu = gaussian_cdf_vec(mu);
    break;
  case Link::identity:
    break;
  case Link::inverse:
    mu = mu.array().inverse().matrix();
    break;
  }
  return mu;
}

// inline Eigen::VectorXd mod_inv_func(const Eigen::VectorXd& muin,
//                                     Link link)
// {
//   Eigen::VectorXd mu(muin);
//   switch (link) {
//   case Link::logit:
//     mu = (mu.array().exp() / (1 + mu.array().exp())).matrix();
//     break;
//   case Link::loglink:
//     mu = mu.array().exp().matrix();
//     break;
//   case Link::probit:
//     mu.unaryExpr(&gaussian_cdf);
//     //mu = gaussian_cdf_vec(mu);
//     break;
//   case Link::identity:
//     break;
//   case Link::inverse:
//     mu = mu.array().inverse().matrix();
//     break;
//   }
//   return mu;
// }

inline double mod_inv_func(const double& muin,
                           Link link)
{
  double mu(muin);
  switch (link) {
  case Link::logit:
    mu = 1/(1+std::exp(-1.0*mu));
    break;
  case Link::loglink:
    mu = std::exp(mu);
    break;
  case Link::probit:
    mu = gaussian_cdf(mu);
    break;
  case Link::identity:
    break;
  case Link::inverse:
    mu = 1/mu;
    break;
  }
  return mu;
}


inline Eigen::MatrixXd dhdmu(const Eigen::MatrixXd& xb,
                             const glmmr::Family& family) {
  Eigen::MatrixXd wdiag(xb.rows(), xb.cols());
  Eigen::MatrixXd p(xb.rows(), xb.cols());
  
  switch(family.family){
  case Fam::poisson:
  {
    switch(family.link){
  case Link::identity:
    wdiag = xb.array().exp().matrix();
    break;
  default:
    wdiag = xb.array().exp().inverse().matrix(); 
  break;
  }
    break;
  }
  case Fam::exponential:
  {
    wdiag.setConstant(1.0);
    break;
  }
  case Fam::bernoulli: case Fam::binomial:
  {
    switch(family.link){
  case Link::loglink:
    p = mod_inv_func(xb, family.link);
    wdiag = ((1.0 - p.array()) * p.array().inverse()).matrix();
    break;
  case Link::identity:
    p = mod_inv_func(xb, family.link);
    wdiag = (p.array() * (1 - p.array())).matrix();
    break;
  case Link::probit:
  {
    p = mod_inv_func(xb, family.link);
    Eigen::MatrixXd pinv(xb);
    pinv.unaryExpr(&gaussian_pdf);
    wdiag = ((p.array() * (1.0 - p.array())) * pinv.array().inverse()).matrix();
    break;
  }
  default:
    p = mod_inv_func(xb, family.link);
    wdiag = (p.array() * (1.0 - p.array())).inverse().matrix();
    break;
  }
    break;
  }
  case Fam::gaussian:
  {
    switch(family.link){
  case Link::loglink:
    wdiag = xb.array().exp().inverse().matrix();
    break;
  default:
    //identity
    wdiag.setConstant(1.0);
  break;
  }
    break;
  }
  case Fam::gamma:
  {
    switch(family.link){
  case Link::inverse:
    wdiag = xb.array().square().inverse().matrix();
    break;
  case Link::identity:
    wdiag = xb.array().square().matrix();
    break;
  default:
    //log
    wdiag.setConstant(1.0);
  break;
  }
    break;
  }
  case Fam::beta:
  {
    //only logit currently
    p = mod_inv_func(xb, family.link);
    wdiag = (p.array() * (1.0 - p.array())).inverse().matrix();
    break;
  }
  case Fam::quantile: case Fam::quantile_scaled:
  {
    throw std::runtime_error("Quantile disabled");
    break;
  }
  }
  return wdiag;
}

inline Eigen::VectorXd dhdmu(const Eigen::VectorXd& xb,
                             const glmmr::Family& family) {
  Eigen::VectorXd wdiag(xb.size());
  Eigen::VectorXd p(xb.size());
  
  switch (family.family) {
  case Fam::poisson:
  {
    switch (family.link) {
  case Link::identity:
    wdiag = xb.array().exp().matrix();
    break;
  default:
    wdiag = xb.array().exp().inverse().matrix();
  break;
  }
    break;
  }
  case Fam::exponential:
  {
    wdiag.setConstant(1.0);
    break;
  }
  case Fam::bernoulli: case Fam::binomial:
  {
    switch (family.link) {
  case Link::loglink:
    p = mod_inv_func(xb, family.link);
    wdiag = ((1.0 - p.array()) * p.array().inverse()).matrix();
    break;
  case Link::identity:
    p = mod_inv_func(xb, family.link);
    wdiag = (p.array() * (1 - p.array())).matrix();
    break;
  case Link::probit:
  {
    p = mod_inv_func(xb, family.link);
    VectorXd pinv(xb);
    pinv.unaryExpr(&gaussian_pdf);
    wdiag = ((p.array() * (1.0 - p.array())) * pinv.array().inverse()).matrix();
    break;
  }
  default:
    p = mod_inv_func(xb, family.link);
    wdiag = (p.array() * (1.0 - p.array())).inverse().matrix();
    break;
  }
    break;
  }
  case Fam::gaussian:
  {
    switch (family.link) {
  case Link::loglink:
    wdiag = xb.array().exp().inverse().matrix();
    break;
  default:
    //identity
    wdiag.setConstant(1.0);
  break;
  }
    break;
  }
  case Fam::gamma:
  {
    switch (family.link) {
  case Link::inverse:
    wdiag = xb.array().square().inverse().matrix();
    break;
  case Link::identity:
    wdiag = xb.array().square().matrix();
    break;
  default:
    //log
    wdiag.setConstant(1.0);
  break;
  }
    break;
  }
  case Fam::beta:
  {
    //only logit currently
    p = mod_inv_func(xb, family.link);
    wdiag = (p.array() * (1.0 - p.array())).inverse().matrix();
    break;
  }
  case Fam::quantile: case Fam::quantile_scaled:
  {
    throw std::runtime_error("Quantile disabled");
    break;
  }
  }
  return wdiag;
}

inline Eigen::VectorXd detadmu(const Eigen::VectorXd& xb,
                               const Link link) {
  Eigen::VectorXd wdiag(xb.size());
  Eigen::VectorXd p(xb.size());
  
  switch (link) {
  case Link::loglink:
    wdiag = xb.array().exp().inverse().matrix();
    break;
  case Link::identity:
    wdiag.setConstant(1.0);
    break;
  case Link::logit:
    p = glmmr::maths::mod_inv_func(xb, link);
    wdiag = (p.array() * (1.0 - p.array())).inverse().matrix();
    break;
  case Link::probit:
  {
    Eigen::ArrayXd pinv = gaussian_pdf_vec(xb);
    wdiag = (pinv.inverse()).matrix();
    break;
  }
  case Link::inverse:
    wdiag = xb.array().square().matrix();
    wdiag *= -1.0;
    break;
    
  }
  return wdiag;
}

inline Eigen::MatrixXd detadmu(const Eigen::MatrixXd& xb,
                               const Link link) {
  Eigen::MatrixXd wdiag(xb.rows(), xb.cols());
  
  switch (link) {
  case Link::loglink:
    wdiag = xb.inverse().matrix();
    break;
  case Link::identity:
    wdiag.setConstant(1.0);
    break;
  case Link::logit:
    wdiag = (xb.array() * (1.0 - xb.array())).inverse().matrix();
    break;
  case Link::probit:
  {
    wdiag = (xb.inverse()).matrix();
    break;
  }
  case Link::inverse:
    wdiag = xb.array().square().matrix();
    wdiag *= -1.0;
    break;
    
  }
  return wdiag;
}

inline double normalCDF(double value)
{
  return 0.5 * erfc(-value * sqrt(0.5));
}

inline Eigen::VectorXd attenuted_xb(const Eigen::VectorXd& xb,
                                    const Eigen::MatrixXd& Z,
                                    const Eigen::MatrixXd& D,
                                    const Link link){
  Eigen::ArrayXd xbnew(xb.array());
  int n = xb.size();
  if(link==Link::loglink){
    for(int i=0; i<n; i++){
      xbnew(i) += (Z.row(i)*D*Z.row(i).transpose())(0)/2;
    }
  } else if(link==Link::probit){
    Eigen::ArrayXd zprod(n);
    Eigen::MatrixXd Dzz(D.rows(),D.cols());
    Eigen::PartialPivLU<Eigen::MatrixXd> pluDzz;
    for(int i=0; i<n; i++){
      Dzz = D*Z.row(i).transpose()*Z.row(i) + Eigen::MatrixXd::Identity(D.rows(),D.cols());
      pluDzz = Eigen::PartialPivLU<Eigen::MatrixXd>(Dzz);
      zprod(i) = pluDzz.determinant();
    }
    zprod = zprod.inverse().sqrt();
    xbnew *= zprod;
  } else if(link==Link::logit){
    double c = 0.5880842;
    Eigen::ArrayXd zprod(n);
    Eigen::MatrixXd Dzz(D.rows(),D.cols());
    Eigen::PartialPivLU<Eigen::MatrixXd> pluDzz;
    for(int i=0; i<n; i++){
      Dzz = c*D*Z.row(i).transpose()*Z.row(i) + Eigen::MatrixXd::Identity(D.rows(),D.cols());
      pluDzz = Eigen::PartialPivLU<Eigen::MatrixXd>(Dzz);
      zprod(i) = pluDzz.determinant();
    }
    zprod = zprod.inverse().sqrt();
    xbnew *= zprod;
  }
  
  return xbnew.matrix();
}

inline Eigen::VectorXd marginal_var(const Eigen::VectorXd& mu,
                                    const Fam family,
                                    double var_par = 1.0){
  Eigen::ArrayXd wdiag(mu.size());
  
  switch (family) {
  case Fam::gaussian: case Fam::quantile: case Fam::quantile_scaled:
    wdiag.setConstant(var_par);
    break;
  case Fam::bernoulli: case Fam::binomial:
    wdiag = mu.array()*(1-mu.array());
    break;
  case Fam::poisson: case Fam::exponential:
    wdiag = mu.array();
    break;
  case Fam::gamma:
    wdiag = mu.array().square();
    break;
  case Fam::beta:
    wdiag = mu.array()*(1-mu.array())/(var_par+1);
    break;
  }
  
  return wdiag.matrix();
}

//ramanujans approximation
inline double log_factorial_approx(const double n){
  static const double LOG_2PI = log(3.141593)/2;
  double ans;
  if(n==0){
    ans = 0;
  } else {
    ans = n*log(n) - n + log(n*(1+4*n*(1+2*n)))/6 + LOG_2PI;
  }
  return ans;
}

//ramanujans approximation
inline Eigen::ArrayXd log_factorial_approx(const Eigen::ArrayXd& n){
  static const double LOG_2PI = log(3.141593)/2;
  Eigen::ArrayXd ans(n);
  for(int i = 0; i < ans.size(); i++){
    if(n(i)==0){
      ans(i) = 0;
    } else {
      ans(i) = n(i)*log(n(i)) - n(i) + log(n(i)*(1+4*n(i)*(1+2*n(i))))/6 + LOG_2PI;
    }
  }
  
  return ans;
}

inline double log_likelihood(const double y,
                             const double mu,
                             const double var_par,
                             const glmmr::Family& family) {
  static const double LOG_2PI = log(2* 3.14159265358979323846);
  double logl = 0;
  switch(family.family){
  case Fam::poisson:
  {
    switch(family.link){
  case Link::identity:
  {
    double lf1 = log_factorial_approx(y);
    logl = y*log(mu) - mu-lf1;
    break;
  }
  default:
  {
    double lf1 = glmmr::maths::log_factorial_approx(y);
    logl = y * mu - exp(mu) - lf1;
    break;
  }
  }
    break;
  }
  case Fam::bernoulli:
  {
    switch(family.link){
  case Link::loglink:
    if(y==1){
      logl = mu;
    } else {
      logl = log(1 - exp(mu));
    }
    break;
  case Link::identity:
    if(y==1){
      logl = log(mu);
    } else {
      logl = log(1 - mu);
    }
    break;
  case Link::probit:
  {
    //boost::math::normal norm(0,1);
    if (y == 1) {
    logl = normalCDF(mu);//(double)cdf(norm,mu);
  }
    else {
      logl = log(1 - normalCDF(mu));//(double)cdf(norm, mu)
    }
    break;
  }
  default:
    //logit
    if(y==1){
      logl = log(1/(1+exp(-1.0*mu)));
    } else {
      logl = log(1 - 1/(1+exp(-1.0*mu)));
    }
    break;
  }
    break;
  }
  case Fam::binomial:
  {
    switch(family.link){
  case Link::loglink:
  {
    double lfk = glmmr::maths::log_factorial_approx(y);
    double lfn = glmmr::maths::log_factorial_approx(var_par);
    double lfnk = glmmr::maths::log_factorial_approx(var_par - y);
    logl = lfn - lfk - lfnk + y*mu + (var_par - y)*log(1 - exp(mu));
    break;
  }
  case Link::identity:
  {
    double lfk = glmmr::maths::log_factorial_approx(y);
    double lfn = glmmr::maths::log_factorial_approx(var_par);
    double lfnk = glmmr::maths::log_factorial_approx(var_par - y);
    logl = lfn - lfk - lfnk + y*log(mu) + (var_par - y)*log(1 - mu);
    break;
  }
  case Link::probit:
  {
    //boost::math::normal norm(0, 1);
    double lfk = glmmr::maths::log_factorial_approx(y);
    double lfn = glmmr::maths::log_factorial_approx(var_par);
    double lfnk = glmmr::maths::log_factorial_approx(var_par - y);
    logl = lfn - lfk - lfnk + y*(normalCDF(mu)) + (var_par - y)*log(1 - normalCDF(mu)); //(double)cdf(norm,mu)
    break;
  }
  default:
  {
    double lfk = glmmr::maths::log_factorial_approx(y);
    double lfn = glmmr::maths::log_factorial_approx(var_par);
    double lfnk = glmmr::maths::log_factorial_approx(var_par - y);
    logl = lfn - lfk - lfnk + y*log(1/(1+exp(-1.0*mu))) + (var_par - y)*log(1 - 1/(1+exp(-1.0*mu)));
    break;
  }
  }
    break;
  }
  case Fam::gaussian:
  {
    switch(family.link){
  case Link::loglink:
    logl = -0.5*(log(var_par) + LOG_2PI + (log(y) - mu)*(log(y) - mu)/var_par);
    break;
  default:
    //identity
    logl = -0.5*(log(var_par) + LOG_2PI + (y - mu)*(y - mu)/var_par);
  break;
  }
    break;
  }
  case Fam::gamma:
  {
    switch(family.link){
  case Link::inverse:
  {
    double ymu = var_par*y*mu;
    logl = log(1/(tgamma(var_par)*y)) + var_par*log(ymu) - ymu;
    break;
  }
  case Link::identity:
    logl = log(1/(tgamma(var_par)*y)) + var_par*log(var_par*y/mu) - var_par*y/mu;
    break;
  default:
    //log
    {
      double ymu = var_par*y/exp(mu);
      logl = log(1/(tgamma(var_par)*y)) + var_par*log(ymu) - ymu;
      break;
    }
  }
    break;
  }
  case Fam::beta:
  {
    //only logit currently
    logl = (mu*var_par - 1)*log(y) + ((1-mu)*var_par - 1)*log(1-y) - lgamma(mu*var_par) - lgamma((1-mu)*var_par) + lgamma(var_par);
    break;
  }
  case Fam::quantile: case Fam::quantile_scaled:
  {
    double resid = y - mod_inv_func(mu,family.link);
    // if(family.family == Fam::quantile_scaled) resid *= (1.0/var_par);
    logl = resid <= 0 ? resid*(1.0 - family.quantile) : -1.0*resid*family.quantile;
    break;
  }
  case Fam::exponential:
  {
    logl = mu - y * exp(mu);
    break;
  }
  }
  return logl;
}

inline double log_likelihood(const Eigen::ArrayXd y,
                             const Eigen::ArrayXd mu,
                             const Eigen::ArrayXd var_par,
                             const glmmr::Family& family) {
  static const double LOG_2PI = log(2* 3.14159265358979323846);
  double logl = 0;
  switch(family.family){
  case Fam::poisson:
  {
    switch(family.link){
  case Link::identity:
  {
    Eigen::ArrayXd lf1 = log_factorial_approx(y);
    logl = (y*mu.log() - mu-lf1).sum();
    break;
  }
  default:
  {
    Eigen::ArrayXd lf1 = log_factorial_approx(y);
    logl = (y * mu - mu.exp() - lf1).sum();
    break;
  }
  }
    break;
  }
  case Fam::bernoulli:
  {
    switch(family.link){
  case Link::loglink:
    logl = (y*mu + (1-y)*(1.0 - mu.exp()).log()).sum();
    break;
  case Link::identity:
    logl = (y*mu.log() + (1 - y)*(1-mu).log()).sum();
    break;
  case Link::probit:
  {
    for(int i = 0; i < mu.size();i++)logl += y(i) * normalCDF(mu(i)) + (1-y(i))* log(1 - normalCDF(mu(i)));
    break;
  }
  default:
    //logit
    Eigen::ArrayXd p = (mu.exp().inverse() + 1.0).inverse();
    logl = (y * p.log() + (1-y) * (1 - p).log()).sum();
    break;
  }
    break;
  }
  case Fam::binomial:
  {
    switch(family.link){
  case Link::loglink:
  {
    Eigen::ArrayXd lfk = glmmr::maths::log_factorial_approx(y);
    Eigen::ArrayXd lfn = glmmr::maths::log_factorial_approx(var_par);
    Eigen::ArrayXd lfnk = glmmr::maths::log_factorial_approx(var_par - y);
    logl = (lfn - lfk - lfnk + y*mu + (var_par - y)*(1 - mu.exp()).log()).sum();
    break;
  }
  case Link::identity:
  {
    Eigen::ArrayXd lfk = glmmr::maths::log_factorial_approx(y);
    Eigen::ArrayXd lfn = glmmr::maths::log_factorial_approx(var_par);
    Eigen::ArrayXd lfnk = glmmr::maths::log_factorial_approx(var_par - y);
    logl = (lfn - lfk - lfnk + y*log(mu) + (var_par - y)*(1-mu).log()).sum();
    break;
  }
  case Link::probit:
  {
    //boost::math::normal norm(0, 1);
    Eigen::ArrayXd lfk = glmmr::maths::log_factorial_approx(y);
    Eigen::ArrayXd lfn = glmmr::maths::log_factorial_approx(var_par);
    Eigen::ArrayXd lfnk = glmmr::maths::log_factorial_approx(var_par - y);
    for(int i =0; i< mu.size(); i++){
      double normmu = normalCDF(mu(i));
      logl += lfn(i) - lfk(i) - lfnk(i) + y(i)*normmu + (var_par(i) - y(i))*log(1 - normmu); 
    }
    break;
  }
  default:
  {
    Eigen::ArrayXd lfk = glmmr::maths::log_factorial_approx(y);
    Eigen::ArrayXd lfn = glmmr::maths::log_factorial_approx(var_par);
    Eigen::ArrayXd lfnk = glmmr::maths::log_factorial_approx(var_par - y);
    Eigen::ArrayXd p = (mu.exp().inverse() + 1.0).inverse();
    logl = (lfn - lfk - lfnk + y*p.log() + (var_par - y)*(1-p).log()).sum();
    break;
  }
  }
    break;
  }
  case Fam::gaussian:
  {
    switch(family.link){
  case Link::loglink:
    logl = -0.5*(var_par.log() + LOG_2PI + (y.log() - mu).square()*var_par.inverse()).sum();
    break;
  default:
    //identity
    logl = -0.5*(var_par.log() + LOG_2PI + (y - mu).square() * var_par.inverse()).sum();
  break;
  }
    break;
  }
  case Fam::gamma:
  {
    switch(family.link){
  case Link::inverse:
  {
    Eigen::ArrayXd ymu = var_par*y*mu;
    for(int i = 0; i < mu.size(); i++) logl += log(1/(tgamma(var_par(i))*y(i))) + var_par(i)*log(ymu(i)) - ymu(i);
    break;
  }
  case Link::identity:
    for(int i = 0; i < mu.size(); i++)logl += log(1/(tgamma(var_par(i))*y(i))) + var_par(i)*log(var_par(i)*y(i)/mu(i)) - var_par(i)*y(i)/mu(i);
    break;
  default:
    //log
    {
      Eigen::ArrayXd ymu = var_par*y*mu.exp().inverse();
      for(int i = 0; i < mu.size(); i++)logl += log(1/(tgamma(var_par(i))*y(i))) + var_par(i)*log(ymu(i)) - ymu(i);
      break;
    }
  }
    break;
  }
  case Fam::beta:
  {
    //only logit currently
    for(int i = 0; i < mu.size(); i++)logl += (mu(i)*var_par(i) - 1)*log(y(i)) + ((1-mu(i))*var_par(i) - 1)*log(1-y(i)) - lgamma(mu(i)*var_par(i)) - lgamma((1-mu(i))*var_par(i)) + lgamma(var_par(i));
    break;
  }
  case Fam::quantile: case Fam::quantile_scaled:
  {
    throw std::runtime_error("Quantile disabled");
    break;
  }
  case Fam::exponential:
  {
    logl = (mu - y * mu.exp()).sum();
  }
  }
  return logl;
}

inline double logdet(const Eigen::MatrixXd& M) {
  double ld = 0;
  Eigen::LLT<Eigen::MatrixXd> chol(M);
  auto& U = chol.matrixL();
  for (int i = 0; i < M.rows(); ++i)
    ld += log(U(i,i));
  ld *= 2;
  return ld;
}

inline MatrixXd sample_MVN(const VectorMatrix& mu,
                           int m) {
  int n = mu.vec.size();
  MatrixXd L = mu.mat.llt().matrixL();
  VectorXd z(n);
  MatrixXd samps(n, m);
  for (int i = 0; i < m; i++) {
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution d{ 0.0, 1.0 };
    auto random_norm = [&d, &gen] { return d(gen); };
    for (int j = 0; j < z.size(); j++) z(j) = random_norm();
    samps.col(i) = z;
    samps.col(i) += mu.vec;
  }
  return samps;
}

}

namespace tools {
inline std::vector<int> linseq(int start, int end) {
  std::vector<int> idx;
  for (int i = start; i <= end; i++) {
    idx.push_back(i);
  }
  return idx;
}
}
}

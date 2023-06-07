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
#include "calculator.hpp"
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
        gen_sigma_blocks();
        setup_calculator();
      };
  
  void set_offset(const VectorXd& offset);
  
  void update_beta(const VectorXd &beta);
  
  void update_beta(const dblvec &beta);
  
  void update_beta_extern(const dblvec &beta);
  
  void update_theta(const VectorXd &theta);
  
  void update_theta(const dblvec &theta);
  
  void update_theta_extern(const dblvec &theta);

  void update_u(const MatrixXd &u);
  
  void update_W();
  
  void update_var_par(const double& v);
  
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
  
  double full_log_likelihood();
  
  void ml_theta();
  
  void ml_beta();
  
  void ml_all();
  
  void laplace_ml_beta_u();
  
  void laplace_ml_theta();
  
  void laplace_ml_beta_theta();
  
  void nr_beta();
  
  void laplace_nr_beta_u();
  
  //MatrixXd laplace_hessian(double tol = 1e-4);
  
  vector_matrix b_score();
  
  vector_matrix re_score();
  
  MatrixXd observed_information_matrix();
  
  MatrixXd re_observed_information_matrix();
  
  MatrixXd u(bool scaled = true){
    if(scaled){
      return covariance_.Lu(u_);
    } else {
      return u_;
    }
  }
  
  MatrixXd Zu();
  
  MatrixXd Sigma(bool inverse = false);
  
  MatrixXd information_matrix();
  
  vector_matrix predict_re(const ArrayXXd& newdata_,
               const ArrayXd& newoffset_);
  
  VectorXd predict_xb(const ArrayXXd& newdata_,
                      const ArrayXd& newoffset_);
  
  void mcmc_sample(int warmup,
                   int samples,
                   int adapt = 100);
  
  void mcmc_set_lambda(double lambda);
  
  void mcmc_set_max_steps(int max_steps);
  
  void mcmc_set_refresh(int refresh);
  
  void mcmc_set_target_accept(double target);
  
  void set_trace(int trace);
  
  double aic();
  
  void make_covariance_sparse();
  
  void make_covariance_dense();
  
  ArrayXd optimum_weights(double N, double sigma_sq, VectorXd C, double tol = 1e-5,
                          int max_iter = 501);
  
private:
  ArrayXd size_m_array;
  ArrayXd size_q_array;
  ArrayXd size_n_array;
  ArrayXd size_p_array;
  glmmr::calculator calc_;
  glmmr::calculator vcalc_;
  glmmr::calculator vvcalc_;
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
  std::vector<glmmr::SigmaBlock> sigma_blocks_;
  
  void setup_calculator();
  
  void gen_sigma_blocks();
  
  MatrixXd sigma_block(int b, bool inverse = false);
  
  MatrixXd sigma_builder(int b, bool inverse = false);
  
  MatrixXd information_matrix_by_block(int b);
  
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
    double denomD_;
  public:
    F_likelihood(Model& M,
                 double denomD = 0,
                 bool importance = false) : 
    M_(M),
    G(M_.covariance_.npar()), 
    importance_(importance), 
    ll(0.0), 
    denomD_(denomD) {}
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

inline void glmmr::Model::nr_beta(){
  
  int niter = u_.cols();
  ArrayXd sigmas(niter);
  // MatrixXd zuOffset_ = zu_;
  // zuOffset_.colwise() += offset_;
  // matrix_matrix deriv = calc_.jacobian_and_hessian(linpred_.parameters_,linpred_.Xdata_,zuOffset_);
  // VectorXd Jsum = deriv.mat2.rowwise().sum();
  vector_matrix score = b_score();
  MatrixXd I = MatrixXd::Identity(P_,P_);
  // deriv.mat1 *= -1.0;
  MatrixXd infomat = score.mat.llt().solve(I);
  update_beta(linpred_.parameter_vector() + infomat*score.vec);
  MatrixXd zd = linpred();
#pragma omp parallel for
  for(int i = 0; i < niter; ++i){
    VectorXd zdu = glmmr::maths::mod_inv_func(zd.col(i), link_);
    ArrayXd resid = (y_ - zdu);
    sigmas(i) = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
  }
  var_par_ = sigmas.mean();
  
//   MatrixXd XtXW = MatrixXd::Zero(P_*niter,P_);
//   MatrixXd Wu = MatrixXd::Zero(n_,niter);
//   
//   double nvar_par = 1.0;
//   if(family_=="gaussian"){
//     nvar_par *= var_par_*var_par_;
//   } else if(family_=="Gamma"){
//     nvar_par *= 1/var_par_;
//   } else if(family_=="beta"){
//     nvar_par *= (1+var_par_);
//   } else if(family_=="binomial"){
//     nvar_par *= 1/var_par_;
//   }
//   MatrixXd zd = linpred();
//   
// #pragma omp parallel for
//   for(int i = 0; i < niter; ++i){
//     VectorXd w = glmmr::maths::dhdmu(zd.col(i),family_,link_);
//     w = (w.array().inverse()).matrix();
//     w *= nvar_par;
//     VectorXd zdu = glmmr::maths::mod_inv_func(zd.col(i), link_);
//     ArrayXd resid = (y_ - zdu);
//     sigmas(i) = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
//     XtXW.block(P_*i, 0, P_, P_) = linpred_.X().transpose() * w.asDiagonal() * linpred_.X();
//     VectorXd dmu = glmmr::maths::detadmu(zd.col(i),link_);
//     w = w.cwiseProduct(dmu);
//     w = w.cwiseProduct(resid.matrix());
//     Wu.col(i) = w;
//   }
//   XtXW *= (double)1/niter;
//   MatrixXd XtWXm = XtXW.block(0,0,P_,P_);
//   for(int i = 1; i<niter; i++) XtWXm += XtXW.block(P_*i,0,P_,P_);
//   XtWXm = XtWXm.inverse();
//   VectorXd Wum = Wu.rowwise().mean();
//   VectorXd bincr = XtWXm * (linpred_.X().transpose()) * Wum;
//   update_beta(linpred_.parameter_vector() + bincr);
//   var_par_ = sigmas.mean();
}

inline void glmmr::Model::laplace_nr_beta_u(){
  // update_W();
  VectorXd zd = (linpred()).col(0);
  // VectorXd dmu =  glmmr::maths::detadmu(zd,link_);
  // MatrixXd LZWZL = covariance_.LZWZL(W_);
  // LZWZL = LZWZL.llt().solve(MatrixXd::Identity(LZWZL.rows(),LZWZL.cols()));
  VectorXd zdu =  glmmr::maths::mod_inv_func(zd, link_);
  ArrayXd resid = (y_ - zdu).array();
  double sigmas = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
  // 
  // MatrixXd XtXW = (linpred_.X()).transpose() * W_.asDiagonal() * linpred_.X();
  // VectorXd w = W_;
  // w = w.cwiseProduct(dmu);
  // w = w.cwiseProduct(resid.matrix());
  // XtXW = XtXW.inverse();
  // VectorXd bincr = XtXW * (linpred_.X()).transpose() * w;
  // VectorXd vgrad = log_gradient(u_.col(0));
  // VectorXd vincr = LZWZL * vgrad;
  vector_matrix score = b_score();
  MatrixXd I = MatrixXd::Identity(P_,P_);
  MatrixXd infomat = score.mat.llt().solve(I);
  update_beta(linpred_.parameter_vector() + infomat*score.vec);
  vector_matrix uscore = re_score();
  MatrixXd Ire = MatrixXd::Identity(Q_,Q_);
  MatrixXd infomatre = uscore.mat.llt().solve(Ire);
  update_u(u_.colwise()+infomatre*uscore.vec);
  // update_u(u_.colwise()+vincr);
  // update_beta(linpred_.parameter_vector() + bincr);
  var_par_ = sigmas;
}

inline VectorXd glmmr::Model::log_gradient(const VectorXd &v,
                                           bool beta){
  //size_n_array = xb();
  size_q_array.setZero();
  size_p_array.setZero();
  //sparse ZLt = ZL_;
  //ZLt.transpose();
  //size_n_array += (ZL_*v).array();
  
  if(beta){
    VectorXd zuOffset_ = ZL_*v;
    zuOffset_ += offset_;
    MatrixXd J = calc_.jacobian(linpred_.parameters_,linpred_.Xdata_,zuOffset_);
    size_p_array = J.transpose().rowwise().sum().array();
  } else {
    VectorXd xbOffset_ = linpred_.xb() + offset_;
    MatrixXd J = vcalc_.jacobian(dblvec(v.data(),v.data()+v.size()),
                                                sparse_to_dense(ZL_,false),
                                                xbOffset_);
    size_q_array = (J.transpose().rowwise().sum() - v).array();
  }
  
//   switch (flink){
//   case 1:
//   {
//     size_n_array = size_n_array.exp();
//     if(!beta){
//       size_n_array = y_.array() - size_n_array;
//       size_q_array = ZLt*size_n_array -v.array() ;
//     } else {
//       size_p_array += (linpred_.X().transpose()*(y_-size_n_array.matrix())).array();
//     }
//     break;
//   }
//   case 2:
//   {
//     size_n_array = size_n_array.inverse();
//     size_n_array = y_.array()*size_n_array;
//     size_n_array -= ArrayXd::Ones(n_);
//     if(beta){
//       size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
//     } else {
//       size_q_array =  ZLt*size_n_array-v.array();
//     }
//     break;
//   }
//   case 3:
//   {
//     size_n_array = size_n_array.exp();
//     size_n_array += 1.0;
//     size_n_array = size_n_array.array().inverse();
//     size_n_array -= 1.0;
//     size_n_array += y_.array();
//     if(beta){
//       size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
//     } else {
//       size_q_array =  ZLt*size_n_array-v.array();
//     }
//     break;
//   }
//   case 4:
//   {
// #pragma omp parallel for
//     for(int i = 0; i < n_; i++){
//       if(y_(i)==1){
//         size_n_array(i) = 1;
//       } else if(y_(i)==0){
//         size_n_array(i) = exp(size_n_array(i))/(1-exp(size_n_array(i)));
//       }
//     }
//     if(beta){
//       size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
//     } else {
//       size_q_array =  ZLt*size_n_array-v.array();
//     }
//     break;
//   }
//   case 5:
//   {
// #pragma omp parallel for
//     for(int i = 0; i < n_; i++){
//       if(y_(i)==1){
//         size_n_array(i) = 1/size_n_array(i);
//       } else if(y_(i)==0){
//         size_n_array(i) = -1/(1-size_n_array(i));
//       }
//     }
//     if(beta){
//       size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
//     } else {
//       size_q_array =  ZLt*size_n_array-v.array();
//     }
//     break;
//   }
//   case 6:
//   {
// #pragma omp parallel for
//     for(int i = 0; i < n_; i++){
//       if(y_(i)==1){
//         size_n_array(i) = (double)R::dnorm(size_n_array(i),0,1,false)/((double)R::pnorm(size_n_array(i),0,1,true,false));
//       } else if(y_(i)==0){
//         size_n_array(i) = -1.0*(double)R::dnorm(size_n_array(i),0,1,false)/(1-(double)R::pnorm(size_n_array(i),0,1,true,false));
//       }
//     }
//     if(beta){
//       size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
//     } else {
//       size_q_array =  ZLt*size_n_array-v.array();
//     }
//     break;
//   }
//   case 7:
//   {
//     if(beta){
//     size_p_array += ((1.0/(var_par_*var_par_))*(linpred_.X().transpose()*(y_ - size_n_array.matrix()))).array();
//   } else {
//     size_n_array = y_.array() - size_n_array;
//     size_q_array = (ZLt*size_n_array)-v.array();
//     size_q_array *= 1.0/(var_par_*var_par_);
//   }
//   break;
//   }
//   case 8:
//   {
//     if(beta){
//     size_p_array += ((1.0/(var_par_*var_par_))*(linpred_.X().transpose()*(y_ - size_n_array.matrix()))).array();
//   } else {
//     size_n_array = y_.array() - size_n_array;
//     size_q_array = ZLt*size_n_array-v.array();
//     size_q_array *= 1.0/(var_par_*var_par_);
//   }
//   break;
//   }
//   case 9:
//   {
//     size_n_array *= -1.0;
//     size_n_array = size_n_array.exp();
//     if(beta){
//       size_p_array += (linpred_.X().transpose()*(y_.array()*size_n_array-1).matrix()*var_par_).array();
//     } else {
//       size_n_array *= y_.array();
//       size_q_array = ZLt*size_n_array-v.array();
//       size_q_array *= var_par_;
//     }
//     break;
//   }
//   case 10:
//   {
//     size_n_array = size_n_array.inverse();
//     if(beta){
//       size_p_array += (linpred_.X().transpose()*(size_n_array.matrix()-y_)*var_par_).array();
//     } else {
//       size_n_array -= y_.array();
//       size_q_array = ZLt*size_n_array-v.array();
//       size_q_array *= var_par_;
//     }
//     break;
//   }
//   case 11:
//   {
//     size_n_array = size_n_array.inverse();
//     if(beta){
//       size_p_array += (linpred_.X().transpose()*((y_.array()*size_n_array*size_n_array).matrix() - size_n_array.matrix())*var_par_).array();
//     } else {
//       size_n_array *= (y_.array()*size_n_array - ArrayXd::Ones(n_));
//       size_q_array = ZLt*size_n_array-v.array();
//       size_q_array *= var_par_;
//     }
//     break;
//   }
//   case 12:
//   {
// #pragma omp parallel for
//     for(int i = 0; i < n_; i++){
//       size_n_array(i) = exp(size_n_array(i))/(exp(size_n_array(i))+1);
//       size_n_array(i) = (size_n_array(i)/(1+exp(size_n_array(i)))) * var_par_ * (log(y_(i)) - log(1- y_(i)) - boost::math::digamma(size_n_array(i)*var_par_) + boost::math::digamma((1-size_n_array(i))*var_par_));
//     }
//     if(beta){
//       size_p_array += (linpred_.X().transpose()*size_n_array.matrix()).array();
//     } else {
//       size_q_array = ZLt*size_n_array-v.array();
//     }
//     break;
//   }
//   }
  
  return beta ? size_p_array.matrix() : size_q_array.matrix();
}

#include "likelihood.ipp"
#include "mhmcmc.ipp"
#include "model.ipp"

#endif
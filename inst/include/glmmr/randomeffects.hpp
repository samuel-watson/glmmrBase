#pragma once

#include "covariance.hpp"
#include "modelbits.hpp"

namespace glmmr {

using namespace Eigen;

enum class RandomEffectMargin {
  AtEstimated = 0,
    At = 1,
    AtZero = 2,
    Average = 3
};


template<typename modeltype>
class RandomEffects{
public:
  MatrixXd    u_;
  MatrixXd    scaled_u_;
  MatrixXd    zu_;
  VectorXd    u_mean_;
  VectorXd    u_var_diag_;
  MatrixXd    u_solve_;
  ArrayXd     u_weight_;
  VectorXd    u_loglik_;
  modeltype&  model;
  int         mcmc_block_size = 1; // for saem
  
  RandomEffects(modeltype& model_) : 
    u_(MatrixXd::Zero(model_.covariance.Q(),1)),
    scaled_u_(MatrixXd::Zero(model_.covariance.Q(),1)),
    zu_(model_.n(),1), u_mean_(VectorXd::Zero(model_.covariance.Q())), u_var_diag_(model_.covariance.Q()), 
    u_solve_(MatrixXd::Zero(model_.covariance.Q(),1)),
    u_weight_(VectorXd::Zero(1)), u_loglik_(VectorXd::Zero(1)), model(model_) {};
  
  RandomEffects(modeltype& model_, int n, int Q) : 
    u_(MatrixXd::Zero(Q,1)),
    scaled_u_(MatrixXd::Zero(Q,1)),
    zu_(n,1), u_mean_(Q), u_var_diag_(Q), u_solve_(Q,1),
    u_weight_(1), u_loglik_(1), model(model_) {};
  
  RandomEffects(const glmmr::RandomEffects<modeltype>& re) : u_(re.u_), scaled_u_(re.scaled_u_), 
    zu_(re.zu_), u_mean_(re.u_mean_), u_var_diag_(re.u_var_diag_), u_solve_(re.u_solve_), u_weight_(re.u_weight_),
    u_loglik_(re.u_loglik_), model(re.model) {};                
  
  MatrixXd      Zu(){return zu_;};
  MatrixXd      u(bool scaled = true);
  VectorXd      centred_u_mean();
  
  VectorMatrix  predict_re(const ArrayXXd& newdata_,const ArrayXd& newoffset_);
  
  template<typename C = modeltype,
           typename Enable = std::enable_if_t<std::is_same_v<C, bits_spde>>>
    VectorMatrix predict_re_spde(const ArrayXXd& newdata_,
                                 const ArrayXd&  newoffset_,
                                 const Eigen::SparseMatrix<double>& A_new);
  
  void          update_zu(const bool weights);
};

}

template<typename modeltype>
inline MatrixXd glmmr::RandomEffects<modeltype>::u(bool scaled){
  if(scaled){
    return model.covariance.Lu(u_);
  } else {
    return u_;
  }
}

template<typename modeltype>
inline VectorXd glmmr::RandomEffects<modeltype>::centred_u_mean(){
  MatrixXd umat = model.covariance.ZLu(u_);
  VectorXd umean = umat.rowwise().mean();
  umean.array() -= umat.mean();
  return umean;
}

template<>
inline void glmmr::RandomEffects<bits_spde>::update_zu(const bool weights){
  auto& cov = model.covariance;
  
  if(u_.rows() != cov.nv){
    const int niter = std::max(1, static_cast<int>(u_.cols()));
    u_.resize(cov.nv, niter);
    u_.setZero();
    u_mean_.resize(cov.nv);
    u_mean_.setZero();
  }
  
  // Size check: ZA must match model.n() observations
  if(cov.ZA_.rows() != model.n() || cov.ZA_.cols() != cov.nv){
    // SPDE not fully loaded or in an inconsistent state — return safe defaults
    const int niter = std::max(1, static_cast<int>(u_.cols()));
    zu_.setZero(model.n(), niter);
    scaled_u_ = u_;
    u_solve_.setZero(u_.rows(), niter);
    u_weight_.setConstant(1.0 / niter);
    return;
  }
  
  zu_.noalias() = cov.ZA_ * u_;
  scaled_u_ = u_;   
  u_solve_.resize(u_.rows(), u_.cols());
#pragma omp parallel for
  for(int k = 0; k < u_.cols(); ++k){
    u_solve_.col(k).noalias() = cov.Q_mat * u_.col(k);
  }
  
  ArrayXd xb = model.xb();
  u_weight_.setZero();
  
  if(weights){
    // log|Q| for the prior constant — cache once
    const double half_logdet_Q = 0.5 * cov.log_determinant();   // or compute inline
    
#pragma omp parallel for
    for(int i = 0; i < u_.cols(); ++i){
      double llmod = maths::log_likelihood(model.data.y.array(),
                                           xb + zu_.col(i).array(),
                                           model.data.variance,
                                           model.family);
      double qf = u_.col(i).dot(u_solve_.col(i));
      double llprior = half_logdet_Q - 0.5 * qf;
      u_weight_(i) = llmod + llprior - u_loglik_(i);
    }
    u_weight_ -= u_weight_.maxCoeff();
    u_weight_ = u_weight_.exp();
    u_weight_ *= 1.0 / u_weight_.sum();
  } else {
    u_weight_.setConstant(1.0 / u_.cols());
  }
}

template<typename modeltype>
inline void glmmr::RandomEffects<modeltype>::update_zu(const bool weights){
  scaled_u_ = model.covariance.Lu(u_);
  MatrixXd Z = model.covariance.Z();
  zu_ = Z * scaled_u_;
  ArrayXd xb = model.xb();
  u_solve_ = model.covariance.solve(scaled_u_);
  u_weight_.setZero();
  if(weights){
#pragma omp parallel for 
    for(int i = 0; i < scaled_u_.cols(); i++){
      double llmod = maths::log_likelihood(model.data.y.array(),xb + zu_.col(i).array(), model.data.variance,model.family);
      double llprior = -0.5 * scaled_u_.col(i).dot(u_solve_.col(i)); 
      u_weight_(i) = llmod + llprior - u_loglik_(i);
    }
    u_weight_ -= u_weight_.maxCoeff();
    u_weight_ = u_weight_.exp();
    double weightsum = u_weight_.sum();
    u_weight_ *= 1.0/weightsum;
  } else {
    u_weight_.setConstant(1.0/scaled_u_.cols());
  }
}

template<>
inline void glmmr::RandomEffects<bits_hsgp>::update_zu(const bool weights){
  // u_ is M-dimensional u-space spectral coefficients, u ~ N(0, diag(Λ))
  // zu_ = Z * Phi * u (no sqrt(Lambda) scaling)
  zu_ = model.covariance.ZLu(u_);  // ZLu now returns ZPhi * u
  scaled_u_ = u_;
  u_solve_ = model.covariance.solve(u_);  // diag(1/Λ) * u
  ArrayXd xb = model.xb();
  u_weight_.setZero();
  if(weights){
    ArrayXd Lambda = model.covariance.LambdaSPD();
    double half_logdet = 0.5 * Lambda.log().sum();
    
#pragma omp parallel for
    for(int i = 0; i < u_.cols(); i++){
      double llmod = maths::log_likelihood(model.data.y.array(), xb + zu_.col(i).array(),
                                           model.data.variance, model.family);
      // log prior: -½ Σ log(Λ_k) - ½ Σ u_k²/Λ_k
      double llprior = -half_logdet - 0.5 * (u_.col(i).array().square() / Lambda).sum();
      u_weight_(i) = llmod + llprior - u_loglik_(i);
    }
    u_weight_ -= u_weight_.maxCoeff();
    u_weight_ = u_weight_.exp();
    u_weight_ *= 1.0 / u_weight_.sum();
  } else {
    u_weight_.setConstant(1.0 / u_.cols());
  }
}

template<typename modeltype>
inline VectorMatrix glmmr::RandomEffects<modeltype>::predict_re(const ArrayXXd& newdata_,
                                                            const ArrayXd& newoffset_){
  if(model.covariance.data_.cols()!=newdata_.cols())throw std::runtime_error("Different numbers of columns in new data");
  // generate the merged data
  int nnew = newdata_.rows();
  ArrayXXd mergedata(model.n()+nnew,model.covariance.data_.cols());
  mergedata.topRows(model.n()) = model.covariance.data_;
  mergedata.bottomRows(nnew) = newdata_;
  ArrayXd mergeoffset(model.n()+nnew);
  mergeoffset.head(model.n()) = model.data.offset;
  mergeoffset.tail(nnew) = newoffset_;
  
  Covariance covariancenew(model.covariance.form_,
                           mergedata,
                           model.covariance.colnames_);
  
  covariancenew.update_parameters(model.covariance.parameters_);
  // //generate sigma
  int newQ = covariancenew.Q() - model.covariance.Q();
  VectorMatrix result(newQ);
  result.vec.setZero();
  result.mat.setZero();
  MatrixXd D = covariancenew.D(false,false);
  result.mat = D.block(model.covariance.Q(),model.covariance.Q(),newQ,newQ);
  MatrixXd D22 = D.block(0,0,model.covariance.Q(),model.covariance.Q());
  D22 = D22.llt().solve(MatrixXd::Identity(model.covariance.Q(),model.covariance.Q()));
  MatrixXd D12 = D.block(model.covariance.Q(),0,newQ,model.covariance.Q());
  MatrixXd Lu = model.covariance.Lu(u(false));
  MatrixXd SSV = D12 * D22 * Lu;
  result.vec = SSV.rowwise().mean();
  result.mat -= D12 * D22 * D12.transpose();
  return result;
}

template<>
inline VectorMatrix glmmr::RandomEffects<bits_hsgp>::predict_re(const ArrayXXd& newdata_,
                                                                const ArrayXd& newoffset_){
  if(model.covariance.data_.cols() != newdata_.cols())
    throw std::runtime_error("Different numbers of columns in new data");
  
  hsgpCovariance covnew(model.covariance.form_,
                        newdata_,
                        model.covariance.colnames_);
  
  // Copy basis function settings from fitted model
  covnew.update_approx_parameters(model.covariance.m, model.covariance.L_factor_);
  covnew.update_parameters(model.covariance.parameters_);
  
  // Phi_new is n_new × M, evaluated at new locations
  MatrixXd Phi_new = covnew.PhiSPD(false, false);
  
  // Map u-space samples to predictions at new locations: Phi_new * u
  MatrixXd umat = u(false);  // M × n_iter
  MatrixXd pred = Phi_new * umat;  // n_new × n_iter
  
  int n_new = newdata_.rows();
  int iter = pred.cols();
  VectorMatrix result(n_new);
  result.vec = pred.rowwise().mean();
  result.mat.setZero();
  for(int i = 0; i < iter; i++){
    VectorXd diff = pred.col(i) - result.vec;
    result.mat.noalias() += diff * diff.transpose();
  }
  result.mat.array() *= (1.0 / (double)iter);
  return result;
}

template<>
inline VectorMatrix glmmr::RandomEffects<bits_spde>::predict_re(const ArrayXXd& newdata_,
                                                                const ArrayXd& newoffset_){
  throw std::runtime_error("Use predict_re_spde with SPDE");
}

template<typename modeltype>
template<typename C, typename>
inline VectorMatrix glmmr::RandomEffects<modeltype>::predict_re_spde(
    const ArrayXXd& newdata_,
    const ArrayXd&  newoffset_,
    const Eigen::SparseMatrix<double>& A_new)
{
  // Validate A_new dimensions against the fitted covariance
  if(A_new.cols() != model.covariance.nv){
    throw std::runtime_error(
        "predict_re (SPDE): A_new columns (" + std::to_string(A_new.cols()) +
          ") must equal n_v (" + std::to_string(model.covariance.nv) + ").");
  }
  if(A_new.rows() != newdata_.rows()){
    throw std::runtime_error(
        "predict_re (SPDE): A_new rows (" + std::to_string(A_new.rows()) +
          ") must equal newdata rows (" + std::to_string(newdata_.rows()) + ").");
  }
  
  // Map u-space samples (mesh coefficients) to predictions at new locations
  MatrixXd umat = u(false);                           // n_v × n_iter
  MatrixXd pred = A_new * umat;                       // n_new × n_iter, sparse * dense
  
  const int n_new = newdata_.rows();
  const int iter  = pred.cols();
  VectorMatrix result(n_new);
  result.vec = pred.rowwise().mean();
  result.mat.setZero();
  for(int i = 0; i < iter; ++i){
    VectorXd diff = pred.col(i) - result.vec;
    result.mat.noalias() += diff * diff.transpose();
  }
  result.mat.array() *= (1.0 / static_cast<double>(iter));
  return result;
}


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
  MatrixXd    u_var_;
  VectorXd    u_weight_;
  VectorXd    u_loglik_;
  modeltype&  model;
  int         mcmc_block_size = 1; // for saem
  
  RandomEffects(modeltype& model_) : 
    u_(MatrixXd::Zero(model_.covariance.Q(),1)),
    scaled_u_(MatrixXd::Zero(model_.covariance.Q(),1)),
    zu_(model_.n(),1), u_mean_(VectorXd::Zero(model_.covariance.Q())), 
    u_var_(MatrixXd::Zero(model_.covariance.Q(),model_.covariance.Q())),
    u_weight_(VectorXd::Zero(1)), u_loglik_(VectorXd::Zero(1)), model(model_) {};
  
  RandomEffects(modeltype& model_, int n, int Q) : 
    u_(MatrixXd::Zero(Q,1)),
    scaled_u_(MatrixXd::Zero(Q,1)),
    zu_(n,1), u_mean_(Q), u_var_(Q,Q),
    u_weight_(1), u_loglik_(1), model(model_) {};
  
  RandomEffects(const glmmr::RandomEffects<modeltype>& re) : u_(re.u_), scaled_u_(re.scaled_u_), 
    zu_(re.zu_), u_mean_(re.u_mean_), u_var_(re.u_var_), u_weight_(re.u_weight_),
    u_loglik_(re.u_loglik_), model(re.model) {};                
  
  MatrixXd      Zu(){return zu_;};
  MatrixXd      u(bool scaled = true);
  VectorMatrix  predict_re(const ArrayXXd& newdata_,const ArrayXd& newoffset_);
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
inline void glmmr::RandomEffects<modeltype>::update_zu(const bool weights){
  scaled_u_ = model.covariance.Lu(u_);
  VectorXd umat_means = scaled_u_.colwise().mean();
  //double umean = scaled_u_.mean();
  //scaled_u_.array() -= u_mean_.mean();
  for(int i = 0; i < scaled_u_.cols(); i++) scaled_u_.col(i).array() += -1.0*umat_means(i);
  MatrixXd Z = model.covariance.Z();
  zu_ = Z * scaled_u_;
  ArrayXd xb = model.xb();
  
  if(weights){
#pragma omp parallel for 
    for(int i = 0; i < scaled_u_.cols(); i++){
      double llmod = maths::log_likelihood(model.data.y.array(),xb + zu_.col(i).array(), model.data.variance,model.family);
      double llprior = model.covariance.log_likelihood(scaled_u_.col(i));
      u_weight_(i) = exp(llmod + llprior - u_loglik_(i));
    }
    double weightsum = u_weight_.sum();
    u_weight_ *= 1.0/weightsum;
    //Rcpp::Rcout << "\nweights:\n" << u_weight_.transpose();
  } else {
    u_weight_.setConstant(1.0/scaled_u_.cols());
  }
}

template<>
inline VectorMatrix glmmr::RandomEffects<bits>::predict_re(const ArrayXXd& newdata_,
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
inline VectorMatrix glmmr::RandomEffects<bits_nngp>::predict_re(const ArrayXXd& newdata_,
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
  
  nngpCovariance covariancenew(model.covariance.form_,
                               mergedata,
                               model.covariance.colnames_);
  
  nngpCovariance covariancenewnew(model.covariance.form_,
                                  newdata_,
                                  model.covariance.colnames_);
  
  covariancenewnew.update_parameters(model.covariance.parameters_);
  covariancenew.update_parameters(model.covariance.parameters_);
  // //generate sigma
  int newQ = covariancenewnew.Q();
  VectorMatrix result(newQ);
  result.vec.setZero();
  result.mat.setZero();
  MatrixXd D = covariancenew.D(false,false);
  result.mat = D.block(model.covariance.Q(),model.covariance.Q(),newQ,newQ);
  MatrixXd D22 = D.block(0,0,model.covariance.Q(),model.covariance.Q());
  D22 = D22.llt().solve(MatrixXd::Identity(model.covariance.Q(),model.covariance.Q()));
  MatrixXd D12 = D.block(model.covariance.Q(),0,newQ,model.covariance.Q());
  MatrixXd Lu = u();
  MatrixXd SSV = D12 * D22 * Lu;
  result.vec = SSV.rowwise().mean();
  result.mat -= D12 * D22 * D12.transpose();
  return result;
}

template<>
inline VectorMatrix glmmr::RandomEffects<bits_hsgp>::predict_re(const ArrayXXd& newdata_,
                                                                 const ArrayXd& newoffset_){
  if(model.covariance.data_.cols()!=newdata_.cols())throw std::runtime_error("Different numbers of columns in new data");
  
  hsgpCovariance covariancenewnew(model.covariance.form_,
                                  newdata_,
                                  model.covariance.colnames_);
  
  covariancenewnew.update_parameters(model.covariance.parameters_);
  MatrixXd newLu = covariancenewnew.Lu(u(false));
  int iter = newLu.cols();
  
  // //generate sigma
  int newQ = newdata_.rows();//covariancenewnew.Q();
  VectorMatrix result(newQ);
  result.vec.setZero();
  result.mat.setZero();
  result.vec = newLu.rowwise().mean();
  VectorXd newLuCol(newLu.rows());
  for(int i = 0; i < iter; i++){
    newLuCol = newLu.col(i) - result.vec;
    result.mat += (newLuCol * newLuCol.transpose());
  }
  result.mat.array() *= (1/(double)iter);
  return result;
}


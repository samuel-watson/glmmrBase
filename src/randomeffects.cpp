#include <glmmr/randomeffects.hpp>


template<typename modeltype>
MatrixXd glmmr::RandomEffects<modeltype>::u(bool scaled){
  if(scaled){
    return model.covariance.Lu(u_);
  } else {
    return u_;
  }
}

template<>
VectorMatrix glmmr::RandomEffects<bits>::predict_re(const ArrayXXd& newdata_,
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
  
  Covariance covariancenewnew(model.covariance.form_,
                              newdata_,
                              model.covariance.colnames_);
  
  covariancenewnew.update_parameters(model.covariance.parameters_);
  covariancenew.update_parameters(model.covariance.parameters_);
  // //generate sigma
  int newQ = covariancenewnew.Q();
  VectorMatrix result(1);
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
VectorMatrix glmmr::RandomEffects<bits_nngp>::predict_re(const ArrayXXd& newdata_,
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
VectorMatrix glmmr::RandomEffects<bits_hsgp>::predict_re(const ArrayXXd& newdata_,
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

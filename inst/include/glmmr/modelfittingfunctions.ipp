#ifndef MODELFITTINGFUNCTIONS_IPP
#define MODELFITTINGFUNCTIONS_IPP

inline void glmmr::Model::ml_theta(){
  MatrixXd Lu = covariance_.Lu(u_);
  D_likelihood ddl(*this,Lu);
  Rbobyqa<D_likelihood,dblvec> opt;
  opt.set_lower(lower_t_);
  opt.control.iprint = trace_;
  dblvec start_t = get_start_values(false,true,false);
  opt.minimize(ddl, start_t);
}

inline void glmmr::Model::ml_beta(){
  L_likelihood ldl(*this);
  Rbobyqa<L_likelihood,dblvec> opt;
  opt.control.iprint = trace_;
  dblvec start = get_start_values(true,false,false);
  dblvec lower = get_lower_values(true,false,false);
  opt.set_lower(lower);
  opt.minimize(ldl, start);
  
  calculate_var_par();
  
}

inline void glmmr::Model::ml_all(){
  MatrixXd Lu = covariance_.Lu(u_);
  double denomD = 0;
  for(int i = 0; i < Lu.cols(); i++){
    denomD += covariance_.log_likelihood(Lu.col(i));
  }
  denomD *= 1/Lu.cols();
  F_likelihood dl(*this,denomD,true);
  Rbobyqa<F_likelihood,dblvec> opt;
  dblvec start = get_start_values(true,true,false);
  dblvec lower = get_lower_values(true,true,false);
  opt.set_lower(lower);
  opt.control.iprint = trace_;
  opt.minimize(dl, start);
  
  calculate_var_par();
}

inline void glmmr::Model::laplace_ml_beta_u(){
  LA_likelihood ldl(*this);
  Rbobyqa<LA_likelihood,dblvec> opt;
  opt.control.iprint = trace_;
  dblvec start = get_start_values(true,false,false);
  for(int i = 0; i< Q_; i++)start.push_back(u_(i,0));
  opt.minimize(ldl, start);
  
  calculate_var_par();
}

inline void glmmr::Model::laplace_ml_theta(){
  LA_likelihood_cov ldl(*this);
  Rbobyqa<LA_likelihood_cov,dblvec> opt;
  dblvec lower = get_lower_values(false,true,false);
  dblvec start = get_start_values(false,true,false);
  opt.set_lower(lower);
  opt.minimize(ldl, start);
}

inline void glmmr::Model::laplace_ml_beta_theta(){
  LA_likelihood_btheta ldl(*this);
  Rbobyqa<LA_likelihood_btheta,dblvec> opt;
  dblvec lower = get_lower_values(true,true,false);
  dblvec start = get_start_values(true,true,false);
  opt.set_lower(lower);
  opt.control.iprint = trace_;
  opt.minimize(ldl, start);
  
  calculate_var_par();
}

inline vector_matrix glmmr::Model::b_score(){
  MatrixXd zuOffset_ = zu_;
  zuOffset_.colwise() += offset_;
  matrix_matrix hess = calc_.jacobian_and_hessian(linpred_.parameters_,linpred_.Xdata_,zuOffset_);
  vector_matrix out(hess.mat1.rows());
  out.mat = hess.mat1;
  out.mat *= -1.0;
  out.vec = hess.mat2.rowwise().sum();
  return out;
}

inline matrix_matrix glmmr::Model::hess_and_grad(){
  MatrixXd zuOffset_ = zu_;
  zuOffset_.colwise() += offset_;
  matrix_matrix hess = calc_.jacobian_and_hessian(linpred_.parameters_,linpred_.Xdata_,zuOffset_);
  return hess;
}

inline vector_matrix glmmr::Model::re_score(){
  VectorXd xbOffset_ = linpred_.xb() + offset_;
  matrix_matrix hess = vcalc_.jacobian_and_hessian(dblvec(u_.col(0).data(),u_.col(0).data()+u_.rows()),sparse_to_dense(ZL_,false),Map<MatrixXd>(xbOffset_.data(),xbOffset_.size(),1));
  
  vector_matrix out(Q_);
  hess.mat1 *= -1.0;
  out.mat = hess.mat1 + MatrixXd::Identity(Q_,Q_);
  out.vec = hess.mat2.rowwise().sum();
  out.vec -= u_.col(0);
  return out;
}

inline MatrixXd glmmr::Model::observed_information_matrix(){
  // this works but its too slow doing all the cross partial derivatives
  //MatrixXd XZ(n_,P_+Q_);
  //int iter = zu_.cols();
  //XZ.leftCols(P_) = linpred_.X();
  //XZ.rightCols(Q_) = sparse_to_dense(ZL_,false);
  //MatrixXd result = MatrixXd::Zero(P_+Q_,P_+Q_);
  //MatrixXd I = MatrixXd::Identity(P_+Q_,P_+Q_);
  //dblvec params(P_+Q_);
  //std::copy_n(linpred_.parameters_.begin(),P_,params.begin());
  //for(int i = 0; i < iter; i++){
  //  for(int j = 0; j < Q_; j++){
  //    params[P_+j] = u_(j,i);
  //  }
  //  matrix_matrix hess = vcalc_.jacobian_and_hessian(params,XZ,Map<MatrixXd>(offset_.data(),offset_.size(),1));
  //  result += hess.mat1;
  //}
  //result *= (1.0/iter);
  //return result;
  update_W();
  MatrixXd XtXW = (linpred_.X()).transpose() * W_.asDiagonal() * linpred_.X();
  MatrixXd ZL = sparse_to_dense(ZL_,false);
  MatrixXd XtWZL = (linpred_.X()).transpose() * W_.asDiagonal() * ZL;
  MatrixXd ZLWLZ = ZL.transpose() * W_.asDiagonal() * ZL;
  ZLWLZ += MatrixXd::Identity(Q_,Q_);
  MatrixXd infomat(P_+Q_,P_+Q_);
  infomat.topLeftCorner(P_,P_) = XtXW;
  infomat.topRightCorner(P_,Q_) = XtWZL;
  infomat.bottomLeftCorner(Q_,P_) = XtWZL.transpose();
  infomat.bottomRightCorner(Q_,Q_) = ZLWLZ;
  return infomat;
}

inline MatrixXd glmmr::Model::sandwich_matrix(){
  MatrixXd infomat = observed_information_matrix();
  infomat = infomat.llt().solve(MatrixXd::Identity(P_+Q_,P_+Q_));
  infomat.conservativeResize(P_,P_);
  MatrixXd zuOffset_ = zu_;
  zuOffset_.colwise() += offset_;
  MatrixXd J = calc_.jacobian(linpred_.parameters_,linpred_.Xdata_,zuOffset_);
  MatrixXd sandwich = infomat * (J * J.transpose()) * infomat;
  return sandwich;
}

inline double glmmr::Model::aic(){
  MatrixXd Lu = covariance_.Lu(u_);
  int niter = u_.cols();
  int dof = P_ + covariance_.npar();
  double logl = 0;
#pragma omp parallel for reduction (+:logl) 
  for(int i = 0; i < Lu.cols(); i++){
    logl += covariance_.log_likelihood(Lu.col(i));
  }
  double ll = log_likelihood();
  
  return (-2*( ll + logl ) + 2*dof); 
}

inline vector_matrix glmmr::Model::predict_re(const ArrayXXd& newdata_,
                                              const ArrayXd& newoffset_){
  if(covariance_.data_.cols()!=newdata_.cols())Rcpp::stop("Different numbers of columns in new data");
  // generate the merged data
  int nnew = newdata_.rows();
  ArrayXXd mergedata(n_+nnew,covariance_.data_.cols());
  mergedata.topRows(n_) = covariance_.data_;
  mergedata.bottomRows(nnew) = newdata_;
  ArrayXd mergeoffset(n_+nnew);
  mergeoffset.head(n_) = offset_;
  mergeoffset.tail(nnew) = newoffset_;
  glmmr::Covariance covariancenew_(formula_,
                                   mergedata,
                                   covariance_.colnames_,
                                   covariance_.parameters_);
  glmmr::Covariance covariancenewnew_(formula_,
                                      newdata_,
                                      covariance_.colnames_,
                                      covariance_.parameters_);
  glmmr::LinearPredictor newlinpred_(formula_,
                                     mergedata,
                                     linpred_.colnames(),
                                     linpred_.parameters_);
  // //generate sigma
  int newQ_ = covariancenewnew_.Q();
  vector_matrix result(newQ_);
  result.vec.setZero();
  result.mat.setZero();
  MatrixXd D = covariancenew_.D(false,false);
  result.mat = D.block(Q_,Q_,newQ_,newQ_);
  MatrixXd D22 = D.block(0,0,Q_,Q_);
  D22 = D22.llt().solve(MatrixXd::Identity(Q_,Q_));
  MatrixXd D12 = D.block(Q_,0,newQ_,Q_);
  MatrixXd Lu = covariance_.Lu(u_);
  MatrixXd SSV = D12 * D22 * Lu;
  result.vec = SSV.rowwise().mean();
  MatrixXd D121 = D12 * D22 * D12.transpose();
  result.mat -= D12 * D22 * D12.transpose();
  return result;
}

inline void glmmr::Model::nr_beta(){
  int niter = u_.cols();
  MatrixXd zd = linpred();
  ArrayXd sigmas(niter);
  
  if(linpred_.calc_.any_nonlinear){
    vector_matrix score = b_score();
    MatrixXd infomat = score.mat.llt().solve(MatrixXd::Identity(P_,P_));
    update_beta(linpred_.parameter_vector() + infomat*score.vec);
  } else {
    MatrixXd XtXW = MatrixXd::Zero(P_*niter,P_);
    MatrixXd Wu = MatrixXd::Zero(n_,niter);
    
    double nvar_par = 1.0;
    if(family_=="gaussian"){
      nvar_par *= var_par_*var_par_;
    } else if(family_=="Gamma"){
      nvar_par *= 1/var_par_;
    } else if(family_=="beta"){
      nvar_par *= (1+var_par_);
    } else if(family_=="binomial"){
      nvar_par *= 1/var_par_;
    }
    
#pragma omp parallel for 
    for(int i = 0; i < niter; ++i){
      VectorXd w = glmmr::maths::dhdmu(zd.col(i),family_,link_);
      w = (w.array().inverse()).matrix();
      w *= nvar_par;
      VectorXd zdu = glmmr::maths::mod_inv_func(zd.col(i), link_);
      ArrayXd resid = (y_ - zdu);
      XtXW.block(P_*i, 0, P_, P_) = linpred_.X().transpose() * w.asDiagonal() * linpred_.X();
      VectorXd dmu = glmmr::maths::detadmu(zd.col(i),link_);
      w = w.cwiseProduct(dmu);
      w = w.cwiseProduct(resid.matrix());
      Wu.col(i) = w;
    }
    XtXW *= (double)1/niter;
    MatrixXd XtWXm = XtXW.block(0,0,P_,P_);
    for(int i = 1; i<niter; i++) XtWXm += XtXW.block(P_*i,0,P_,P_);
    XtWXm = XtWXm.inverse();
    VectorXd Wum = Wu.rowwise().mean();
    VectorXd bincr = XtWXm * (linpred_.X().transpose()) * Wum;
    update_beta(linpred_.parameter_vector() + bincr);
  }
  
#pragma omp parallel for 
  for(int i = 0; i < niter; ++i){
    VectorXd zdu1 = glmmr::maths::mod_inv_func(zd.col(i), link_);
    ArrayXd resid = (y_ - zdu1);
    sigmas(i) = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
  }
  update_var_par(sigmas.mean());
}

inline void glmmr::Model::laplace_nr_beta_u(){
  update_W();
  VectorXd zd = (linpred()).col(0);
  VectorXd dmu =  glmmr::maths::detadmu(zd,link_);
  MatrixXd infomat = observed_information_matrix();
  infomat = infomat.llt().solve(MatrixXd::Identity(P_+Q_,P_+Q_));
  VectorXd zdu =  glmmr::maths::mod_inv_func(zd, link_);
  ArrayXd resid = (y_ - zdu).array();
  double sigmas = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
  VectorXd w = W_;
  w = w.cwiseProduct(dmu);
  w = w.cwiseProduct(resid.matrix());
  VectorXd params(P_+Q_);
  params.head(P_) = linpred_.parameter_vector();
  params.tail(Q_) = u_.col(0);
  VectorXd pderiv(P_+Q_);
  pderiv.head(P_) = (linpred_.X()).transpose() * w;
  pderiv.tail(Q_) = log_gradient(u_.col(0));
  
  params += infomat*pderiv;
  
  update_beta(params.head(P_));
  update_u(params.tail(Q_));
  update_var_par(sigmas);
  
}

inline VectorXd glmmr::Model::log_gradient(const VectorXd &v,
                                           bool beta){
  size_n_array = xb();
  size_q_array.setZero();
  size_p_array.setZero();
  sparse ZLt = ZL_;
  ZLt.transpose();
  size_n_array += (ZL_*v).array();
  
  if(beta){
    VectorXd zuOffset_ = ZL_*v;
    zuOffset_ += offset_;
    MatrixXd J = calc_.jacobian(linpred_.parameters_,linpred_.Xdata_,zuOffset_);
    size_p_array = J.transpose().rowwise().sum().array();
  } else {
    switch (flink){
    case 1:
  {
    size_n_array = size_n_array.exp();
    if(!beta){
      size_n_array = y_.array() - size_n_array;
      size_q_array = ZLt*size_n_array -v.array() ;
    } else {
      size_p_array += (linpred_.X().transpose()*(y_-size_n_array.matrix())).array();
    }
    break;
  }
    case 2:
  {
    size_n_array = size_n_array.inverse();
    size_n_array = y_.array()*size_n_array;
    size_n_array -= ArrayXd::Ones(n_);
    if(beta){
      size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 3:
  {
    size_n_array = size_n_array.exp();
    size_n_array += 1.0;
    size_n_array = size_n_array.array().inverse();
    size_n_array -= 1.0;
    size_n_array += y_.array();
    if(beta){
      size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 4:
  {
#pragma omp parallel for 
    for(int i = 0; i < n_; i++){
      if(y_(i)==1){
        size_n_array(i) = 1;
      } else if(y_(i)==0){
        size_n_array(i) = exp(size_n_array(i))/(1-exp(size_n_array(i)));
      }
    }
    if(beta){
      size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 5:
  {
#pragma omp parallel for 
    for(int i = 0; i < n_; i++){
      if(y_(i)==1){
        size_n_array(i) = 1/size_n_array(i);
      } else if(y_(i)==0){
        size_n_array(i) = -1/(1-size_n_array(i));
      }
    }
    if(beta){
      size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 6:
  {
#pragma omp parallel for 
    for(int i = 0; i < n_; i++){
      if(y_(i)==1){
        size_n_array(i) = (double)R::dnorm(size_n_array(i),0,1,false)/((double)R::pnorm(size_n_array(i),0,1,true,false));
      } else if(y_(i)==0){
        size_n_array(i) = -1.0*(double)R::dnorm(size_n_array(i),0,1,false)/(1-(double)R::pnorm(size_n_array(i),0,1,true,false));
      }
    }
    if(beta){
      size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 7:
  {
    if(beta){
    size_p_array += ((1.0/(var_par_*var_par_))*(linpred_.X().transpose()*(y_ - size_n_array.matrix()))).array();
  } else {
    size_n_array = y_.array() - size_n_array;
    size_q_array = (ZLt*size_n_array)-v.array();
    size_q_array *= 1.0/(var_par_*var_par_);
  }
  break;
  }
    case 8:
  {
    if(beta){
    size_p_array += ((1.0/(var_par_*var_par_))*(linpred_.X().transpose()*(y_ - size_n_array.matrix()))).array();
  } else {
    size_n_array = y_.array() - size_n_array;
    size_q_array = ZLt*size_n_array-v.array();
    size_q_array *= 1.0/(var_par_*var_par_);
  }
  break;
  }
    case 9:
  {
    size_n_array *= -1.0;
    size_n_array = size_n_array.exp();
    if(beta){
      size_p_array += (linpred_.X().transpose()*(y_.array()*size_n_array-1).matrix()*var_par_).array();
    } else {
      size_n_array *= y_.array();
      size_q_array = ZLt*size_n_array-v.array();
      size_q_array *= var_par_;
    }
    break;
  }
    case 10:
  {
    size_n_array = size_n_array.inverse();
    if(beta){
      size_p_array += (linpred_.X().transpose()*(size_n_array.matrix()-y_)*var_par_).array();
    } else {
      size_n_array -= y_.array();
      size_q_array = ZLt*size_n_array-v.array();
      size_q_array *= var_par_;
    }
    break;
  }
    case 11:
  {
    size_n_array = size_n_array.inverse();
    if(beta){
      size_p_array += (linpred_.X().transpose()*((y_.array()*size_n_array*size_n_array).matrix() - size_n_array.matrix())*var_par_).array();
    } else {
      size_n_array *= (y_.array()*size_n_array - ArrayXd::Ones(n_));
      size_q_array = ZLt*size_n_array-v.array();
      size_q_array *= var_par_;
    }
    break;
  }
    case 12:
  {
#pragma omp parallel for 
    for(int i = 0; i < n_; i++){
      size_n_array(i) = exp(size_n_array(i))/(exp(size_n_array(i))+1);
      size_n_array(i) = (size_n_array(i)/(1+exp(size_n_array(i)))) * var_par_ * (log(y_(i)) - log(1- y_(i)) - boost::math::digamma(size_n_array(i)*var_par_) + boost::math::digamma((1-size_n_array(i))*var_par_));
    }
    if(beta){
      size_p_array += (linpred_.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array = ZLt*size_n_array-v.array();
    }
    break;
  }
    }
  }
  
  // we can use autodiff here, but the above method is faster
  // else {
  //   VectorXd xbOffset_ = linpred_.xb() + offset_;
  //   MatrixXd J = vcalc_.jacobian(dblvec(v.data(),v.data()+v.size()),
  //                                               sparse_to_dense(ZL_,false),
  //                                               xbOffset_);
  //   size_q_array = (J.transpose().rowwise().sum() - v).array();
  // }
  
  
  
  return beta ? size_p_array.matrix() : size_q_array.matrix();
}


inline std::vector<MatrixXd> glmmr::Model::sigma_derivatives(){
  std::vector<MatrixXd> derivs;
  covariance_.derivatives(derivs,2);
  return derivs;
}

inline MatrixXd glmmr::Model::information_matrix_theta(){
  if(family_=="gamma" || family_=="beta")Rcpp::stop("Not currently supported for gamma or beta families");
  int R = covariance_.npar();
  int Rmod = family_=="gaussian" ? R+1 : R;
  MatrixXd M_theta = MatrixXd::Zero(Rmod,Rmod);
  std::vector<MatrixXd> A_matrix;
  MatrixXd SigmaInv = Sigma(true);
  MatrixXd Z = covariance_.Z();
  std::vector<MatrixXd> derivs;
  covariance_.derivatives(derivs,1);
  for(int i = 0; i < R; i++){
    A_matrix.push_back(SigmaInv*Z*derivs[1+i]*Z.transpose());
  }
  if(family_=="gaussian"){
    A_matrix.push_back(2*var_par_*SigmaInv);
  }
  for(int i = 0; i < Rmod; i++){
    for(int j = i; j < Rmod; j++){
      M_theta(i,j) = 0.5 * (A_matrix[i]*A_matrix[j]).trace();
      if(i!=j)M_theta(j,i)=M_theta(i,j);
    }
  }
  return M_theta;
}

inline matrix_matrix glmmr::Model::kenward_roger(){
  if(family_=="gamma" || family_=="beta")Rcpp::stop("Not currently supported for gamma or beta families");
  int R = covariance_.npar();
  int Rmod = family_=="gaussian" ? R+1 : R;
  MatrixXd M_theta = MatrixXd::Zero(Rmod,Rmod);
  std::vector<MatrixXd> A_matrix;
  MatrixXd SigmaInv = Sigma(true);
  MatrixXd Z = covariance_.Z();
  MatrixXd M = information_matrix();
  M = M.llt().solve(MatrixXd::Identity(P_,P_));
  MatrixXd X = linpred_.X();
  MatrixXd SigX = SigmaInv*X;
  MatrixXd middle = MatrixXd::Identity(n_,n_) - X*M*SigX.transpose();
  
  std::vector<MatrixXd> derivs;
  covariance_.derivatives(derivs,2);
  for(int i = 0; i < R; i++){
    A_matrix.push_back(Z*derivs[1+i]*Z.transpose());
  }
  if(family_=="gaussian"){
    A_matrix.push_back(2*var_par_*var_par_*MatrixXd::Identity(n_,n_));
  }
  
  //possible parallelisation?
  for(int i = 0; i < Rmod; i++){
    for(int j = i; j < Rmod; j++){
      M_theta(i,j) = (SigmaInv*A_matrix[i]*SigmaInv*A_matrix[j]).trace();
      M_theta(i,j) -= (M*SigX.transpose()*A_matrix[i]*SigmaInv*(middle+MatrixXd::Identity(n_,n_))*A_matrix[j]*SigX).trace();
      M_theta(i,j) *= 0.5;
      if(i!=j)M_theta(j,i)=M_theta(i,j);
    }
  }
  
  M_theta = M_theta.llt().solve(MatrixXd::Identity(Rmod,Rmod));
  MatrixXd meat = MatrixXd::Zero(P_,P_);
  for(int i = 0; i < Rmod; i++){
    for(int j = 0; j < Rmod; j++){
      int scnd_idx = i <= j ? i + j*(R-1) - j*(j-1)/2 : j + i*(R-1) - i*(i-1)/2;
      meat += M_theta(i,j)*SigX.transpose()*A_matrix[i]*SigmaInv*middle*A_matrix[j]*SigX;
      if(i < R && j < R){
        meat -= M_theta(i,j)*0.25*SigX.transpose()*Z*derivs[R+1+scnd_idx]*Z.transpose()*SigX;
      }
      if(i==R && j==R){
        meat -= M_theta(i,j)*0.5*SigX.transpose()*SigX;
      }
    }
  }
  
  M += 2*M*meat*M;
  matrix_matrix out(P_,P_,Rmod,Rmod);
  out.mat1 = M;
  out.mat2 = M_theta;
  return out;
}

#endif
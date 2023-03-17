#ifndef MODEL_IPP
#define MODEL_IPP


inline void glmmr::Model::set_offset(const Eigen::VectorXd& offset){
  if(offset.size()!=n_)Rcpp::stop("offset wrong length");
    offset_ = offset;
}

inline void glmmr::Model::update_beta(const Eigen::VectorXd &beta){
  if(beta.size()!=P_)Rcpp::stop("beta wrong length");
    linpred_.update_parameters(beta.array());
}

inline void glmmr::Model::update_beta(const dblvec &beta){
  if(beta.size()!=P_)Rcpp::stop("beta wrong length");
    linpred_.update_parameters(beta);
}

inline void glmmr::Model::update_beta_extern(const dblvec &beta){
  if(beta.size()!=P_)Rcpp::stop("beta wrong length");
    linpred_.update_parameters(beta);
}

inline void glmmr::Model::update_theta(const Eigen::VectorXd &theta){
  if(theta.size()!=covariance_.npar())Rcpp::stop("theta wrong length");
    covariance_.update_parameters(theta.array());
  L_ = covariance_.D(true,false);
  ZL_ = Z_*L_;
  if(useLflag)zu_ = ZL_*u_;
}

inline void glmmr::Model::update_theta(const dblvec &theta){
  if(theta.size()!=covariance_.npar())Rcpp::stop("theta wrong length");
    covariance_.update_parameters(theta);
  L_ = covariance_.D(true,false);
  ZL_ = Z_*L_;
  if(useLflag)zu_ = ZL_*u_;
}

inline void glmmr::Model::update_theta_extern(const dblvec &theta){
  if(theta.size()!=covariance_.npar())Rcpp::stop("theta wrong length");
    covariance_.update_parameters(theta);
  L_ = covariance_.D(true,false);
  ZL_ = Z_*L_;
  if(useLflag)zu_ = ZL_*u_;
}

inline void glmmr::Model::update_u(const Eigen::MatrixXd &u){
  if(u.rows()!=Q_)Rcpp::stop("u has wrong number of random effects");
    if(u.cols()!=u_.cols()){
      Rcpp::Rcout << "\nDifferent numbers of random effect samples";
      u_.resize(Q_,u.cols());
      zu_.resize(Q_,u.cols());
      size_m_array.resize(u.cols());
    }
    u_ = u;
    zu_ = useLflag ? ZL_*u : Z_*u;
}

inline void glmmr::Model::update_W(int i){
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
  
  if(attenuate_){
    size_n_array = glmmr::maths::attenuted_xb(xb(),Z_,covariance_.D(),link_);
  } else {
    size_n_array = xb();
  }
  W_ = glmmr::maths::dhdmu(size_n_array,family_,link_);
  W_.noalias() = (W_.array().inverse()).matrix();
  W_ *= nvar_par;
}

inline double glmmr::Model::log_prob(const Eigen::VectorXd &v){
  Eigen::VectorXd mu = xb() + ZL_*v;
  double lp1 = 0;
  double lp2 = 0;
#pragma omp parallel for reduction (+:lp1)
  for(int i = 0; i < n_; i++){
    lp1 += glmmr::maths::log_likelihood(y_(i),mu(i),var_par_,flink);
  }
//#pragma omp parallel for reduction (+:lp2)
//  for(int i = 0; i < v.size(); i++){
//    lp2 += glmmr::maths::log_likelihood(v(i),0,1,7);
//  }
  lp2 = -0.5*v.transpose()*v;
  return lp1+lp2;
}

inline Eigen::VectorXd glmmr::Model::log_gradient(const Eigen::VectorXd &v,
                                                              bool usezl,
                                                              bool beta){
  //note that this function DOES NOT calculate the gradient correctly for usezl false because 
  // the matrix ZL is used in the calculations below. This will be updated, but usezl false is 
  // not currently used anywhere so this has been left for now.
  size_n_array = xb();
  size_q_array.setZero();
  size_p_array.setZero();
  if(usezl){
    size_n_array += (ZL_*v).array();
    if(!beta)size_q_array = -1.0*v.array();
  } else {
    size_n_array += (Z_*v).array();
    if(!beta)size_q_array = (-1.0*covariance_.D()*v).array();
  }
  
  switch (flink){
  case 1:
  {
    size_n_array = size_n_array.exp();
    if(!beta){
      size_q_array += (ZL_.transpose()*(y_-size_n_array.matrix())).array();
    } else {
      size_p_array += (linpred_.X().transpose()*(y_-size_n_array.matrix())).array();
    }
    break;
  }
  case 2:
  {
    size_n_array = size_n_array.inverse();
    size_n_array = y_.array()*size_n_array;
    size_n_array -= Eigen::ArrayXd::Ones(n_);
    if(beta){
      size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array +=  (ZL_.transpose()*size_n_array.matrix()).array();
    }
    break;
  }
  case 3:
  {
    size_n_array = size_n_array.exp();
    size_n_array += Eigen::ArrayXd::Ones(n_);
    size_n_array = size_n_array.array().inverse();
    size_n_array -= Eigen::ArrayXd::Ones(n_);
    size_n_array += y_.array();
    if(beta){
      size_p_array +=  (linpred_.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array +=  (ZL_.transpose()*size_n_array.matrix()).array();
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
      size_q_array +=  (ZL_.transpose()*size_n_array.matrix()).array();
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
      size_q_array +=  (ZL_.transpose()*size_n_array.matrix()).array();
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
      size_q_array +=  (ZL_.transpose()*size_n_array.matrix()).array();
    }
    break;
  }
  case 7:
  {
    if(beta){
    size_p_array += ((1.0/(var_par_*var_par_))*(linpred_.X().transpose()*(y_ - size_n_array.matrix()))).array();
  } else {
    size_q_array += ((1.0/(var_par_*var_par_))*(ZL_.transpose()*(y_ - size_n_array.matrix()))).array();
  }
  break;
  }
  case 8: 
  {
    if(beta){
    size_p_array += ((1.0/(var_par_*var_par_))*(linpred_.X().transpose()*(y_ - size_n_array.matrix()))).array();
  } else {
    size_q_array += ((1.0/(var_par_*var_par_))*(ZL_.transpose()*(y_ - size_n_array.matrix()))).array();
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
      size_q_array += (ZL_.transpose()*(y_.array()*size_n_array-1).matrix()*var_par_).array();
    }
    break;
  }
  case 10:
  {
    size_n_array = size_n_array.inverse();
    if(beta){
      size_p_array += (linpred_.X().transpose()*(size_n_array.matrix()-y_)*var_par_).array();
    } else {
      size_q_array += (ZL_.transpose()*(size_n_array.matrix()-y_)*var_par_).array();
    }
    break;
  }
  case 11:
  {
    size_n_array = size_n_array.inverse();
    if(beta){
      size_p_array += (linpred_.X().transpose()*((y_.array()*size_n_array*size_n_array).matrix() - size_n_array.matrix())*var_par_).array();
    } else {
      size_q_array += (ZL_.transpose()*((y_.array()*size_n_array*size_n_array).matrix() - size_n_array.matrix())*var_par_).array();
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
      size_q_array += (ZL_.transpose()*size_n_array.matrix()).array();
    }
    break;
  }
  }
  
  return beta ? size_p_array.matrix() : size_q_array.matrix();
}


inline double glmmr::Model::log_likelihood() { 
  double ll = 0;
  size_n_array = xb();
  
#pragma omp parallel for reduction (+:ll)
  for(int j=0; j<u_.cols() ; j++){
    for(int i = 0; i<n_; i++){
      ll += glmmr::maths::log_likelihood(y_(i),size_n_array(i) + zu_(i,j),var_par_,flink);
    }
  }
  return ll/u_.cols();
}


inline dblvec glmmr::Model::get_start_values(bool beta, bool theta, bool var){
  dblvec start;
  if(beta){
  for(int i =0 ; i < P_; i++)start.push_back(linpred_.parameters_(i));
    
    if(theta){
      for(int i=0; i< covariance_.npar(); i++) {
        start.push_back(covariance_.parameters_[i]);
      }
    }
  } else {
    start = covariance_.parameters_;
  }
  if(var && (family_=="gaussian"||family_=="Gamma"||family_=="beta")){
    start.push_back(var_par_);
  }
  return start;
}


inline dblvec glmmr::Model::get_lower_values(bool beta, bool theta, bool var){
  dblvec lower;
  if(beta){
    lower = lower_b_;
    if(theta){
      for(int i=0; i< Q_; i++) {
        lower.push_back(lower_t_[i]);
      }
    }
  } else {
    lower = lower_t_;
  }
  if(var && (family_=="gaussian"||family_=="Gamma"||family_=="beta")){
    lower.push_back(0.0);
  }
  return lower;
}


inline dblvec glmmr::Model::get_upper_values(bool beta, bool theta, bool var){
  dblvec upper;
  if(beta){
    upper = upper_b_;
    if(theta){
      for(int i=0; i< Q_; i++) {
        upper.push_back(upper_t_[i]);
      }
    }
  } else {
    upper = upper_t_;
  }
  if(var && (family_=="gaussian"||family_=="Gamma"||family_=="beta")){
    upper.push_back(R_PosInf);
  }
  return upper;
}


inline void glmmr::Model::ml_theta(){
  D_likelihood ddl(*this);
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
  dblvec start = get_start_values(true,false);
  dblvec lower = get_lower_values(true,false);
  opt.set_lower(lower);
  opt.minimize(ldl, start);
}

inline void glmmr::Model::ml_all(){
  F_likelihood dl(*this,true);
  Rbobyqa<F_likelihood,dblvec> opt;
  dblvec start = get_start_values(true,true);
  dblvec lower = get_lower_values(true,true);
  opt.set_lower(lower);
  opt.control.iprint = trace_;
  opt.minimize(dl, start);
}

inline void glmmr::Model::laplace_ml_beta_u(){
  LA_likelihood ldl(*this);
  Rbobyqa<LA_likelihood,dblvec> opt;
  opt.control.iprint = trace_;
  dblvec start = get_start_values(true,false,false);
  for(int i = 0; i< Q_; i++)start.push_back(u_(i,0));
  opt.minimize(ldl, start);
}

inline void glmmr::Model::laplace_ml_theta(){
  LA_likelihood_cov ldl(*this);
  Rbobyqa<LA_likelihood_cov,dblvec> opt;
  dblvec lower = get_lower_values(false,true);
  dblvec start = get_start_values(false,true);
  opt.set_lower(lower);
  opt.minimize(ldl, start);
}

inline void glmmr::Model::laplace_ml_beta_theta(){
  LA_likelihood_btheta ldl(*this);
  Rbobyqa<LA_likelihood_btheta,dblvec> opt;
  dblvec lower = get_lower_values(true,true);
  dblvec start = get_start_values(true,true);
  opt.set_lower(lower);
  opt.control.iprint = trace_;
  opt.minimize(ldl, start);
}

inline void glmmr::Model::nr_beta(){
  
  int niter = u_.cols();
  Eigen::ArrayXd sigmas(niter);
  
  Eigen::MatrixXd XtXW = Eigen::MatrixXd::Zero(P_*niter,P_);
  Eigen::MatrixXd Wu = Eigen::MatrixXd::Zero(n_,niter);
  
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
  Eigen::MatrixXd zd = linpred();
  
#pragma omp parallel for
  for(int i = 0; i < niter; ++i){
    Eigen::VectorXd w = glmmr::maths::dhdmu(zd.col(i),family_,link_);
    w = (w.array().inverse()).matrix();
    w *= 1/nvar_par;
    Eigen::VectorXd zdu = glmmr::maths::mod_inv_func(zd.col(i), link_);
    Eigen::ArrayXd resid = (y_ - zdu);
    sigmas(i) = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
    XtXW.block(P_*i, 0, P_, P_) = linpred_.X().transpose() * w.asDiagonal() * linpred_.X();
    Eigen::VectorXd dmu = glmmr::maths::detadmu(zd.col(i),link_);
    w = w.cwiseProduct(dmu);
    w = w.cwiseProduct(resid.matrix());
    Wu.col(i) = w;
  }
  XtXW *= (double)1/niter;
  Eigen::MatrixXd XtWXm = XtXW.block(0,0,P_,P_);
  for(int i = 1; i<niter; i++) XtWXm += XtXW.block(P_*i,0,P_,P_);
  XtWXm = XtWXm.inverse();
  Eigen::VectorXd Wum = Wu.rowwise().mean();
  Eigen::VectorXd bincr = XtWXm * (linpred_.X().transpose()) * Wum;
  update_beta(linpred_.parameters_ + bincr);
  var_par_ = sigmas.mean();
}

inline void glmmr::Model::laplace_nr_beta_u(){
  double sigmas;
  update_W();
  Eigen::VectorXd zd = (linpred()).col(0);
  Eigen::VectorXd dmu = glmmr::maths::detadmu(zd,link_);
  
  Eigen::MatrixXd LZWZL = ZL_.transpose() * W_.asDiagonal() * ZL_;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(LZWZL.rows(),LZWZL.cols());
  LZWZL.noalias() += I;
  LZWZL = LZWZL.llt().solve(I);
  
  Eigen::VectorXd zdu = glmmr::maths::mod_inv_func(zd, link_);
  Eigen::ArrayXd resid = (y_ - zdu).array();
  sigmas = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
  
  Eigen::MatrixXd XtXW = linpred_.X().transpose() * W_.asDiagonal() * linpred_.X();
  Eigen::VectorXd w = W_;
  w = w.cwiseProduct(dmu);
  w = w.cwiseProduct(resid.matrix());
  
  XtXW = XtXW.inverse();
  Eigen::VectorXd bincr = XtXW * (linpred_.X()).transpose() * w;
  Eigen::VectorXd vgrad = log_gradient(u_.col(0));
  Eigen::VectorXd vincr = LZWZL * vgrad;
  update_u(u_.colwise()+vincr);
  update_beta(linpred_.parameters_ + bincr);
  var_par_ = sigmas;
}

inline Eigen::MatrixXd glmmr::Model::laplace_hessian(double tol){
  LA_likelihood_btheta hdl(*this);
  int nvar = P_ + covariance_.npar();
  if(family_=="gaussian"||family_=="Gamma"||family_=="beta")nvar++;
  dblvec ndep(nvar,tol);
  hdl.os.ndeps_ = ndep;
  dblvec hessian(nvar * nvar,0.0);
  dblvec start = get_start_values(true,true,false);
  hdl.Hessian(start,hessian);
  Eigen::MatrixXd hess = Eigen::Map<Eigen::MatrixXd>(hessian.data(),nvar,nvar);
  return hess;
}

inline Eigen::MatrixXd glmmr::Model::hessian(double tol){
  int npars = P_+covariance_.npar();
  F_likelihood fhdl(*this);
  fhdl.os.usebounds_ = 1;
  dblvec start = get_start_values(true,true,false);
  dblvec upper = get_upper_values(true,true,false);
  dblvec lower = get_lower_values(true,true,false);
  fhdl.os.lower_ = lower;
  fhdl.os.upper_ = upper;
  dblvec ndep;
  for(int i = 0; i < npars; i++) ndep.push_back(tol);
  fhdl.os.ndeps_ = ndep;
  dblvec hessian(npars * npars,0.0);
  fhdl.Hessian(start,hessian);
  Eigen::MatrixXd hess = Eigen::Map<Eigen::MatrixXd>(hessian.data(),npars,npars);
  return hess;
}

inline double glmmr::Model::aic(){
  int niter = u_.cols();
  int dof = P_ + covariance_.npar();
  double logl = 0;
#pragma omp parallel for reduction (+:logl)
  for(int i = 0; i < u_.cols(); i++){
    logl += covariance_.log_likelihood(u_.col(i));
  }
  double ll = log_likelihood();
  
  return (-2*( ll + logl ) + 2*dof); 
}

#endif
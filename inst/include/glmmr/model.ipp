#ifndef MODEL_IPP
#define MODEL_IPP


inline void glmmr::Model::set_offset(const VectorXd& offset){
  if(offset.size()!=n_)Rcpp::stop("offset wrong length");
    offset_ = offset;
}

inline void glmmr::Model::update_beta(const VectorXd &beta){
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

inline void glmmr::Model::update_theta(const VectorXd &theta){
  if(theta.size()!=covariance_.npar())Rcpp::stop("theta wrong length");
  covariance_.update_parameters(theta.array());
  ZL_ = covariance_.ZL_sparse();
  zu_ = ZL_*u_;
}

inline void glmmr::Model::update_theta(const dblvec &theta){
  if(theta.size()!=covariance_.npar())Rcpp::stop("theta wrong length");
  covariance_.update_parameters(theta);
  ZL_ = covariance_.ZL_sparse();
  zu_ = ZL_*u_;
}

inline void glmmr::Model::update_theta_extern(const dblvec &theta){
  if(theta.size()!=covariance_.npar())Rcpp::stop("theta wrong length");
  covariance_.update_parameters(theta);
  ZL_ = covariance_.ZL_sparse();
  zu_ = ZL_*u_;
}

inline void glmmr::Model::update_u(const MatrixXd &u){
  if(u.rows()!=Q_)Rcpp::stop("u has wrong number of random effects");
    if(u.cols()!=u_.cols()){
      Rcpp::Rcout << "\nDifferent numbers of random effect samples";
      u_.resize(Q_,u.cols());
      zu_.resize(Q_,u.cols());
      size_m_array.resize(u.cols());
    }
    u_ = u;
    zu_ = ZL_*u_;
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

inline double glmmr::Model::log_prob(const VectorXd &v){
  VectorXd zu = ZL_ * v;
  VectorXd mu = xb() + zu;
  double lp1 = 0;
  double lp2 = 0;
#pragma omp parallel for reduction (+:lp1)
  for(int i = 0; i < n_; i++){
    lp1 += glmmr::maths::log_likelihood(y_(i),mu(i),var_par_,flink);
  }
#pragma omp parallel for reduction (+:lp2)
  for(int i = 0; i < v.size(); i++){
    lp2 += -0.5*v(i)*v(i); //glmmr::maths::log_likelihood(v(i),0,1,7);
  }
  //lp2 = -0.5*v.transpose()*v -0.5*v.size()*log(2*M_PI);
  return lp1+lp2-0.5*v.size()*log(2*M_PI);
}

inline VectorXd glmmr::Model::log_gradient(const VectorXd &v,
                                                  bool beta){
  size_n_array = xb();
  size_q_array.setZero();
  size_p_array.setZero();
  sparse ZLt = ZL_;
  ZLt.transpose();
  size_n_array += (ZL_*v).array();
  
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
    size_n_array += ArrayXd::Ones(n_);
    size_n_array = size_n_array.array().inverse();
    size_n_array -= ArrayXd::Ones(n_);
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
  return beta ? size_p_array.matrix() : size_q_array.matrix();
}


inline double glmmr::Model::log_likelihood() { 
  double ll = 0;
  size_n_array = xb();
  
#pragma omp parallel for reduction (+:ll)
  for(int j=0; j<zu_.cols() ; j++){
    for(int i = 0; i<n_; i++){
      ll += glmmr::maths::log_likelihood(y_(i),size_n_array(i) + zu_(i,j),var_par_,flink);
    }
  }
  return ll/zu_.cols();
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
  ArrayXd sigmas(niter);
  
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
  MatrixXd zd = linpred();
  
#pragma omp parallel for
  for(int i = 0; i < niter; ++i){
    VectorXd w = glmmr::maths::dhdmu(zd.col(i),family_,link_);
    w = (w.array().inverse()).matrix();
    w *= 1/nvar_par;
    VectorXd zdu = glmmr::maths::mod_inv_func(zd.col(i), link_);
    ArrayXd resid = (y_ - zdu);
    sigmas(i) = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
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
  update_beta(linpred_.parameters_ + bincr);
  var_par_ = sigmas.mean();
}

inline void glmmr::Model::laplace_nr_beta_u(){
  double sigmas;
  update_W();
  VectorXd zd = (linpred()).col(0);
  VectorXd dmu = glmmr::maths::detadmu(zd,link_);
  
  MatrixXd LZWZL = covariance_.LZWZL(W_);
  LZWZL = LZWZL.llt().solve(MatrixXd::Identity(LZWZL.rows(),LZWZL.cols()));
  VectorXd zdu = glmmr::maths::mod_inv_func(zd, link_);
  ArrayXd resid = (y_ - zdu).array();
  sigmas = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
  
  MatrixXd XtXW = linpred_.X().transpose() * W_.asDiagonal() * linpred_.X();
  VectorXd w = W_;
  w = w.cwiseProduct(dmu);
  w = w.cwiseProduct(resid.matrix());
  
  XtXW = XtXW.inverse();
  VectorXd bincr = XtXW * (linpred_.X()).transpose() * w;
  VectorXd vgrad = log_gradient(u_.col(0));
  VectorXd vincr = LZWZL * vgrad;
  update_u(u_.colwise()+vincr);
  update_beta(linpred_.parameters_ + bincr);
  var_par_ = sigmas;
}


inline VectorXd glmmr::Model::new_proposal(const VectorXd& u0_,
                                           bool adapt, 
                                           int iter,
                                           double runif){
  Rcpp::NumericVector z = Rcpp::rnorm(Q_);
  VectorXd r_ = Rcpp::as<Map<VectorXd> >(z);
  VectorXd grad_ = log_gradient(u0_,false);
  double lpr_ = 0.5*r_.transpose()*r_;
  VectorXd up_ = u0_;
  
  steps_ = std::max(1,(int)std::round(lambda_/e_));
  steps_ = std::min(steps_, max_steps_);
  // leapfrog integrator
  for(int i=0; i< steps_; i++){
    r_ += (e_/2)*grad_;
    up_ += e_ * r_;
    grad_ = log_gradient(up_,false);
    r_ += (e_/2)*grad_;
  }
  
  double lprt_ = 0.5*r_.transpose()*r_;
  
  double l1 = log_prob(u0_);
  double l2 = log_prob(up_);
  double prob = std::min(1.0,exp(-l1 + lpr_ + l2 - lprt_));
  bool accept = runif < prob;
  
  if(trace_==2){
    int printSize = u0_.size() < 10 ? u0_.size() : 10;
    Rcpp::Rcout << "\nIter: " << iter << " l1 " << l1 << " h1 " << lpr_ << " l2 " << l2 << " h2 " << lprt_;
    Rcpp::Rcout << "\nCurrent value: " << u0_.transpose().head(printSize);
    Rcpp::Rcout << "\nvelocity: " << r_.transpose().head(printSize);
    Rcpp::Rcout << "\nProposal: " << up_.transpose().head(printSize);
    Rcpp::Rcout << "\nAccept prob: " << prob << " step size: " << e_ << " mean: " << ebar_ << " steps: " << steps_;
    if(accept){
      Rcpp::Rcout << " ACCEPT \n";
    } else {
      Rcpp::Rcout << " REJECT \n";
    }
  }
  
  if(adapt){
    double f1 = 1.0/(iter + 10);
    H_ = (1-f1)*H_ + f1*(target_accept_ - prob);
    double loge = -4.60517 - (sqrt((double)iter / 0.05))*H_;
    double powm = std::pow(iter,-0.75);
    double logbare = powm*loge + (1-powm)*log(ebar_);
    e_ = exp(loge);
    ebar_ = exp(logbare);
  } else {
    e_ = ebar_;
  }
  
  if(accept){
    accept_++;
    return up_;
  } else {
    return u0_;
  }
  
}


inline void glmmr::Model::sample(int warmup,
                                     int nsamp,
                                     int adapt){
  // e_ = 0.001;
  // ebar_ = 1.0;
  // H_ = 0;
  Rcpp::NumericVector z = Rcpp::rnorm(Q_);
  VectorXd unew_ = Rcpp::as<Map<VectorXd> >(z);
  accept_ = 0;
  std::minstd_rand gen_(std::random_device{}());
  std::uniform_real_distribution<double> dist_(0.0, 1.0);
  if(nsamp!=u_.cols())u_.resize(Q_,nsamp);
  u_.setZero();
  int totalsamps = nsamp + warmup;
  int i;
  double prob;
  prob = dist_(gen_);
  
  // warmups
  for(i = 0; i < warmup; i++){
    prob = dist_(gen_);
    if(i < adapt){
      unew_ = new_proposal(unew_,true,i+1,prob);
    } else {
      unew_ = new_proposal(unew_,false,i+1,prob);
    }
    if(verbose_ && i%refresh_== 0){
      Rcpp::Rcout << "\nWarmup: Iter " << i << " of " << totalsamps;
    }
  }
  u_.col(0) = unew_;
  int iter = 1;
  //sampling
  for(i = 0; i < nsamp-1; i++){
    prob = dist_(gen_);
    u_.col(i+1) = new_proposal(u_.col(i),false,i+1,prob);
    if(verbose_ && i%refresh_== 0){
      Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
    }
  }
  if(trace_>0)Rcpp::Rcout << "\nAccept rate: " << (double)accept_/(warmup+nsamp) << " steps: " << steps_ << " step size: " << e_;
  if(verbose_)Rcpp::Rcout << "\n" << std::string(40, '-');
  // return samples_.matrix();//.array();//remove L
}

inline MatrixXd glmmr::Model::laplace_hessian(double tol){
  LA_likelihood_btheta hdl(*this);
  int nvar = P_ + covariance_.npar();
  if(family_=="gaussian"||family_=="Gamma"||family_=="beta")nvar++;
  dblvec ndep(nvar,tol);
  hdl.os.ndeps_ = ndep;
  dblvec hessian(nvar * nvar,0.0);
  dblvec start = get_start_values(true,true,false);
  hdl.Hessian(start,hessian);
  MatrixXd hess = Map<MatrixXd>(hessian.data(),nvar,nvar);
  return hess;
}

inline MatrixXd glmmr::Model::hessian(double tol){
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
  MatrixXd hess = Map<MatrixXd>(hessian.data(),npars,npars);
  return hess;
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

#endif
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

inline void glmmr::Model::update_W(){
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
    size_n_array = glmmr::maths::attenuted_xb(xb(),covariance_.Z(),covariance_.D(),link_);
  } else {
    size_n_array = xb();
  }
  W_ = glmmr::maths::dhdmu(size_n_array,family_,link_);
  W_.noalias() = (W_.array().inverse()).matrix();
  W_ *= 1/nvar_par;
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
    lp2 += -0.5*v(i)*v(i); 
  }
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
  
  return beta ? size_p_array.matrix() : size_q_array.matrix();
}

inline MatrixXd glmmr::Model::Zu(){
    return zu_;
}
  
inline MatrixXd glmmr::Model::Sigma(bool inverse){
    update_W();
    MatrixXd S = sigma_builder(0,inverse);
    return S;
}
  
inline MatrixXd glmmr::Model::information_matrix(){
    update_W();
    MatrixXd M = MatrixXd::Zero(P_,P_);
    for(int i = 0; i< sigma_blocks_.size(); i++){
      M += information_matrix_by_block(i);
    }
    return M;
}

inline MatrixXd glmmr::Model::sigma_block(int b,
                                          bool inverse){
  if(b >= sigma_blocks_.size())Rcpp::stop("Index out of range");
  sparse ZLs = submat_sparse(covariance_.ZL_sparse(),sigma_blocks_[b].RowIndexes);
  MatrixXd ZL = sparse_to_dense(ZLs,false);
  MatrixXd S = ZL * ZL.transpose();
  for(int i = 0; i < S.rows(); i++){
    S(i,i)+= 1/W_(sigma_blocks_[b].RowIndexes[i]);
  }
  if(inverse){
    S = S.llt().solve(MatrixXd::Identity(S.rows(),S.cols()));
  }
  return S;
}

inline MatrixXd glmmr::Model::sigma_builder(int b,
                                             bool inverse){
  int B_ = sigma_blocks_.size();
  if (b == B_ - 1) {
    return sigma_block(b,inverse);
  }
  else {
    MatrixXd mat1 = sigma_block(b,inverse);
    MatrixXd mat2;
    if (b == B_ - 2) {
      mat2 = sigma_block(b+1,inverse);
    }
    else {
      mat2 = sigma_builder(b + 1,  inverse);
    }
    int n1 = mat1.rows();
    int n2 = mat2.rows();
    MatrixXd dmat = MatrixXd::Zero(n1+n2, n1+n2);
    dmat.block(0,0,n1,n1) = mat1;
    dmat.block(n1, n1, n2, n2) = mat2;
    return dmat;
  }
}

inline MatrixXd glmmr::Model::information_matrix_by_block(int b){
  ArrayXi rows = Map<ArrayXi,Unaligned>(sigma_blocks_[b].RowIndexes.data(),sigma_blocks_[b].RowIndexes.size());
  MatrixXd X = glmmr::Eigen_ext::submat(linpred_.X(),rows,ArrayXi::LinSpaced(P_,0,P_-1));
  MatrixXd S = sigma_block(b,true);
  MatrixXd M = X.transpose()*S*X;
  return M;
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

inline double glmmr::Model::full_log_likelihood(){
  double ll = log_likelihood();
  double logl = 0;
  MatrixXd Lu = covariance_.Lu(u_);
#pragma omp parallel for reduction (+:logl)
  for(int i = 0; i < Lu.cols(); i++){
    logl += covariance_.log_likelihood(Lu.col(i));
  }
  logl *= 1/Lu.cols();
  return ll+logl;
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
  MatrixXd Lu = covariance_.Lu(u_);
  double denomD = 0;
  for(int i = 0; i < Lu.cols(); i++){
      denomD += covariance_.log_likelihood(Lu.col(i));
  }
  denomD *= 1/Lu.cols();
  F_likelihood dl(*this,denomD,true);
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
    w *= nvar_par;
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
  VectorXd dmu =  glmmr::maths::detadmu(zd,link_);
  
  MatrixXd LZWZL = covariance_.LZWZL(W_);
  LZWZL = LZWZL.llt().solve(MatrixXd::Identity(LZWZL.rows(),LZWZL.cols()));
  VectorXd zdu =  glmmr::maths::mod_inv_func(zd, link_);
  ArrayXd resid = (y_ - zdu).array();
  sigmas = std::sqrt((resid - resid.mean()).square().sum()/(resid.size()-1));
  
  MatrixXd XtXW = (linpred_.X()).transpose() * W_.asDiagonal() * linpred_.X();
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
                                     linpred_.colnames_,
                                     linpred_.parameters_.array());
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

  
inline VectorXd glmmr::Model::predict_xb(const ArrayXXd& newdata_,
                      const ArrayXd& newoffset_){
    glmmr::LinearPredictor newlinpred_(formula_,
                                       newdata_,
                                       linpred_.colnames_,
                                       linpred_.parameters_.array());
    VectorXd xb = newlinpred_.xb() + newoffset_.matrix();
    return xb;
  }
  
inline void glmmr::Model::mcmc_sample(int warmup,
                   int samples,
                   int adapt){
    sample(warmup,samples,adapt);
    if(u_.cols()!=zu_.cols())zu_.resize(Q_,u_.cols());
    zu_ = ZL_*u_;
  }
  
inline void glmmr::Model::mcmc_set_lambda(double lambda){
    lambda_ = lambda;
  }
  
inline void glmmr::Model::mcmc_set_max_steps(int max_steps){
    max_steps_ = max_steps;
  }
  
inline void glmmr::Model::mcmc_set_refresh(int refresh){
    refresh_ = refresh;
  }
  
inline void glmmr::Model::mcmc_set_target_accept(double target){
    target_accept_ = target;
  }
  
inline void glmmr::Model::set_trace(int trace){
    trace_ = trace;
  }
  
inline void glmmr::Model::make_covariance_sparse(){
    covariance_.set_sparse(true);
  }
  
inline void glmmr::Model::make_covariance_dense(){
    covariance_.set_sparse(false);
  }
  
inline void glmmr::Model::gen_sigma_blocks(){
  int block_counter = 0;
  intvec2d block_ids(n_);
  int block_size;
  sparse Z = covariance_.Z_sparse();
  int i,j,k;
  auto it_begin = Z.Ai.begin();
  for(int b = 0; b < covariance_.B(); b++){
    block_size = covariance_.block_dim(b);
    for(i = 0; i < block_size; i++){
#pragma omp parallel for shared(it_begin, i)
      for(j = 0; j < n_; j++){
        auto it = std::find(it_begin + Z.Ap[j], it_begin + Z.Ap[j+1], (i+block_counter));
        if(it != (it_begin + Z.Ap[j+1])){
          block_ids[j].push_back(b);
        }
      }
    }
    block_counter += block_size;
  }
  
  block_counter = 0;
  intvec idx_matches;
  int n_matches;
  for(i = 0; i < n_; i++){
    if(block_counter == 0){
      glmmr::SigmaBlock newblock(block_ids[i]);
      newblock.add_row(0);
      sigma_blocks_.push_back(newblock);
    } else {
      for(j = 0; j < block_counter; j++){
        if(sigma_blocks_[j] == block_ids[i]){
          idx_matches.push_back(j);
        }
      }
      n_matches = idx_matches.size();
      if(n_matches==0){
        glmmr::SigmaBlock newblock(block_ids[i]);
        newblock.add_row(i);
        sigma_blocks_.push_back(newblock);
      } else if(n_matches==1){
        sigma_blocks_[idx_matches[0]].add(block_ids[i]);
        sigma_blocks_[idx_matches[0]].add_row(i);
      } else if(n_matches>1){
        std::reverse(idx_matches.begin(),idx_matches.end());
        for(k = 0; k < (n_matches-1); k++){
          sigma_blocks_[idx_matches[n_matches-1]].merge(sigma_blocks_[idx_matches[k]]);
          sigma_blocks_.erase(sigma_blocks_.begin()+idx_matches[k]);
        }
      }
    }
    idx_matches.clear();
    block_counter = sigma_blocks_.size();
  }
}

inline ArrayXd glmmr::Model::optimum_weights(double N, 
                                             double sigma_sq,
                                             VectorXd C,
                                             double tol,
                                             int max_iter){
  if(C.size()!=P_)Rcpp::stop("C is wrong size");
  VectorXd Cvec(C);
  ArrayXd weights = ArrayXd::Constant(n_,1.0*n_);
  VectorXd holder(n_);
  weights = weights.inverse();
  ArrayXd weightsnew(weights);
  std::vector<MatrixXd> ZDZ;
  std::vector<MatrixXd> Sigmas;
  std::vector<MatrixXd> Xs;
  std::vector<glmmr::SigmaBlock> SB(sigma_blocks_);
  Rcpp::Rcout << "\n### Preparing data ###";
  Rcpp::Rcout << "\nThere are " << SB.size() << " independent blocks and " << n_ << " cells.";
  int maxprint = n_ < 10 ? n_ : 10;
  for(int i = 0 ; i < SB.size(); i++){
    sparse ZLs = submat_sparse(covariance_.ZL_sparse(),SB[i].RowIndexes);
    MatrixXd ZL = sparse_to_dense(ZLs,false);
    MatrixXd S = ZL * ZL.transpose();
    ZDZ.push_back(S);
    Sigmas.push_back(S);
    ArrayXi rows = Map<ArrayXi,Unaligned>(SB[i].RowIndexes.data(),SB[i].RowIndexes.size());
    MatrixXd X = glmmr::Eigen_ext::submat(linpred_.X(),rows,ArrayXi::LinSpaced(P_,0,P_-1));
    Xs.push_back(X);
  }
  
  double diff = 1;
  int block_size;
  MatrixXd M(P_,P_);
  int iter = 0;
  int counter;
  Rcpp::Rcout << "\n### Starting optimisation ###";
  while(diff > tol && iter < max_iter){
    iter++;
    Rcpp::Rcout << "\nIteration " << iter << "\n------------\nweights: [" << weights.segment(0,maxprint).transpose() << " ...]";
    
    //add check to remove weights that are below a certain threshold
    if((weights < 1e-8).any()){
      for(int i = 0 ; i < SB.size(); i++){
        auto it = SB[i].RowIndexes.begin();
        while(it != SB[i].RowIndexes.end()){
          if(weights(*it) < 1e-8){
            weights(*it) = 0;
            int idx = it - SB[i].RowIndexes.begin();
            glmmr::Eigen_ext::removeRow(Xs[i],idx);
            glmmr::Eigen_ext::removeRow(ZDZ[i],idx);
            glmmr::Eigen_ext::removeColumn(ZDZ[i],idx);
            Sigmas[i].conservativeResize(ZDZ[i].rows(),ZDZ[i].cols());
            it = SB[i].RowIndexes.erase(it);
            Rcpp::Rcout << "\n Removing point " << idx << " in block " << i;
          } else {
            it++;
          }
        }
      }
    }
    
    M.setZero();
    for(int i = 0 ; i < SB.size(); i++){
      Sigmas[i] = ZDZ[i];
      for(int j = 0; j < Sigmas[i].rows(); j++){
        Sigmas[i](j,j) += sigma_sq/(N*weights(SB[i].RowIndexes[j]));
      }
      Sigmas[i] = Sigmas[i].llt().solve(MatrixXd::Identity(Sigmas[i].rows(),Sigmas[i].cols()));
      M += Xs[i].transpose() * Sigmas[i] * Xs[i];
    }
    
    //check if positive definite, if not remove the offending column(s)
    bool isspd = glmmr::Eigen_ext::issympd(M);
    if(isspd){
      Rcpp::Rcout << "\n Information matrix not postive definite: ";
      ArrayXd M_row_sums = M.rowwise().sum();
      int fake_it = 0;
      int countZero = 0;
      for(int j = 0; j < M_row_sums.size(); j++){
        if(M_row_sums(j) == 0){
          Rcpp::Rcout << "\n   Removing column " << fake_it;
          for(int k = 0; k < Xs.size(); k++){
            glmmr::Eigen_ext::removeColumn(Xs[k],fake_it);
          }
          glmmr::Eigen_ext::removeElement(Cvec,fake_it);
          countZero++;
        } else {
          fake_it++;
        }
      }
      M.conservativeResize(M.rows()-countZero,M.cols()-countZero);
      M.setZero();
      for(int k = 0; k < SB.size(); k++){
        M += Xs[k].transpose() * Sigmas[k] * Xs[k];
      }
    }
    M = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
    VectorXd Mc = M*Cvec;
    counter = 0;
    weightsnew.setZero();
    for(int i = 0 ; i < SB.size(); i++){
      block_size = SB[i].RowIndexes.size();
      holder.segment(0,block_size) = Sigmas[i] * Xs[i] * Mc;
      for(int j = 0; j < block_size; j++){
        weightsnew(SB[i].RowIndexes[j]) = holder(j);
      }
    }
    weightsnew = weightsnew.abs();
    weightsnew *= 1/weightsnew.sum();
    diff = ((weights-weightsnew).abs()).maxCoeff();
    weights = weightsnew;
    Rcpp::Rcout << "\n(Max. diff: " << diff << ")\n";
  }
  if(iter<max_iter){
    Rcpp::Rcout << "\n### CONVERGED Final weights: [" << weights.segment(0,maxprint).transpose() << "...]";
  } else {
    Rcpp::Rcout << "\n### NOT CONVERGED Reached maximum iterations";
  }
  return weights;
}

#endif
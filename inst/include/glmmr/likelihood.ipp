#ifndef LIKELIHOOD_IPP
#define LIKELIHOOD_IPP

inline double glmmr::Model::D_likelihood::operator()(const dblvec &par) {
  M_.update_theta(par);
  logl = 0;
#pragma omp parallel for reduction (+:logl)
  for(int i = 0; i < M_.u_.cols(); i++){
    logl += M_.covariance_.log_likelihood(Lu_.col(i));
  }
  return -1*logl/Lu_.cols();
}



inline double glmmr::Model::L_likelihood::operator()(const dblvec &par) {
  if(M_.family_=="gaussian" || M_.family_=="Gamma" || M_.family_=="beta"){
    auto first = par.begin();
    auto last = par.begin() + M_.P_;
    dblvec par2(first, last);
    M_.update_beta(par2);
    M_.var_par_ = par[M_.P_];
  } else {
    M_.update_beta(par);
  }
  ll = M_.log_likelihood();
  return -1*ll;
}

inline double glmmr::Model::F_likelihood::operator()(const dblvec &par) {
  auto first = par.begin();
  auto last1 = par.begin() + M_.P_;
  auto last2 = par.begin() + M_.P_ + G;
  dblvec beta(first,last1);
  dblvec theta(last1,last2);
  M_.update_beta(beta);
  M_.update_theta(theta);
  if(M_.family_=="gaussian" || M_.family_=="Gamma" || M_.family_=="beta")M_.var_par_ = par[M_.P_+G];
  ll = M_.log_likelihood();
  logl = 0;
#pragma omp parallel for reduction (+:logl)
  for(int i = 0; i < M_.u_.cols(); i++){
    logl += M_.covariance_.log_likelihood(M_.u_.col(i));
  }
  logl *= 1/M_.u_.cols();
  if(importance_){
    return -1.0 * log(exp(ll + logl)/ exp(denomD));
  } else {
    return -1.0*(ll + logl);
  }
}

inline double glmmr::Model::LA_likelihood::operator()(const dblvec &par) {
  logl = 0;
  auto start = par.begin();
  auto end = par.begin()+M_.P_;
  dblvec beta(start,end);
  for(int i = 0; i<M_.Q_; i++)v(i,0) = par[M_.P_ + i];
  M_.update_beta(beta);
  M_.update_u(v);
  logl = v.col(0).transpose()*v.col(0);
  ll = M_.log_likelihood();
  if(M_.family_!="gaussian"){
    M_.update_W();
    //LZWZL = M_.ZL_.transpose() * M_.W_.asDiagonal() * M_.ZL_;
    //LZWZL.noalias() += Eigen::MatrixXd::Identity(LZWZL.rows(),LZWZL.cols());
    LZWZL = M_.covariance_.LZWZL(M_.W_);
    LZWdet = glmmr::maths::logdet(LZWZL);
  }
  return -1.0*(ll - 0.5*logl - 0.5*LZWdet);
}

inline double glmmr::Model::LA_likelihood_cov::operator()(const dblvec &par) {
  if(M_.family_=="gaussian" || M_.family_=="Gamma" || M_.family_=="beta"){
    int G = M_.covariance_.npar();
    auto start = par.begin();
    auto end = par.begin()+G;
    dblvec theta(start,end);
    M_.update_theta(theta);
    M_.var_par_ = par[G];
  } else {
    M_.update_theta(par);
  }
  logl = M_.u_.col(0).transpose() * M_.u_.col(0);
  ll = M_.log_likelihood();
  M_.update_W();
  LZWZL = M_.covariance_.LZWZL(M_.W_);
  LZWdet = glmmr::maths::logdet(LZWZL);
  
  return -1*(ll - 0.5*logl - 0.5*LZWdet);
}

inline double glmmr::Model::LA_likelihood_btheta::operator()(const dblvec &par) {
  auto start = par.begin();
  auto end1 = par.begin() +M_.P_;
  auto end2 = par.begin() + M_.P_ + M_.covariance_.npar();
  dblvec beta(start,end1);
  dblvec theta(end1,end2);
  
  if(M_.family_=="gaussian" || M_.family_=="Gamma" || M_.family_=="beta"){
    M_.var_par_ = par[par.size()-1];
  } 
  
  M_.update_beta(beta);
  M_.update_theta(theta);
  ll = M_.log_likelihood();
  logl = M_.u_.col(0).transpose() * M_.u_.col(0);
  M_.update_W();
  LZWZL = M_.covariance_.LZWZL(M_.W_);
  LZWdet = glmmr::maths::logdet(LZWZL);
  return -1*(ll - 0.5*logl - 0.5*LZWdet);
}

#endif
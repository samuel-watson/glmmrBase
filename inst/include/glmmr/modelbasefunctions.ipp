#ifndef MODELBASEFUNCTIONS_IPP
#define MODELBASEFUNCTIONS_IPP


inline void glmmr::Model::set_offset(const VectorXd& offset){
  if(offset.size()!=n_)Rcpp::stop("offset wrong length");
    offset_ = offset;
}

inline void glmmr::Model::setup_calculator(){
  dblvec yvec(y_.data(),y_.data()+y_.size());
  calc_ = linpred_.calc_;
  glmmr::linear_predictor_to_link(calc_,link_);
  glmmr::link_to_likelihood(calc_,family_);
  calc_.y = yvec;
  calc_.var_par = var_par_;
  vcalc_ = linpred_.calc_;
  glmmr::re_linear_predictor(vcalc_,Q_);
  glmmr::linear_predictor_to_link(vcalc_,link_);
  glmmr::link_to_likelihood(vcalc_,family_);
  vcalc_.y = yvec;
  vcalc_.var_par = var_par_;
}

inline void glmmr::Model::set_num_threads(int n){
  if(OMP_IS_USED){
    n_threads_ = n;
    if(n <= omp_get_max_threads()){
      omp_set_num_threads(n);
    } else {
      omp_set_num_threads(omp_get_max_threads()); 
    }
  } else {
    n_threads_ = 1;
  }
}

inline void glmmr::Model::set_num_threads(){
  if(OMP_IS_USED){
    n_threads_ = omp_get_max_threads();
  } else {
    n_threads_ = 1;
  }
}

inline void glmmr::Model::update_beta(const VectorXd &beta){
  if(beta.size()!=P_)Rcpp::stop("beta wrong length");
    linpred_.update_parameters(beta.array());
    dblvec new_parameters(beta.data(),beta.data()+beta.size());
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

inline void glmmr::Model::set_trace(int trace){
  trace_ = trace;
}

inline void glmmr::Model::make_covariance_sparse(){
  covariance_.set_sparse(true);
}

inline void glmmr::Model::make_covariance_dense(){
  covariance_.set_sparse(false);
}

inline VectorXd glmmr::Model::predict_xb(const ArrayXXd& newdata_,
                                         const ArrayXd& newoffset_){
  glmmr::LinearPredictor newlinpred_(formula_,
                                     newdata_,
                                     linpred_.colnames(),
                                     linpred_.parameters_);
  VectorXd xb = newlinpred_.xb() + newoffset_.matrix();
  return xb;
}

inline double glmmr::Model::log_likelihood() {
  double ll = 0;
  size_n_array = xb();
#pragma omp parallel for reduction (+:ll) collapse(2) num_threads(n_threads_)
  for(int j=0; j<zu_.cols() ; j++){
    for(int i = 0; i<n_; i++){
      ll += glmmr::maths::log_likelihood(y_(i),size_n_array(i) + zu_(i,j),var_par_,flink);
    }
  }
  
  // to use the calculator object instead... seems to be generally slower so have opted 
  // for specific formulae above. Will try to optimise this in future versions
  // #pragma omp parallel for reduction (+:ll) collapse(2) num_threads(n_threads_)
  //  for(int j=0; j<zu_.cols() ; j++){
  //    for(int i = 0; i<n_; i++){
  //      double ozu = offset_(i)+zu_(i,j);
  //      ll += calc_.calculate(i,linpred_.parameters_,linpred_.Xdata_,0,0,ozu)[0];
  //    }
  //  }
  
  return ll/zu_.cols();
}

inline double glmmr::Model::full_log_likelihood(){
  double ll = log_likelihood();
  double logl = 0;
  MatrixXd Lu = covariance_.Lu(u_);
#pragma omp parallel for reduction (+:logl) num_threads(n_threads_)
  for(int i = 0; i < Lu.cols(); i++){
    logl += covariance_.log_likelihood(Lu.col(i));
  }
  logl *= 1/Lu.cols();
  return ll+logl;
}


inline dblvec glmmr::Model::get_start_values(bool beta, bool theta, bool var){
  dblvec start;
  if(beta){
    for(int i =0 ; i < P_; i++)start.push_back(linpred_.parameters_[i]);
    
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

#endif
#ifndef MODELMCMCFUNCTIONS_IPP
#define MODELMCMCFUNCTIONS_IPP

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



#endif
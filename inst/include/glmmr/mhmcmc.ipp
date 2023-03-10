#ifndef MHMCMC_IPP
#define MHMCMC_IPP

inline void glmmr::Model::HMC::initialise_u(){
  Rcpp::NumericVector z = Rcpp::rnorm(M_.Q_);
  u0_ = Rcpp::as<Eigen::Map<Eigen::VectorXd> >(z);
  z = Rcpp::rnorm(u0_.size());
  r_ = Rcpp::as<Eigen::Map<Eigen::VectorXd> >(z);
  up_ = u0_;
  accept_ = 0;
  H_ = 0;
  gen_ = std::minstd_rand(std::random_device{}());
  dist_ = std::uniform_real_distribution<double>(0.0, 1.0);
  e_ = 0.001;
  ebar_ = 1.0;
}

inline void glmmr::Model::HMC::new_proposal(bool adapt, 
                                            int iter){
  
  Rcpp::NumericVector z = Rcpp::rnorm(r_.size());
  r_ = Rcpp::as<Eigen::Map<Eigen::VectorXd> >(z);
  grad_ = M_.log_gradient(u0_);
  double lpr_ = 0.5*r_.transpose()*r_;
  up_ = u0_;

  steps_ = std::max(1,(int)std::round(lambda_/e_));
  steps_ = std::min(steps_, max_steps_);
  // leapfrog integrator
  for(int i=0; i< steps_; i++){
    r_ += (e_/2)*grad_;
    up_ += e_ * r_;
    grad_ = M_.log_gradient(up_);
    r_ += (e_/2)*grad_;
  }

  double lprt_ = 0.5*r_.transpose()*r_;

  double l1 = M_.log_prob(u0_);
  double l2 = M_.log_prob(up_);
  double prob = std::min(1.0,exp(-l1 + lpr_ + l2 - lprt_));
  double runif = dist_(gen_);
  bool accept = runif < prob;

  if(M_.trace_==2){
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


  if(accept){
    u0_ = up_;
    accept_++;
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
  
}


inline Eigen::ArrayXXd glmmr::Model::HMC::sample(int warmup,
                                                 int nsamp,
                                                 int adapt){
  int totalsamps = nsamp + warmup;
  int Q = M_.Q_;
  Eigen::MatrixXd samples(Q,nsamp+1);
  initialise_u();
  int i;
  if(verbose_)Rcpp::Rcout << "\nMCMC Sampling";
  samples.setZero();
  // warmups
  for(i = 0; i < warmup; i++){
    if(i < adapt){
      new_proposal(true,i+1);
    } else {
      new_proposal(false);
    }
    if(verbose_ && i%refresh_== 0){
      Rcpp::Rcout << "\nWarmup: Iter " << i << " of " << totalsamps;
    }
  }

  samples.col(0) = u0_;
  int iter = 1;
  //sampling
  for(i = 0; i < nsamp; i++){
    new_proposal(false);
    samples.col(i+1) = u0_;
    if(verbose_ && i%refresh_== 0){
      Rcpp::Rcout << "\nSampling: Iter " << i + warmup << " of " << totalsamps;
    }
  }
  if(M_.trace_>0)Rcpp::Rcout << "\nAccept rate: " << (double)accept_/(warmup+nsamp) << " steps: " << steps_ << " step size: " << e_;
  //return samples;
  if(verbose_)Rcpp::Rcout << "\n" << std::string(40, '-');
  return (M_.L() * samples).array();
}

#endif
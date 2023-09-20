#include <glmmr.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP Model__get_W(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  VectorXd W = ptr->matrix.W.W();
  return wrap(W);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  VectorXd W = ptr->matrix.W.W();
  return wrap(W);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__log_prob(SEXP xp, SEXP v_, int type = 0){
  Eigen::VectorXd v = as<Eigen::VectorXd>(v_);
  double logprob;
  switch(type){
  case 0:
  {
    XPtr<glmm> ptr(xp);
    logprob = ptr->mcmc.log_prob(v);
    break;
  }
  case 1:
  {
    XPtr<glmm_nngp> ptr(xp);
    logprob = ptr->mcmc.log_prob(v);
    break;
  }
  }
  return wrap(logprob);
}

// [[Rcpp::export]]
SEXP Model__log_gradient(SEXP xp, SEXP v_, SEXP beta_, int type = 0){
  Eigen::VectorXd v = as<Eigen::VectorXd>(v_);
  bool beta = as<bool>(beta_);
  switch(type){
  case 0:
  {
    XPtr<glmm> ptr(xp);
    Eigen::VectorXd loggrad = ptr->matrix.log_gradient(v,beta);
    return wrap(loggrad);
    break;
  }
  case 1:
  {
    XPtr<glmm_nngp> ptr(xp);
    Eigen::VectorXd loggrad = ptr->matrix.log_gradient(v,beta);
    return wrap(loggrad);
    break;
  }
  }
}

// [[Rcpp::export]]
SEXP Model__linear_predictor(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd linpred = ptr->matrix.linpred();
  return wrap(linpred);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd linpred = ptr->matrix.linpred();
  return wrap(linpred);
  break;
}
  }
  
}

// [[Rcpp::export]]
SEXP Model__log_likelihood(SEXP xp, int type = 0){
  double logl;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  logl = ptr->optim.log_likelihood();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  logl = ptr->optim.log_likelihood();
  break;
}
  }
  return wrap(logl);
}

// [[Rcpp::export]]
void Model__ml_theta(SEXP xp, int type = 0){
  switch(type){
  case 0:
    {
      XPtr<glmm> ptr(xp);
      ptr->optim.ml_theta();
      break;
    }
  case 1:
    {
      XPtr<glmm_nngp> ptr(xp);
      ptr->optim.ml_theta();
      break;
    }
  }
}

// [[Rcpp::export]]
void Model__cov_set_nn(SEXP xp, int nn){
  XPtr<glmm_nngp> ptr(xp);
  ptr->model.covariance.gen_NN(nn);
}

// [[Rcpp::export]]
void Model__ml_beta(SEXP xp, int type = 0){
  switch(type){
  case 0:
    {
      XPtr<glmm> ptr(xp);
      ptr->optim.ml_beta();
      break;
    }
  case 1:
    {
      XPtr<glmm_nngp> ptr(xp);
      ptr->optim.ml_beta();
      break;
    }
  }
}

// [[Rcpp::export]]
void Model__ml_all(SEXP xp, int type = 0){
  switch(type){
  case 0:
    {
      XPtr<glmm> ptr(xp);
      ptr->optim.ml_all();
      break;
    }
  case 1:
    {
      XPtr<glmm_nngp> ptr(xp);
      ptr->optim.ml_all();
      break;
    }
  }
}

// [[Rcpp::export]]
void Model__laplace_ml_beta_u(SEXP xp, int type = 0){
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->optim.laplace_ml_beta_u();
        break;
      }
    case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        ptr->optim.laplace_ml_beta_u();
        break;
      }
  }
}

// [[Rcpp::export]]
void Model__laplace_ml_theta(SEXP xp, int type = 0){
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->optim.laplace_ml_theta();
        break;
      }
    case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        ptr->optim.laplace_ml_theta();
        break;
      }
  }
}

// [[Rcpp::export]]
void Model__laplace_ml_beta_theta(SEXP xp, int type = 0){
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->optim.laplace_ml_beta_theta();
        break;
      }
    case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        ptr->optim.laplace_ml_beta_theta();
        break;
      }
  }
}

// [[Rcpp::export]]
void Model__nr_beta(SEXP xp, int type = 0){
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->optim.nr_beta();
        break;
      }
    case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        ptr->optim.nr_beta();
        break;
      }
  }
}

// [[Rcpp::export]]
void Model__laplace_nr_beta_u(SEXP xp, int type = 0){
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->optim.laplace_nr_beta_u();
        break;
      }
    case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        ptr->optim.laplace_nr_beta_u();
        break;
      }
  }
}

// [[Rcpp::export]]
SEXP Model__Sigma(SEXP xp, bool inverse, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd S = ptr->matrix.Sigma(inverse);
  return wrap(S);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd S = ptr->matrix.Sigma(inverse);
  return wrap(S);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__information_matrix(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd M = ptr->matrix.information_matrix();
  return wrap(M);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd M = ptr->matrix.information_matrix();
  return wrap(M);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__obs_information_matrix(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd infomat = ptr->matrix.observed_information_matrix();
  return wrap(infomat);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd infomat = ptr->matrix.observed_information_matrix();
  return wrap(infomat);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__u(SEXP xp, bool scaled_, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd u = ptr->re.u(scaled_);
  return wrap(u);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd u = ptr->re.u(scaled_);
  return wrap(u);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__Zu(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd Zu = ptr->re.Zu();
  return wrap(Zu);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd Zu = ptr->re.Zu();
  return wrap(Zu);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__X(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd X = ptr->model.linear_predictor.X();
  return wrap(X);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd X = ptr->model.linear_predictor.X();
  return wrap(X);
  break;
}
  }
}

// [[Rcpp::export]]
void Model__mcmc_sample(SEXP xp, SEXP warmup_, SEXP samples_, SEXP adapt_, int type = 0){
  int warmup = as<int>(warmup_);
  int samples = as<int>(samples_);
  int adapt = as<int>(adapt_);
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->mcmc.mcmc_sample(warmup,samples,adapt);
        break;
      }
    case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        ptr->mcmc.mcmc_sample(warmup,samples,adapt);
        break;
      }
  }
}

// [[Rcpp::export]]
void Model__set_trace(SEXP xp, SEXP trace_, int type = 0){
  int trace = as<int>(trace_);
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->set_trace(trace);
        break;
      }
    case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        ptr->set_trace(trace);
        break;
      }
  }
}

// [[Rcpp::export]]
SEXP Model__get_beta(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd beta = ptr->model.linear_predictor.parameter_vector();
  return wrap(beta);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::VectorXd beta = ptr->model.linear_predictor.parameter_vector();
  return wrap(beta);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__y(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd y = ptr->model.data.y;
  return wrap(y);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::VectorXd y = ptr->model.data.y;
  return wrap(y);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__get_theta(SEXP xp, int type = 0){
  std::vector<double> theta;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  theta = ptr->model.covariance.parameters_;
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  theta = ptr->model.covariance.parameters_;
  break;
}
  }
  return wrap(theta);
}

// [[Rcpp::export]]
SEXP Model__get_var_par(SEXP xp, int type = 0){
  double theta;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  theta = ptr->model.data.var_par;
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  theta = ptr->model.data.var_par;
  break;
}
  }
  return wrap(theta);
}

// [[Rcpp::export]]
SEXP Model__get_variance(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::ArrayXd theta = ptr->model.data.variance;
  return wrap(theta);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::ArrayXd theta = ptr->model.data.variance;
  return wrap(theta);
  break;
}
  }
}

// [[Rcpp::export]]
void Model__set_var_par(SEXP xp, SEXP var_par_, int type = 0){
  double var_par = as<double>(var_par_);
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->model.data.set_var_par(var_par);
        break;
      }
  case 1:
    {
      XPtr<glmm_nngp> ptr(xp);
      ptr->model.data.set_var_par(var_par);
      break;
    }
  }
}

// [[Rcpp::export]]
void Model__set_trials(SEXP xp, SEXP trials, int type = 0){
  Eigen::ArrayXd var_par = as<Eigen::ArrayXd>(trials);
  switch(type){
  case 0:
  {
    XPtr<glmm> ptr(xp);
    if(ptr->model.family.family != "binomial")Rcpp::stop("trials can only be set for binomial family.");
    if(var_par.size()!=ptr->model.n())Rcpp::stop("trials wrong length");
    ptr->model.data.set_variance(var_par);
    break;
  }
  case 1:
  {
    XPtr<glmm_nngp> ptr(xp);
    if(ptr->model.family.family != "binomial")Rcpp::stop("trials can only be set for binomial family.");
    if(var_par.size()!=ptr->model.n())Rcpp::stop("trials wrong length");
    ptr->model.data.set_variance(var_par);
    break;
  }
  }
}

// [[Rcpp::export]]
SEXP Model__L(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd L = ptr->model.covariance.D(true,false);
  return wrap(L);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd L = ptr->model.covariance.D(true,false);
  return wrap(L);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__ZL(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd ZL = ptr->model.covariance.ZL();
  return wrap(ZL);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd ZL = ptr->model.covariance.ZL();
  return wrap(ZL);
  break;
}
  }  
}

// [[Rcpp::export]]
SEXP Model__xb(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd xb = ptr->model.xb();
  return wrap(xb);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::VectorXd xb = ptr->model.xb();
  return wrap(xb);
  break;
}
  }  
}
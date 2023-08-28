#include <glmmr.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP ModelBits__new(SEXP formula_, SEXP data_, SEXP colnames_,
                    SEXP family_, SEXP link_, SEXP beta_,
                    SEXP theta_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  std::vector<double> beta = as<std::vector<double> >(beta_);
  std::vector<double> theta = as<std::vector<double> >(theta_);
  XPtr<bits>ptr(new bits(formula,data,colnames,family,link),true);
  ptr->linear_predictor.update_parameters(beta);
  ptr->covariance.update_parameters(theta);
  return ptr;
}

// [[Rcpp::export]]
void ModelBits__update_beta(SEXP xp, SEXP beta_){
  std::vector<double> beta = as<std::vector<double> >(beta_);
  XPtr<bits> ptr(xp);
  ptr->linear_predictor.update_parameters(beta);
}

// [[Rcpp::export]]
void ModelBits__update_theta(SEXP xp, SEXP theta_){
  std::vector<double> theta = as<std::vector<double> >(theta_);
  XPtr<bits> ptr(xp);
  ptr->covariance.update_parameters(theta);
}

// [[Rcpp::export]]
SEXP Model__new_from_bits(SEXP bptr_){
  XPtr<bits> bptr(bptr_);
  XPtr<glmm> ptr(new glmm(*bptr),true);
  return ptr;
}

// [[Rcpp::export]]
void Model__set_y(SEXP xp, SEXP y_){
  Eigen::VectorXd y = as<Eigen::VectorXd>(y_);
  XPtr<glmm> ptr(xp);
  ptr->set_y(y);
}

// [[Rcpp::export]]
void Model__set_offset(SEXP xp, SEXP offset_){
  Eigen::VectorXd offset = as<Eigen::VectorXd>(offset_);
  XPtr<glmm> ptr(xp);
  ptr->set_offset(offset);
}

// [[Rcpp::export]]
void Model__set_weights(SEXP xp, SEXP weights_){
  Eigen::ArrayXd weights = as<Eigen::ArrayXd>(weights_);
  XPtr<glmm> ptr(xp);
  ptr->set_weights(weights);
}

// [[Rcpp::export]]
void Model__update_beta(SEXP xp, SEXP beta_){
  std::vector<double> beta = as<std::vector<double> >(beta_);
  XPtr<glmm> ptr(xp);
  ptr->update_beta(beta);
}

// [[Rcpp::export]]
void Model__update_theta(SEXP xp, SEXP theta_){
  std::vector<double> theta = as<std::vector<double> >(theta_);
  XPtr<glmm> ptr(xp);
  ptr->update_theta(theta);
}

// [[Rcpp::export]]
void Model__update_u(SEXP xp, SEXP u_){
  Eigen::MatrixXd u = as<Eigen::MatrixXd>(u_);
  XPtr<glmm> ptr(xp);
  ptr->update_u(u);
}

// [[Rcpp::export]]
void Model__use_attenuation(SEXP xp, SEXP use_){
  bool use = as<bool>(use_);
  XPtr<glmm> ptr(xp);
  ptr->matrix.W.attenuated = use;
}

// [[Rcpp::export]]
void Model__update_W(SEXP xp){
  XPtr<glmm> ptr(xp);
  ptr->matrix.W.update();
}

// [[Rcpp::export]]
SEXP Model__get_W(SEXP xp){
  XPtr<glmm> ptr(xp);
  VectorXd W = ptr->matrix.W.W();
  return wrap(W);
}

// [[Rcpp::export]]
SEXP Model__log_prob(SEXP xp, SEXP v_){
  Eigen::VectorXd v = as<Eigen::VectorXd>(v_);
  XPtr<glmm> ptr(xp);
  double logprob = ptr->mcmc.log_prob(v);
  return wrap(logprob);
}

// [[Rcpp::export]]
SEXP Model__log_gradient(SEXP xp, SEXP v_, SEXP beta_){
  Eigen::VectorXd v = as<Eigen::VectorXd>(v_);
  bool beta = as<bool>(beta_);
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd loggrad = ptr->matrix.log_gradient(v,beta);
  return wrap(loggrad);
}

// [[Rcpp::export]]
SEXP Model__linear_predictor(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd linpred = ptr->matrix.linpred();
  return wrap(linpred);
}

// [[Rcpp::export]]
SEXP Model__log_likelihood(SEXP xp){
  XPtr<glmm> ptr(xp);
  double logl = ptr->optim.log_likelihood();
  return wrap(logl);
}

// [[Rcpp::export]]
void Model__ml_theta(SEXP xp){
  XPtr<glmm> ptr(xp);
  ptr->optim.ml_theta();
}

// [[Rcpp::export]]
void Model__ml_beta(SEXP xp){
  XPtr<glmm> ptr(xp);
  ptr->optim.ml_beta();
}

// [[Rcpp::export]]
void Model__ml_all(SEXP xp){
  XPtr<glmm> ptr(xp);
  ptr->optim.ml_all();
}

// [[Rcpp::export]]
void Model__laplace_ml_beta_u(SEXP xp){
  XPtr<glmm> ptr(xp);
  ptr->optim.laplace_ml_beta_u();
}

// [[Rcpp::export]]
void Model__laplace_ml_theta(SEXP xp){
  XPtr<glmm> ptr(xp);
  ptr->optim.laplace_ml_theta();
}

// [[Rcpp::export]]
void Model__laplace_ml_beta_theta(SEXP xp){
  XPtr<glmm> ptr(xp);
  ptr->optim.laplace_ml_beta_theta();
}

// [[Rcpp::export]]
void Model__nr_beta(SEXP xp){
  XPtr<glmm> ptr(xp);
  ptr->optim.nr_beta();
}

// [[Rcpp::export]]
void Model__laplace_nr_beta_u(SEXP xp){
  XPtr<glmm> ptr(xp);
  ptr->optim.laplace_nr_beta_u();
}

// [[Rcpp::export]]
SEXP Model__Sigma(SEXP xp, bool inverse){
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd S = ptr->matrix.Sigma(inverse);
  return wrap(S);
}

// [[Rcpp::export]]
SEXP Model__information_matrix(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd M = ptr->matrix.information_matrix();
  return wrap(M);
}

// [[Rcpp::export]]
SEXP Model__obs_information_matrix(SEXP xp){
  XPtr<glmm> ptr(xp);
  MatrixXd infomat = ptr->matrix.observed_information_matrix();
  return wrap(infomat);
}

// [[Rcpp::export]]
SEXP Model__u(SEXP xp, bool scaled_){
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd u = ptr->re.u(scaled_);
  return wrap(u);
}

// [[Rcpp::export]]
SEXP Model__Zu(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd Zu = ptr->re.Zu();
  return wrap(Zu);
}

// [[Rcpp::export]]
SEXP Model__P(SEXP xp){
  XPtr<glmm> ptr(xp);
  int u = ptr->model.linear_predictor.P();
  return wrap(u);
}

// [[Rcpp::export]]
SEXP Model__Q(SEXP xp){
  XPtr<glmm> ptr(xp);
  int Q = ptr->model.covariance.Q();
  return wrap(Q);
}

// [[Rcpp::export]]
SEXP Model__X(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd X = ptr->model.linear_predictor.X();
  return wrap(X);
}

// [[Rcpp::export]]
void Model__mcmc_sample(SEXP xp, SEXP warmup_, SEXP samples_, SEXP adapt_){
  int warmup = as<int>(warmup_);
  int samples = as<int>(samples_);
  int adapt = as<int>(adapt_);
  XPtr<glmm> ptr(xp);
  ptr->mcmc.mcmc_sample(warmup,samples,adapt);
}

// [[Rcpp::export]]
void Model__set_trace(SEXP xp, SEXP trace_){
  int trace = as<int>(trace_);
  XPtr<glmm> ptr(xp);
  ptr->set_trace(trace);
}

// [[Rcpp::export]]
SEXP Model__get_beta(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd beta = ptr->model.linear_predictor.parameter_vector();
  return wrap(beta);
}

// [[Rcpp::export]]
SEXP Model__y(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd y = ptr->model.data.y;
  return wrap(y);
}

// [[Rcpp::export]]
SEXP Model__get_theta(SEXP xp){
  XPtr<glmm> ptr(xp);
  std::vector<double> theta = ptr->model.covariance.parameters_;
  return wrap(theta);
}

// [[Rcpp::export]]
SEXP Model__get_var_par(SEXP xp){
  XPtr<glmm> ptr(xp);
  double theta = ptr->model.data.var_par;
  return wrap(theta);
}

// [[Rcpp::export]]
SEXP Model__get_variance(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::ArrayXd theta = ptr->model.data.variance;
  return wrap(theta);
}

// [[Rcpp::export]]
void Model__set_var_par(SEXP xp, SEXP var_par_){
  double var_par = as<double>(var_par_);
  XPtr<glmm> ptr(xp);
  ptr->model.data.set_var_par(var_par);
}

// [[Rcpp::export]]
void Model__set_trials(SEXP xp, SEXP trials){
  Eigen::ArrayXd var_par = as<Eigen::ArrayXd>(trials);
  XPtr<glmm> ptr(xp);
  if(ptr->model.family.family != "binomial")Rcpp::stop("trials can only be set for binomial family.");
  if(var_par.size()!=ptr->model.n())Rcpp::stop("trials wrong length");
  ptr->model.data.set_variance(var_par);
}

// [[Rcpp::export]]
SEXP Model__L(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd L = ptr->model.covariance.D(true,false);
  return wrap(L);
}

// [[Rcpp::export]]
SEXP Model__ZL(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd ZL = ptr->model.covariance.ZL();
  return wrap(ZL);
}

// [[Rcpp::export]]
SEXP Model__xb(SEXP xp){
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd xb = ptr->model.xb();
  return wrap(xb);
}


#include <glmmr.h>

namespace Rcpp {
template<>
SEXP wrap(const vector_matrix& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vec") = Rcpp::wrap(x.vec),
      Rcpp::Named("mat") = Rcpp::wrap(x.mat)
  ));
}
}

using namespace Rcpp;

// [[Rcpp::export(.Model__new)]]
SEXP Model__new(SEXP y_,SEXP formula_, SEXP data_, SEXP colnames_,
                SEXP family_, SEXP link_){
  Eigen::VectorXd y = as<Eigen::VectorXd>(y_);
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  XPtr<glmmr::Model> ptr(new glmmr::Model(y,formula,data,colnames,family,link),true);
  return ptr;
}

// [[Rcpp::export(.Model__set_offset)]]
void Model__set_offset(SEXP xp, SEXP offset_){
  Eigen::VectorXd offset = as<Eigen::VectorXd>(offset_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->set_offset(offset);
}

// [[Rcpp::export(.Model__update_beta)]]
void Model__update_beta(SEXP xp, SEXP beta_){
  std::vector<double> beta = as<std::vector<double> >(beta_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->update_beta_extern(beta);
}

// [[Rcpp::export(.Model__update_theta)]]
void Model__update_theta(SEXP xp, SEXP theta_){
  std::vector<double> theta = as<std::vector<double> >(theta_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->update_theta_extern(theta);
}

// [[Rcpp::export(.Model__update_u)]]
void Model__update_u(SEXP xp, SEXP u_){
  Eigen::MatrixXd u = as<Eigen::MatrixXd>(u_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->update_u(u);
}

// [[Rcpp::export(.Model__predict)]]
SEXP Model__predict(SEXP xp, SEXP newdata_,
                          SEXP newoffset_,
                          int m){
  Eigen::ArrayXXd newdata = Rcpp::as<Eigen::ArrayXXd>(newdata_);
  Eigen::ArrayXd newoffset = Rcpp::as<Eigen::ArrayXd>(newoffset_);
  XPtr<glmmr::Model> ptr(xp);
  vector_matrix res = ptr->predict_re(newdata,newoffset);
  Eigen::MatrixXd samps(newdata.rows(),m>0 ? m : 1);
  if(m>0){
    samps = glmmr::maths::sample_MVN(res,m);
  } else {
    samps.setZero();
  }
  Eigen::VectorXd xb = ptr->predict_xb(newdata,newoffset);
  return Rcpp::List::create(
    Rcpp::Named("linear_predictor") = wrap(xb),
    Rcpp::Named("re_parameters") = wrap(res),
    Rcpp::Named("samples") = wrap(samps)
  );
}

// [[Rcpp::export(.Model__use_attenuation)]]
void Model__use_attenuation(SEXP xp, SEXP use_){
  bool use = as<bool>(use_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->attenuate_ = use;
}

// [[Rcpp::export(.Model__update_W)]]
void Model__update_W(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->update_W();
}

// [[Rcpp::export(.Model__log_prob)]]
SEXP Model__log_prob(SEXP xp, SEXP v_){
  Eigen::VectorXd v = as<Eigen::VectorXd>(v_);
  XPtr<glmmr::Model> ptr(xp);
  double logprob = ptr->log_prob(v);
  return wrap(logprob);
}

// [[Rcpp::export(.Model__log_gradient)]]
SEXP Model__log_gradient(SEXP xp, SEXP v_, SEXP beta_){
  Eigen::VectorXd v = as<Eigen::VectorXd>(v_);
  bool beta = as<bool>(beta_);
  XPtr<glmmr::Model> ptr(xp);
  Eigen::VectorXd loggrad = ptr->log_gradient(v,beta);
  return wrap(loggrad);
}

// [[Rcpp::export(.Model__linear_predictor)]]
SEXP Model__linear_predictor(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd linpred = ptr->linpred();
  return wrap(linpred);
}

// [[Rcpp::export(.Model__log_likelihood)]]
SEXP Model__log_likelihood(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  double logl = ptr->log_likelihood();
  return wrap(logl);
}

// [[Rcpp::export(.Model__ml_theta)]]
void Model__ml_theta(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->ml_theta();
}

// [[Rcpp::export(.Model__ml_beta)]]
void Model__ml_beta(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->ml_beta();
}

// [[Rcpp::export(.Model__ml_all)]]
void Model__ml_all(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->ml_all();
}

// [[Rcpp::export(.Model__laplace_ml_beta_u)]]
void Model__laplace_ml_beta_u(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->laplace_ml_beta_u();
}

// [[Rcpp::export(.Model__laplace_ml_theta)]]
void Model__laplace_ml_theta(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->laplace_ml_theta();
}

// [[Rcpp::export(.Model__laplace_ml_beta_theta)]]
void Model__laplace_ml_beta_theta(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->laplace_ml_beta_theta();
}

// [[Rcpp::export(.Model__nr_beta)]]
void Model__nr_beta(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->nr_beta();
}

// [[Rcpp::export(.Model__laplace_nr_beta_u)]]
void Model__laplace_nr_beta_u(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->laplace_nr_beta_u();
}

// [[Rcpp::export(.Model__laplace_hessian)]]
SEXP Model__laplace_hessian(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd hess = ptr->laplace_hessian();
  return wrap(hess);
}

// [[Rcpp::export(.Model__Sigma)]]
SEXP Model__Sigma(SEXP xp, bool inverse){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd S = ptr->Sigma(inverse);
  return wrap(S);
}

// [[Rcpp::export(.Model__information_matrix)]]
SEXP Model__information_matrix(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd M = ptr->information_matrix();
  return wrap(M);
}

// [[Rcpp::export(.Model__hessian)]]
SEXP Model__hessian(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd hess = ptr->hessian();
  return wrap(hess);
}

// [[Rcpp::export(.Model__u)]]
SEXP Model__u(SEXP xp, bool scaled_){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd u = ptr->u(scaled_);
  return wrap(u);
}

// [[Rcpp::export(.Model__Zu)]]
SEXP Model__Zu(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd Zu = ptr->Zu();
  return wrap(Zu);
}

// [[Rcpp::export(.Model__P)]]
SEXP Model__P(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  int u = ptr->P_;
  return wrap(u);
}

// [[Rcpp::export(.Model__Q)]]
SEXP Model__Q(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  int Q = ptr->Q_;
  return wrap(Q);
}

// [[Rcpp::export(.Model__X)]]
SEXP Model__X(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd X = ptr->linpred_.X();
  return wrap(X);
}

// [[Rcpp::export(.Model__mcmc_sample)]]
void Model__mcmc_sample(SEXP xp, SEXP warmup_, SEXP samples_, SEXP adapt_){
  int warmup = as<int>(warmup_);
  int samples = as<int>(samples_);
  int adapt = as<int>(adapt_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->mcmc_sample(warmup,samples,adapt);
}

// [[Rcpp::export(.Model__set_trace)]]
void Model__set_trace(SEXP xp, SEXP trace_){
  int trace = as<int>(trace_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->set_trace(trace);
}

// [[Rcpp::export(.Model__get_beta)]]
SEXP Model__get_beta(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::VectorXd beta = ptr->linpred_.parameter_vector();
  return wrap(beta);
}

// [[Rcpp::export(.Model__y)]]
SEXP Model__y(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::VectorXd y = ptr->y_;
  return wrap(y);
}

// [[Rcpp::export(.Model__get_theta)]]
SEXP Model__get_theta(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  std::vector<double> theta = ptr->covariance_.parameters_;
  return wrap(theta);
}

// [[Rcpp::export(.Model__get_var_par)]]
SEXP Model__get_var_par(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  double theta = ptr->var_par_;
  return wrap(theta);
}

// [[Rcpp::export(.Model__set_var_par)]]
void Model__set_var_par(SEXP xp, SEXP var_par_){
  double var_par = as<double>(var_par_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->var_par_ = var_par;
}

// [[Rcpp::export(.Model__L)]]
SEXP Model__L(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd L = ptr->covariance_.D(true,false);
  return wrap(L);
}

// [[Rcpp::export(.Model__ZL)]]
SEXP Model__ZL(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd ZL = ptr->covariance_.ZL();
  return wrap(ZL);
}

// [[Rcpp::export(.Model__xb)]]
SEXP Model__xb(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::VectorXd xb = ptr->xb();
  return wrap(xb);
}

// [[Rcpp::export(.Model__aic)]]
SEXP Model__aic(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  double aic = ptr->aic();
  return wrap(aic);
}

// [[Rcpp::export(.Model__mcmc_set_lambda)]]
void Model__mcmc_set_lambda(SEXP xp, SEXP lambda_){
  double lambda = as<double>(lambda_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->mcmc_set_lambda(lambda);
}

// [[Rcpp::export(.Model__mcmc_set_max_steps)]]
void Model__mcmc_set_max_steps(SEXP xp, SEXP max_steps_){
  int max_steps = as<int>(max_steps_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->mcmc_set_max_steps(max_steps);
}

// [[Rcpp::export(.Model__mcmc_set_refresh)]]
void Model__mcmc_set_refresh(SEXP xp, SEXP refresh_){
  int refresh = as<int>(refresh_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->mcmc_set_refresh(refresh);
}

// [[Rcpp::export(.Model__mcmc_set_target_accept)]]
void Model__mcmc_set_target_accept(SEXP xp, SEXP target_){
  double target = as<double>(target_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->mcmc_set_target_accept(target);
}

// [[Rcpp::export(.Model__make_sparse)]]
void Model__make_sparse(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->make_covariance_sparse();
}

// [[Rcpp::export(.Model__make_dense)]]
void Model__make_dense(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->make_covariance_dense();
}

// [[Rcpp::export(.Form_test)]]
SEXP Form__test(SEXP formula){
  std::string formula_ = as<std::string>(formula);
  XPtr<glmmr::Formula> ptr(new glmmr::Formula(formula_));
  return ptr;
}

// [[Rcpp::export(.Linpred_test)]]
SEXP Linpred__test(SEXP formula_,
                SEXP data_,
                SEXP colnames_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  glmmr::Formula f1(formula);
  XPtr<glmmr::LinearPredictor> ptr(new glmmr::LinearPredictor(f1,data,colnames));
  return ptr;
}

// [[Rcpp::export(.Linpred__update_pars)]]
void Linpred__update_pars(SEXP xp,
                          SEXP parameters_){
  std::vector<double> parameters = as<std::vector<double>>(parameters_);
  XPtr<glmmr::LinearPredictor> ptr(xp);
  ptr->update_parameters(parameters);
}

// [[Rcpp::export(.Linpred__xb)]]
SEXP Linpred__xb(SEXP xp){
  XPtr<glmmr::LinearPredictor> ptr(xp);
  Eigen::VectorXd xb = ptr->xb();
  return wrap(xb);
}

// // [[Rcpp::export(.Linpred__dxb)]]
// SEXP Linpred__dxb(SEXP xp){
//   XPtr<glmmr::LinearPredictor> ptr(xp);
//   Eigen::ArrayXXd xb = ptr->deriv();
//   return wrap(xb);
// }

// [[Rcpp::export(.girling_algorithm)]]
SEXP girling_algorithm(SEXP xp, SEXP N_,
                       SEXP sigma_sq_, SEXP C_,
                       SEXP tol_){
  double N = as<double>(N_);
  double sigma_sq = as<double>(sigma_sq_);
  double tol = as<double>(tol_);
  Eigen::VectorXd C = as<Eigen::VectorXd>(C_);
  XPtr<glmmr::Model> ptr(xp);
  Eigen::ArrayXd w = ptr->optimum_weights(N,sigma_sq,C,tol);
  return wrap(w);
}


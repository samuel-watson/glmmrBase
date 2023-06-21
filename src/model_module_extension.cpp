#include <glmmr.h>

namespace Rcpp {
template<>
SEXP wrap(const vector_matrix& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vec") = Rcpp::wrap(x.vec),
      Rcpp::Named("mat") = Rcpp::wrap(x.mat)
  ));
}

template<>
SEXP wrap(const matrix_matrix& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("mat1") = Rcpp::wrap(x.mat1),
      Rcpp::Named("mat2") = Rcpp::wrap(x.mat2)
  ));
}
}

using namespace Rcpp;

// [[Rcpp::export]]
SEXP Model__aic(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  double aic = ptr->aic();
  return wrap(aic);
}

// [[Rcpp::export]]
void Model__mcmc_set_lambda(SEXP xp, SEXP lambda_){
  double lambda = as<double>(lambda_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->mcmc_set_lambda(lambda);
}

// [[Rcpp::export]]
void Model__mcmc_set_max_steps(SEXP xp, SEXP max_steps_){
  int max_steps = as<int>(max_steps_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->mcmc_set_max_steps(max_steps);
}

// [[Rcpp::export]]
void Model__mcmc_set_refresh(SEXP xp, SEXP refresh_){
  int refresh = as<int>(refresh_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->mcmc_set_refresh(refresh);
}

// [[Rcpp::export]]
void Model__mcmc_set_target_accept(SEXP xp, SEXP target_){
  double target = as<double>(target_);
  XPtr<glmmr::Model> ptr(xp);
  ptr->mcmc_set_target_accept(target);
}

// [[Rcpp::export]]
void Model__make_sparse(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->make_covariance_sparse();
}

// [[Rcpp::export]]
void Model__make_dense(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  ptr->make_covariance_dense();
}

// [[Rcpp::export]]
SEXP Model__beta_parameter_names(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  strvec parnames = ptr->linpred_.parameter_names();
  return wrap(parnames);
}

// [[Rcpp::export]]
SEXP Model__theta_parameter_names(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  strvec parnames = ptr->covariance_.parameter_names();
  return wrap(parnames);
}

// [[Rcpp::export]]
SEXP Model__hess_and_grad(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  matrix_matrix parnames = ptr->hess_and_grad();
  return wrap(parnames);
}

// [[Rcpp::export]]
SEXP Model__sandwich(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd sandwich = ptr->sandwich_matrix();
  return wrap(sandwich);
}

// [[Rcpp::export]]
SEXP Model__infomat_theta(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  Eigen::MatrixXd M = ptr->information_matrix_theta();
  return wrap(M);
}

// [[Rcpp::export]]
SEXP Model__kenward_roger(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  matrix_matrix M = ptr->kenward_roger();
  return wrap(M);
}

// [[Rcpp::export]]
SEXP Model__cov_deriv(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  std::vector<Eigen::MatrixXd> M = ptr->sigma_derivatives();
  return wrap(M);
}

// [[Rcpp::export]]
SEXP Model__hessian(SEXP xp){
  XPtr<glmmr::Model> ptr(xp);
  vector_matrix hess = ptr->re_score();
  return wrap(hess);
}

// [[Rcpp::export]]
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

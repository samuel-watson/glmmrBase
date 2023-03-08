#include "../inst/include/glmmr.h"

using namespace Rcpp;

// for some reason for R >4.2.1 the RCPP MODULE crashed R when loading, so it has been rewritten using export of pointers instead
// since these are wrapped in the R6 classes, exposing them to the user should present minimal risk

// [[Rcpp::export(.Covariance__new)]]
SEXP Covariance__new(SEXP form_,SEXP data_, SEXP colnames_){
  std::string form = as<std::string>(form_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  Rcpp::XPtr<glmmr::Covariance> ptr(new glmmr::Covariance(form,data,colnames),true);
  return ptr;
}

// [[Rcpp::export(.Covariance__Z)]]
SEXP Covariance__Z(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd Z = ptr->Z();
  return wrap(Z);
}

// [[Rcpp::export(.Covariance__Update_parameters)]]
void Covariance__Update_parameters(SEXP xp, SEXP parameters_){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  std::vector<double> parameters = as<std::vector<double> >(parameters_);
  ptr->update_parameters_extern(parameters);
}

// [[Rcpp::export(.Covariance__D)]]
SEXP Covariance__D(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd D = ptr->D(false,false);
  return wrap(D);
}

// [[Rcpp::export(.Covariance__D_chol)]]
SEXP Covariance__D_chol(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd D = ptr->D(true,false);
  return wrap(D);
}

// [[Rcpp::export(.Covariance__B)]]
SEXP Covariance__B(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  int B = ptr->B();
  return wrap(B);
}

// [[Rcpp::export(.Covariance__Q)]]
SEXP Covariance__Q(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  int Q = ptr->Q();
  return wrap(Q);
}

// [[Rcpp::export(.Covariance__log_likelihood)]]
SEXP Covariance__log_likelihood(SEXP xp, SEXP u_){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::VectorXd u = as<Eigen::VectorXd>(u_);
  double ll = ptr->log_likelihood(u);
  return wrap(ll);
}

// [[Rcpp::export(.Covariance__log_determinant)]]
SEXP Covariance__log_determinant(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  double ll = ptr->log_determinant();
  return wrap(ll);
}

// [[Rcpp::export(.Covariance__n_cov_pars)]]
SEXP Covariance__n_cov_pars(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  int G = ptr->npar();
  return wrap(G);
}

// [[Rcpp::export(.Covariance__simulate_re)]]
SEXP Covariance__simulate_re(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::VectorXd rr = ptr->sim_re();
  return wrap(rr);
}

// [[Rcpp::export(.Covariance__make_sparse)]]
void Covariance__make_sparse(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  ptr->make_sparse();
}

// [[Rcpp::export(.Covariance__make_dense)]]
void Covariance__make_dense(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  ptr->make_dense();
}

// [[Rcpp::export(.Covariance__any_gr)]]
SEXP Covariance__any_gr(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  bool gr = ptr->any_group_re();
  return wrap(gr);
}

// [[Rcpp::export(.Covariance__parameter_fn_index)]]
SEXP Covariance__parameter_fn_index(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  std::vector<int> gr = ptr->parameter_fn_index();
  return wrap(gr);
}

// [[Rcpp::export(.Covariance__re_terms)]]
SEXP Covariance__re_terms(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  std::vector<std::string> gr = ptr->form_.re_terms();
  return wrap(gr);
}

// [[Rcpp::export(.Covariance__re_count)]]
SEXP Covariance__re_count(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  std::vector<int> gr = ptr->re_count();
  return wrap(gr);
}
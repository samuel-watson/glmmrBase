#include <glmmr.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP Covariance__new(SEXP form_,SEXP data_, SEXP colnames_){
  std::string form = Rcpp::as<std::string>(form_);
  Eigen::ArrayXXd data = Rcpp::as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = Rcpp::as<std::vector<std::string> >(colnames_);
  Rcpp::XPtr<glmmr::Covariance> ptr(new glmmr::Covariance(form,data,colnames),true);
  return ptr;
}

// [[Rcpp::export]]
SEXP Covariance__Z(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd Z = ptr->Z();
  return wrap(Z);
}

// [[Rcpp::export]]
SEXP Covariance__ZL(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd Z = ptr->ZL();
  return wrap(Z);
}

// [[Rcpp::export]]
SEXP Covariance__LZWZL(SEXP xp, SEXP w_){
  Eigen::VectorXd w = Rcpp::as<Eigen::VectorXd>(w_);
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd Z = ptr->LZWZL(w);
  return wrap(Z);
}

// [[Rcpp::export]]
void Covariance__Update_parameters(SEXP xp, SEXP parameters_){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  std::vector<double> parameters = as<std::vector<double> >(parameters_);
  ptr->update_parameters_extern(parameters);
}

// [[Rcpp::export]]
SEXP Covariance__D(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd D = ptr->D(false,false);
  return wrap(D);
}

// [[Rcpp::export]]
SEXP Covariance__D_chol(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd D = ptr->D(true,false);
  return wrap(D);
}

// [[Rcpp::export]]
SEXP Covariance__B(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  int B = ptr->B();
  return Rcpp::wrap(B);
}

// [[Rcpp::export]]
SEXP Covariance__Q(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  int Q = ptr->Q();
  return Rcpp::wrap(Q);
}

// [[Rcpp::export]]
SEXP Covariance__log_likelihood(SEXP xp, SEXP u_){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::VectorXd u = as<Eigen::VectorXd>(u_);
  double ll = ptr->log_likelihood(u);
  return wrap(ll);
}

// [[Rcpp::export]]
SEXP Covariance__log_determinant(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  double ll = ptr->log_determinant();
  return wrap(ll);
}

// [[Rcpp::export]]
SEXP Covariance__n_cov_pars(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  int G = ptr->npar();
  return wrap(G);
}

// [[Rcpp::export]]
SEXP Covariance__simulate_re(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::VectorXd rr = ptr->sim_re();
  return wrap(rr);
}

// [[Rcpp::export]]
void Covariance__make_sparse(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  ptr->set_sparse(true);
}

// [[Rcpp::export]]
void Covariance__make_dense(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  ptr->set_sparse(false);
}

// [[Rcpp::export]]
SEXP Covariance__any_gr(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  bool gr = ptr->any_group_re();
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__parameter_fn_index(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  std::vector<int> gr = ptr->parameter_fn_index();
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__re_terms(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  std::vector<std::string> gr = ptr->form_.re_terms();
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__re_count(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  std::vector<int> gr = ptr->re_count();
  return wrap(gr);
}
#include <glmmr.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP Linpred__new(SEXP formula_,
                  SEXP data_,
                  SEXP colnames_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  glmmr::Formula f1(formula);
  XPtr<glmmr::LinearPredictor> ptr(new glmmr::LinearPredictor(f1,data,colnames));
  return ptr;
}

// [[Rcpp::export]]
void Linpred__update_pars(SEXP xp,
                          SEXP parameters_){
  std::vector<double> parameters = as<std::vector<double>>(parameters_);
  XPtr<glmmr::LinearPredictor> ptr(xp);
  ptr->update_parameters(parameters);
}

// [[Rcpp::export]]
SEXP Linpred__xb(SEXP xp){
  XPtr<glmmr::LinearPredictor> ptr(xp);
  Eigen::VectorXd xb = ptr->xb();
  return wrap(xb);
}

// [[Rcpp::export]]
SEXP Linpred__x(SEXP xp){
  XPtr<glmmr::LinearPredictor> ptr(xp);
  Eigen::MatrixXd X = ptr->X();
  return wrap(X);
}

// [[Rcpp::export]]
SEXP Linpred__beta_names(SEXP xp){
  XPtr<glmmr::LinearPredictor> ptr(xp);
  std::vector<std::string> X = ptr->parameter_names();
  return wrap(X);
}

// [[Rcpp::export]]
SEXP Linpred__any_nonlinear(SEXP xp){
  XPtr<glmmr::LinearPredictor> ptr(xp);
  bool anl = ptr->any_nonlinear();
  return wrap(anl);
}

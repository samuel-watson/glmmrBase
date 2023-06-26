#include <glmmr.h>

using namespace Rcpp;

// [[Rcpp::export]]
std::vector<std::string> re_names(const std::string& formula){
  glmmr::Formula form(formula);
  std::vector<std::string> re(form.re_.size());
  for(int i = 0; i < form.re_.size(); i++){
    re[i] = "("+form.z_[i]+"|"+form.re_[i]+")";
  }
  return re;
}

// [[Rcpp::export]]
Eigen::VectorXd attenuate_xb(const Eigen::VectorXd& xb,
                             const Eigen::MatrixXd& Z,
                             const Eigen::MatrixXd& D,
                             const std::string& link){
  Eigen::VectorXd linpred = glmmr::maths::attenuted_xb(xb,Z,D,link);
  return linpred;
}

// [[Rcpp::export]]
Eigen::VectorXd dlinkdeta(const Eigen::VectorXd& xb,
                          const std::string& link){
  Eigen::VectorXd deta = glmmr::maths::detadmu(xb,link);
  return deta;
}

// This is a function in development - it works as expected,
// but is subject to current research and so it is currently
// not exposed to the user through the model class yet
// [[Rcpp::export]]
SEXP girling_algorithm(SEXP xp, SEXP N_,
                       SEXP sigma_sq_, SEXP C_,
                       SEXP tol_){
  double N = as<double>(N_);
  double sigma_sq = as<double>(sigma_sq_);
  double tol = as<double>(tol_);
  Eigen::VectorXd C = as<Eigen::VectorXd>(C_);
  XPtr<glmmr::Model> ptr(xp);
  Eigen::ArrayXd w = ptr->optim.optimum_weights(N,sigma_sq,C,tol);
  return wrap(w);
}


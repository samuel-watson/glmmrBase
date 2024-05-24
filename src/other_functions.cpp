#include <glmmr.h>

using namespace Rcpp;

// [[Rcpp::export]]
std::vector<std::string> re_names(const std::string& formula,
                                  bool as_formula = true){
  glmmr::Formula form(formula);
  std::vector<std::string> re;
  if(as_formula){
    re.resize(form.re_.size());
    for(int i = 0; i < form.re_.size(); i++){
      re[i] = "("+form.z_[i]+"|"+form.re_[i]+")";
    }
  } else {
    for(int i = 0; i < form.re_.size(); i++){
      re.push_back(form.re_[i]);
      re.push_back(form.z_[i]);
    }
  }
  
  return re;
}

// [[Rcpp::export]]
Eigen::VectorXd attenuate_xb(const Eigen::VectorXd& xb,
                             const Eigen::MatrixXd& Z,
                             const Eigen::MatrixXd& D,
                             const std::string& link){
  Eigen::VectorXd linpred = glmmr::maths::attenuted_xb(xb,Z,D,glmmr::str_to_link.at(link));
  return linpred;
}

// [[Rcpp::export]]
Eigen::VectorXd dlinkdeta(const Eigen::VectorXd& xb,
                          const std::string& link){
  Eigen::VectorXd deta = glmmr::maths::detadmu(xb,glmmr::str_to_link.at(link));
  return deta;
}

// Access to this function is provided to the user in the 
// glmmrOptim package
// [[Rcpp::export]]
SEXP girling_algorithm(SEXP xp, 
                       SEXP N_,
                       SEXP C_,
                       SEXP tol_){
  double N = as<double>(N_);
  double tol = as<double>(tol_);
  Eigen::VectorXd C = as<Eigen::VectorXd>(C_);
  XPtr<glmm> ptr(xp);
  Eigen::ArrayXd w = ptr->optim.optimum_weights(N,C,tol);
  return wrap(w);
}

// [[Rcpp::export]]
SEXP get_variable_names(SEXP formula_, 
                        SEXP colnames_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data(1,1);
  Eigen::MatrixXd Xdata(1,1);
  data.setZero();
  Xdata.setZero();
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  glmmr::Formula form(formula);
  glmmr::calculator calc;
  bool out = glmmr::parse_formula(form.linear_predictor_,calc,data,colnames,Xdata,false,false);
  (void)out;
  return wrap(calc.data_names);
}

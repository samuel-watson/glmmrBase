#include <glmmr.h>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export(.genX)]]
Eigen::MatrixXd genX(const std::string& formula,
                  const Eigen::ArrayXXd& data,
                  const std::vector<std::string>& colnames) {
 glmmr::Formula form(formula);
 glmmr::LinearPredictor lin(form,data,colnames);
 return lin.X();
}

// // [[Rcpp::export(.x_names)]]
// std::vector<std::string> x_names(const std::string& formula){
//   glmmr::Formula form(formula);
//   return form.fe_;
// }

// [[Rcpp::export(.re_names)]]
std::vector<std::string> re_names(const std::string& formula){
 glmmr::Formula form(formula);
 std::vector<std::string> re(form.re_.size());
 for(int i = 0; i < form.re_.size(); i++){
   re[i] = "("+form.z_[i]+"|"+form.re_[i]+")";
 }
 return re;
}

// [[Rcpp::export(.gen_dhdmu)]]
Eigen::VectorXd gen_dhdmu(const Eigen::VectorXd& xb,
                          std::string family,
                          std::string link) {
  Eigen::VectorXd out = glmmr::maths::dhdmu(xb, family, link);
  return out;
}


// [[Rcpp::export(.gen_sigma_approx)]]
Eigen::MatrixXd gen_sigma_approx(const Eigen::VectorXd& xb,
                                 const Eigen::MatrixXd& Z,
                                 const Eigen::MatrixXd& D,
                                 std::string family,
                                 std::string link,
                                 double var_par,
                                 bool attenuate
                                 ){
  Eigen::MatrixXd S(xb.size(),xb.size());
  Eigen::VectorXd linpred(xb);
  if(attenuate){
    linpred = glmmr::maths::attenuted_xb(xb,Z,D,link);
  }

  Eigen::VectorXd W = glmmr::maths::dhdmu(linpred,family,link);
  double nvar_par = 1.0;
  if(family=="gaussian"){
    nvar_par *= var_par*var_par;
  } else if(family=="Gamma"){
    nvar_par *= 1/var_par;
  } else if(family=="beta"){
    nvar_par *= (1+var_par);
  } else if(family=="binomial"){
    nvar_par *= 1/var_par;
  }
  W *= nvar_par;
  //W = W.array().inverse().matrix();
  S = Z*D*Z.transpose();
  S += W.asDiagonal();
  return S;
}

// [[Rcpp::export(.attenuate_xb)]]
Eigen::VectorXd attenuate_xb(const Eigen::VectorXd& xb,
                             const Eigen::MatrixXd& Z,
                             const Eigen::MatrixXd& D,
                             const std::string& link){
  Eigen::VectorXd linpred = glmmr::maths::attenuted_xb(xb,Z,D,link);
  return linpred;
}

// [[Rcpp::export(.dlinkdeta)]]
Eigen::VectorXd dlinkdeta(const Eigen::VectorXd& xb,
                            const std::string& link){
 Eigen::VectorXd deta = glmmr::maths::detadmu(xb,link);
 return deta;
}
#include "../inst/include/glmmr.h"

// [[Rcpp::depends(RcppEigen)]]

//' Generates the covariance matrix of the random effects
//' 
//' Generates the covariance matrix of the random effects.
//' @param formula A string specifying the formula
//' @param data A matrix with the data 
//' @param colnames A vector of strings specifying the column names of the data
//' @param theta Vector of parameters used to generate the matrix D. 
//' @return A symmetric positive definite covariance matrix
// [[Rcpp::export]]
Eigen::MatrixXd genD(const std::string& formula,
                     const Eigen::ArrayXXd& data,
                     const std::vector<std::string>& colnames,
                     const Eigen::VectorXd& theta) {
  glmmr::Formula form(formula);
  glmmr::Covariance cov(form,data,colnames,theta);
  return cov.D();
}

//' Generates the design matrix of the random effects
//' 
//' Generates the design matrix of the random effects.
//' @param formula A string specifying the formula
//' @param data A matrix with the data 
//' @param colnames A vector of strings specifying the column names of the data
//' @return A symmetric positive definite covariance matrix
// [[Rcpp::export]]
Eigen::MatrixXd genZ(const std::string& formula,
                    const Eigen::ArrayXXd& data,
                    const std::vector<std::string>& colnames) {
 glmmr::Formula form(formula);
 glmmr::Covariance cov(form,data,colnames);
 return cov.Z();
}

//' Generates the matrix X
//' 
//' Generates the matrix X from a formula and data. This function replicates the functionality of the 
//' R function model.matrix, but is intended to be extended to accomodate linearisation of non-linear functions
//' and other functionality in future versions.
//' @param formula A string specifying the formula
//' @param data A matrix with the data 
//' @param colnames A vector of strings specifying the column names of the data
//' @return A symmetric positive definite covariance matrix
// [[Rcpp::export]]
Eigen::MatrixXd genX(const std::string& formula,
                    const Eigen::ArrayXXd& data,
                    const std::vector<std::string>& colnames) {
 glmmr::Formula form(formula);
 glmmr::LinearPredictor lin(form,data,colnames);
 return lin.X();
}

//' Generates the Cholesky decomposition covariance matrix of the random effects
//' 
//' Generates the Cholesky Decomposition of the covariance matrix of the random effects. Used internally in the Covariance class.
//' @param formula A string specifying the formula
//' @param data A matrix with the data 
//' @param colnames A vector of strings specifying the column names of the data
//' @param theta Vector of parameters used to generate the matrix D. 
//' @return A symmetric positive definite covariance matrix
// [[Rcpp::export]]
Eigen::MatrixXd genCholD(const std::string& formula,
                         const Eigen::ArrayXXd& data,
                         const std::vector<std::string>& colnames,
                         const Eigen::VectorXd& theta) {
 glmmr::Formula form(formula);
 glmmr::Covariance cov(form,data,colnames,theta);
 return cov.D(true,false);
}

//' Returns the number of covariance parameters required for the formula
//' 
//' @param formula A string specifying the formula
//' @param data A matrix with the data 
//' @param colnames A vector of strings specifying the column names of the data
//' @return Integer count of parameters
// [[Rcpp::export]]
int n_cov_pars(const std::string& formula,
                          const Eigen::ArrayXXd& data,
                          const std::vector<std::string>& colnames) {
 glmmr::Formula form(formula);
 glmmr::Covariance cov(form,data,colnames);
 return cov.npar();
}


//' Generates a sample of random effects
//' 
//' Generates a sample of random effects from the specified covariance matrix.
//' @param formula A string specifying the formula
//' @param data A matrix with the data 
//' @param colnames A vector of strings specifying the column names of the data
//' @param theta Vector of parameters used to generate the matrix D. 
//' @return A symmetric positive definite covariance matrix
// [[Rcpp::export]]
Eigen::VectorXd sample_re(const std::string& formula,
                        const Eigen::ArrayXXd& data,
                        const std::vector<std::string>& colnames,
                        const Eigen::VectorXd& theta) {
  glmmr::Formula form(formula);
  glmmr::Covariance cov(form,data,colnames,theta);
  return cov.sim_re();
}

//' Gets the names of the fixed effects variables
//' 
//' @param formula A string specifying the formula
//' @return A vector of variable names
// [[Rcpp::export]]
std::vector<std::string> x_names(const std::string& formula){
  glmmr::Formula form(formula);
  return form.fe_;
}
 
//' Generates the inverse GLM iterated weights.
//' 
//' @param xb Vector with mean function value evaluated at fitted model parameters
//' @param family String declaring model family
//' @param link String declaring model link function
//' @return Vector of derivative values
// [[Rcpp::export]]
Eigen::VectorXd gen_dhdmu(const Eigen::VectorXd& xb,
                          std::string family,
                          std::string link) {
  Eigen::VectorXd out = glmmr::maths::dhdmu(xb, family, link);
  return out;
}

//' Generates an approximation to the covariance of y
//' 
//' Generates a first-order approximation to the covariance matrix 
//' of y based on the marginal quasi-likelihood. This approximation is
//' exact for the Gaussian-identity model. Used internally by the \link[glmmrBase]{Model} class.
//' @param xb Vector of values of the linear predictor
//' @param Z Random effects design matrix
//' @param D Covariance matrix of the random effects
//' @param family String specifying the family
//' @param link String specifying the link function
//' @param var_par Value of the optional scale parameter
//' @param attenuate Logical indicating whether to use "attenuated" values of the linear predictor
//' @return A matrix
// [[Rcpp::export]]
Eigen::MatrixXd gen_sigma_approx(const Eigen::VectorXd& xb,
                                 const Eigen::MatrixXd& Z,
                                 const Eigen::MatrixXd& D,
                                 std::string family,
                                 std::string link,
                                 double var_par,
                                 bool attenuate
                                 ){
  Eigen::MatrixXd S(xb.size(),xb.size());
  // generate the linear predictor
  Eigen::VectorXd linpred(xb);
  if(attenuate){
    linpred = glmmr::maths::attenuted_xb(xb,Z,D,link);
  }

  Eigen::VectorXd W = glmmr::maths::dhdmu(linpred,family,link);
  double nvar_par = 1.0;
  if(family=="gaussian"){
    nvar_par *= var_par*var_par;
  } else if(family=="Gamma"){
    nvar_par *= var_par;
  } else if(family=="beta"){
    nvar_par *= (1+var_par);
  }
  W *= nvar_par;
  //W = W.array().inverse().matrix();
  S = Z*D*Z.transpose();
  S += W.asDiagonal();
  return S;
}

//' Return marginal expectation with attenuation
//' 
//' The marginal expectation may be better approximated using an attenuation 
//' scheme in non-linear models. This function returns the attenuated linear predictor.
//' Used internally.
//' @param xb Vector of values of the linear predictor
//' @param Z Random effects design matrix
//' @param D Covariance matrix of the random effects
//' @param link String specifying the link function
//' @return A vector
// [[Rcpp::export]]
Eigen::VectorXd attenuate_xb(const Eigen::VectorXd& xb,
                             const Eigen::MatrixXd& Z,
                             const Eigen::MatrixXd& D,
                             const std::string& link){
  Eigen::VectorXd linpred = glmmr::maths::attenuted_xb(xb,Z,D,link);
  return linpred;
}

//' Partial derivative of link function with respect to linear predictor
//' 
//' Returns the partial derivative of link function with respect to linear predictor.
//' Used internally.
//' @param xb Vector of values of the linear predictor
//' @param link String specifying the link function
//' @return A vector
// [[Rcpp::export]]
Eigen::VectorXd dlinkdeta(const Eigen::VectorXd& xb,
                            const std::string& link){
 Eigen::VectorXd deta = glmmr::maths::detadmu(xb,link);
 return deta;
}
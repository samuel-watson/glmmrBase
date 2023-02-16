#include "../inst/include/glmmr.h"

// [[Rcpp::depends(RcppEigen)]]

//' Generates the covariance matrix of the random effects
//' 
//' Generates the covariance matrix of the random effects from a sparse representation. Used internally in the Covariance class.
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param gamma Vector of parameters used to generate the matrix D. 
//' @return A symmetric positive definite covariance matrix
// [[Rcpp::export]]
Eigen::MatrixXd genD(const Eigen::ArrayXXi &cov,
                     const Eigen::ArrayXd &data,
                     const Eigen::ArrayXd &eff_range,
                     const Eigen::VectorXd& gamma) {
  glmmr::DMatrix dmat(cov,data,eff_range, gamma);
  Eigen::MatrixXd DBlocks = dmat.genD();
  return(DBlocks);
}

//' Generates the Cholesky decomposition covariance matrix of the random effects
//' 
//' Generates the Cholesky Decomposition of the covariance matrix of the random effects. Used internally in the Covariance class.
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of varaibles
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param gamma Vector of parameters used to generate the matrix D. 
//' @return A lower triangular matrix
// [[Rcpp::export]]
Eigen::MatrixXd genCholD(const Eigen::ArrayXXi &cov,
                         const Eigen::ArrayXd &data,
                         const Eigen::ArrayXd &eff_range,
                         const Eigen::VectorXd& gamma) {
  glmmr::DMatrix dmat(cov,data,eff_range, gamma);
  Eigen::MatrixXd DBlocks = dmat.genD(true, false);
  return(DBlocks);
}

//' Generates a sample of random effects
//' 
//' Generates a sample of random effects from the specified covariance matrix.
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of varaibles
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param gamma Vector of parameters used to generate the matrix D. 
//' @return A lower triangular matrix
// [[Rcpp::export]]
Eigen::VectorXd sample_re(const Eigen::ArrayXXi &cov,
                          const Eigen::ArrayXd &data,
                          const Eigen::ArrayXd &eff_range,
                          const Eigen::VectorXd& gamma){
  glmmr::DMatrix dmat(cov,data,eff_range, gamma);
  Eigen::VectorXd samps = dmat.sim_re();
  return samps;
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
//' @param qlik Not used.
//' @return A matrix
// [[Rcpp::export]]
Eigen::MatrixXd gen_sigma_approx(const Eigen::VectorXd& xb,
                                 const Eigen::MatrixXd& Z,
                                 const Eigen::MatrixXd& D,
                                 std::string family,
                                 std::string link,
                                 double var_par,
                                 bool attenuate,
                                 bool qlik = true
                                 ){
  Eigen::MatrixXd S(xb.size(),xb.size());
  // generate the linear predictor
  Eigen::VectorXd linpred(xb);
  if(attenuate){
    linpred = glmmr::maths::attenuted_xb(xb,Z,D,link);
  }

  if(qlik){
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
  } else {
    // this is useless - supposed to be a GEE approach but
    // doesn't provide useful answers. leaving here for now
    // incase we come back to it
    Eigen::VectorXd L = glmmr::maths::detadmu(linpred,link);
    Eigen::VectorXd mu = glmmr::maths::mod_inv_func(linpred,link);
    Eigen::VectorXd A = glmmr::maths::marginal_var(mu,family,var_par);
    L = L.array().inverse().matrix();
    S = L.asDiagonal()*Z*D*Z.transpose()*L.asDiagonal();
    S += A.asDiagonal();
  }
  return S;
}

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

//' Generates the derivative of the link function with respect to the mean. Used internally in the Model function class.
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

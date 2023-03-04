#include "../inst/include/glmmr.h"

RCPP_MODULE(covariance_cpp){
  using namespace Rcpp;
  
  class_<glmmr::Covariance>("covariance")
    .constructor<std::string,Eigen::ArrayXXd,std::vector<std::string> >()
    .method("Z",&glmmr::Covariance::Z,"Return matrix Z")
    .method("D",&glmmr::Covariance::D,"Return matrix D")
    .method("B",&glmmr::Covariance::B,"Return the number of blocks")
    .method("Q",&glmmr::Covariance::Q,"Return the number of random effects")
    .method("log_likelihood",&glmmr::Covariance::log_likelihood,"Log likelihood")
    .method("log_determinant",&glmmr::Covariance::log_determinant,"Log determinant of D")
    .method("n_cov_pars",&glmmr::Covariance::npar,"Number of covariance function parameters")
    .method("simulate_re",&glmmr::Covariance::sim_re,"Simulate random effects")
    .method("update_parameters",&glmmr::Covariance::update_parameters_extern,"updates the parameters")
  ;
}
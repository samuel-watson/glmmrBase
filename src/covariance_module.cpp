#include "../inst/include/glmmr.h"

using namespace Rcpp;

// for some reason for R >4.2.1 the RCPP MODULE crashed R when loading, so it has been rewritten using export of pointers instead
// since these are wrapped in the R6 classes, exposing them to the user should present minimal risk

SEXP Covariance__new(SEXP form_,SEXP data_, SEXP colnames_){
  std::string form = as<std::string>(form_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  Rcpp::XPtr<glmmr::Covariance> ptr(new glmmr::Covariance(form,data,colnames),true);
  return ptr;
}

SEXP Covariance__Z(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd Z = ptr->Z();
  return wrap(Z);
}

void Covariance__Update_parameters(SEXP xp, SEXP parameters_){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  std::vector<double> parameters = as<std::vector<double> >(parameters_);
  ptr->update_parameters_extern(parameters);
}

SEXP Covariance__D(SEXP xp){
  Rcpp::XPtr<glmmr::Covariance> ptr(xp);
  Eigen::MatrixXd D = ptr->D(false,false);
  return wrap(D);
}

// RCPP_EXPOSED_CLASS(glmmr::Covariance)
//   
// 
// RCPP_MODULE(covariance_cpp){
//   using namespace Rcpp;
//   
//   
//   class_<glmmr::Covariance>("covariance")
//     .constructor<std::string,Eigen::ArrayXXd,std::vector<std::string> >()
//     .method("Z",&glmmr::Covariance::Z,"Return matrix Z")
//     // .method("D",&glmmr::Covariance::D,"Return matrix D")
//     // .method("B",&glmmr::Covariance::B,"Return the number of blocks")
//     // .method("Q",&glmmr::Covariance::Q,"Return the number of random effects")
//     // .method("log_likelihood",&glmmr::Covariance::log_likelihood,"Log likelihood")
//     // .method("log_determinant",&glmmr::Covariance::log_determinant,"Log determinant of D")
//     // .method("n_cov_pars",&glmmr::Covariance::npar,"Number of covariance function parameters")
//     // .method("simulate_re",&glmmr::Covariance::sim_re,"Simulate random effects")
//     // .method("update_parameters",&glmmr::Covariance::update_parameters_extern,"updates the parameters")
//     // .method("make_sparse",&glmmr::Covariance::make_sparse,"uses sparse matrix methods")
//   ;
// }
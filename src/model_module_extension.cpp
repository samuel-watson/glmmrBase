#include <glmmr.h>

namespace Rcpp {
template<>
SEXP wrap(const vector_matrix& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vec") = Rcpp::wrap(x.vec),
      Rcpp::Named("mat") = Rcpp::wrap(x.mat)
  ));
}

template<>
SEXP wrap(const matrix_matrix& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("mat1") = Rcpp::wrap(x.mat1),
      Rcpp::Named("mat2") = Rcpp::wrap(x.mat2),
      Rcpp::Named("a") = Rcpp::wrap(x.a),
      Rcpp::Named("b") = Rcpp::wrap(x.b)
  ));
}

template<>
SEXP wrap(const kenward_data& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vcov_beta") = Rcpp::wrap(x.vcov_beta),
      Rcpp::Named("vcov_theta") = Rcpp::wrap(x.vcov_theta),
      Rcpp::Named("dof") = Rcpp::wrap(x.dof)
  ));
}
}

using namespace Rcpp;

// [[Rcpp::export]]
SEXP Covariance__submatrix(SEXP xp, int i){
  XPtr<nngp> ptr(xp);
  vector_matrix result = ptr->submatrix(i);
  return wrap(result);
}


// [[Rcpp::export]]
SEXP Model__aic(SEXP xp, int type = 0){
  double aic;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  aic = ptr->optim.aic();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  aic = ptr->optim.aic();
  break;
}
  }
  return wrap(aic);
}

// [[Rcpp::export]]
void Model__mcmc_set_lambda(SEXP xp, SEXP lambda_, int type = 0){
  double lambda = as<double>(lambda_);
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->mcmc.mcmc_set_lambda(lambda);
        break;
      }
    case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        ptr->mcmc.mcmc_set_lambda(lambda);
        break;
      }
  }
}

// [[Rcpp::export]]
void Model__mcmc_set_max_steps(SEXP xp, SEXP max_steps_, int type = 0){
  int max_steps = as<int>(max_steps_);
  switch(type){
  case 0:
  {
    XPtr<glmm> ptr(xp);
    ptr->mcmc.mcmc_set_max_steps(max_steps);
    break;
  }
  case 1:
  {
    XPtr<glmm_nngp> ptr(xp);
    ptr->mcmc.mcmc_set_max_steps(max_steps);
    break;
  }
  }
}

// [[Rcpp::export]]
void Model__mcmc_set_refresh(SEXP xp, SEXP refresh_, int type = 0){
  int refresh = as<int>(refresh_);
  switch(type){
  case 0:
  {
    XPtr<glmm> ptr(xp);
    ptr->mcmc.mcmc_set_refresh(refresh);
    break;
  }
  case 1:
  {
    XPtr<glmm_nngp> ptr(xp);
    ptr->mcmc.mcmc_set_refresh(refresh);
    break;
  }
  }
}

// [[Rcpp::export]]
void Model__mcmc_set_target_accept(SEXP xp, SEXP target_, int type = 0){
  double target = as<double>(target_);
  switch(type){
  case 0:
  {
    XPtr<glmm> ptr(xp);
    ptr->mcmc.mcmc_set_target_accept(target);
    break;
  }
  case 1:
  {
    XPtr<glmm_nngp> ptr(xp);
    ptr->mcmc.mcmc_set_target_accept(target);
    break;
  }
  }
}

// [[Rcpp::export]]
void Model__make_sparse(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  ptr->model.make_covariance_sparse();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  ptr->model.make_covariance_sparse();
  break;
}
  }
  
}

// [[Rcpp::export]]
void Model__make_dense(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  ptr->model.make_covariance_dense();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  ptr->model.make_covariance_dense();
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__beta_parameter_names(SEXP xp, int type = 0){
  strvec parnames;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  parnames = ptr->model.linear_predictor.parameter_names();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  parnames = ptr->model.linear_predictor.parameter_names();
  break;
}
  }
  return wrap(parnames);
}

// [[Rcpp::export]]
SEXP Model__theta_parameter_names(SEXP xp, int type = 0){
  strvec parnames;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  parnames = ptr->model.covariance.parameter_names();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  parnames = ptr->model.covariance.parameter_names();
  break;
}
  }
  return wrap(parnames);
}

// [[Rcpp::export]]
SEXP Model__hess_and_grad(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  matrix_matrix parnames = ptr->matrix.hess_and_grad();
  return wrap(parnames);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  matrix_matrix parnames = ptr->matrix.hess_and_grad();
  return wrap(parnames);
  break;
}
  }  
}

// [[Rcpp::export]]
SEXP Model__sandwich(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd sandwich = ptr->matrix.sandwich_matrix();
  return wrap(sandwich);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd sandwich = ptr->matrix.sandwich_matrix();
  return wrap(sandwich);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__infomat_theta(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Eigen::MatrixXd M = ptr->matrix.information_matrix_theta();
  return wrap(M);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Eigen::MatrixXd M = ptr->matrix.information_matrix_theta();
  return wrap(M);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__kenward_roger(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  kenward_data M = ptr->matrix.kenward_roger();
  return wrap(M);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  kenward_data M = ptr->matrix.kenward_roger();
  return wrap(M);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__cov_deriv(SEXP xp, int type = 0){
  std::vector<Eigen::MatrixXd> M;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  M = ptr->matrix.sigma_derivatives();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  M = ptr->matrix.sigma_derivatives();
  break;
}
  }
  return wrap(M);
}

// [[Rcpp::export]]
SEXP Model__hessian(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  vector_matrix hess = ptr->matrix.re_score();
  return wrap(hess);
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  vector_matrix hess = ptr->matrix.re_score();
  return wrap(hess);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Model__predict(SEXP xp, SEXP newdata_,
                    SEXP newoffset_,
                    int m, int type = 0){
  Eigen::ArrayXXd newdata = Rcpp::as<Eigen::ArrayXXd>(newdata_);
  Eigen::ArrayXd newoffset = Rcpp::as<Eigen::ArrayXd>(newoffset_);
  switch(type){
    case 0:
      {
    XPtr<glmm> ptr(xp);
    vector_matrix res = ptr->re.predict_re(newdata,newoffset);
    Eigen::MatrixXd samps(newdata.rows(),m>0 ? m : 1);
    if(m>0){
      samps = glmmr::maths::sample_MVN(res,m);
    } else {
      samps.setZero();
    }
    Eigen::VectorXd xb = ptr->model.linear_predictor.predict_xb(newdata,newoffset);
    return Rcpp::List::create(
      Rcpp::Named("linear_predictor") = wrap(xb),
      Rcpp::Named("re_parameters") = wrap(res),
      Rcpp::Named("samples") = wrap(samps)
    );
      }
  case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        vector_matrix res = ptr->re.predict_re(newdata,newoffset);
        Eigen::MatrixXd samps(newdata.rows(),m>0 ? m : 1);
        if(m>0){
          samps = glmmr::maths::sample_MVN(res,m);
        } else {
          samps.setZero();
        }
        Eigen::VectorXd xb = ptr->model.linear_predictor.predict_xb(newdata,newoffset);
        return Rcpp::List::create(
          Rcpp::Named("linear_predictor") = wrap(xb),
          Rcpp::Named("re_parameters") = wrap(res),
          Rcpp::Named("samples") = wrap(samps)
        );
      }
  }
  
}

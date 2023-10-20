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
void Model_hsgp__set_approx_pars(SEXP xp, SEXP m_, SEXP L_){
  std::vector<int> m = as<std::vector<int> >(m_);
  Eigen::ArrayXd L = as<Eigen::ArrayXd>(L_);
  XPtr<glmm_hsgp> ptr(xp);
  ptr->model.covariance.update_approx_parameters(m,L);
}

// [[Rcpp::export]]
SEXP Model_hsgp__dim(SEXP xp){
  XPtr<glmm_hsgp> ptr(xp);
  int dim = ptr->model.covariance.dim;
  return wrap(dim);
}

// [[Rcpp::export]]
SEXP Model__aic(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.aic());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
void Model__mcmc_set_lambda(SEXP xp, SEXP lambda_, int type = 0){
  double lambda = as<double>(lambda_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [&lambda](auto ptr){ptr->mcmc.mcmc_set_lambda(lambda);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__mcmc_set_max_steps(SEXP xp, SEXP max_steps_, int type = 0){
  int max_steps = as<int>(max_steps_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [&max_steps](auto ptr){ptr->mcmc.mcmc_set_max_steps(max_steps);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__mcmc_set_refresh(SEXP xp, SEXP refresh_, int type = 0){
  int refresh = as<int>(refresh_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [&refresh](auto ptr){ptr->mcmc.mcmc_set_refresh(refresh);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__mcmc_set_target_accept(SEXP xp, SEXP target_, int type = 0){
  double target = as<double>(target_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [&target](auto ptr){ptr->mcmc.mcmc_set_target_accept(target);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__make_sparse(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->model.make_covariance_sparse();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__make_dense(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->model.make_covariance_dense();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__beta_parameter_names(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.parameter_names());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<std::string> >(S));
}

// [[Rcpp::export]]
SEXP Model__theta_parameter_names(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.parameter_names());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<std::string> >(S));
}

// [[Rcpp::export]]
SEXP Model__hess_and_grad(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.hess_and_grad());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<matrix_matrix>(S));
}

// [[Rcpp::export]]
SEXP Model__sandwich(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.sandwich_matrix());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__infomat_theta(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.information_matrix_theta());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__kenward_roger(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.kenward_roger());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<kenward_data>(S));
}

// [[Rcpp::export]]
SEXP Model__cov_deriv(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.sigma_derivatives());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<Eigen::MatrixXd> >(S));
}

// [[Rcpp::export]]
SEXP Model__hessian(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.re_score());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<vector_matrix>(S));
}

// [[Rcpp::export]]
SEXP Model__predict(SEXP xp, SEXP newdata_,
                    SEXP newoffset_,
                    int m, int type = 0){
  Eigen::ArrayXXd newdata = Rcpp::as<Eigen::ArrayXXd>(newdata_);
  Eigen::ArrayXd newoffset = Rcpp::as<Eigen::ArrayXd>(newoffset_);
  
  glmmrType model(xp,type);
  auto functor_re = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->re.predict_re(newdata,newoffset));}
  };
  auto functor_xb = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->model.linear_predictor.predict_xb(newdata,newoffset));}
  };
  auto S_re = std::visit(functor_re,model.ptr);
  auto S_xb = std::visit(functor_xb,model.ptr);
  
  vector_matrix res = std::get<vector_matrix>(S_re);
  Eigen::VectorXd xb = std::get<Eigen::VectorXd>(S_xb);
  
  Eigen::MatrixXd samps(newdata.rows(),m>0 ? m : 1);
  if(m>0){
    samps = glmmr::maths::sample_MVN(res,m);
  } else {
    samps.setZero();
  }
  
  return Rcpp::List::create(
    Rcpp::Named("linear_predictor") = wrap(xb),
    Rcpp::Named("re_parameters") = wrap(res),
    Rcpp::Named("samples") = wrap(samps)
  );
}

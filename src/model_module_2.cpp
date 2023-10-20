#include <glmmr.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP Model__get_W(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.W.W());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}

// [[Rcpp::export]]
SEXP Model__log_prob(SEXP xp, SEXP v_, int type = 0){
  Eigen::VectorXd v = as<Eigen::VectorXd>(v_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&v](auto ptr){return returnType(ptr->mcmc.log_prob(v));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
void Model__set_bobyqa_control(SEXP xp,SEXP npt_, SEXP rhobeg_, SEXP rhoend_, int type = 0){
  int npt = as<int>(npt_);
  double rhobeg = as<double>(rhobeg_);
  double rhoend = as<double>(rhoend_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->optim.set_bobyqa_control(npt,rhobeg,rhoend);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__log_gradient(SEXP xp, SEXP v_, SEXP beta_, int type = 0){
  Eigen::VectorXd v = as<Eigen::VectorXd>(v_);
  bool beta = as<bool>(beta_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->matrix.log_gradient(v,beta));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}

// [[Rcpp::export]]
SEXP Model__linear_predictor(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.linpred());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__log_likelihood(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.log_likelihood());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
void Model__ml_theta(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.ml_theta();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__cov_set_nn(SEXP xp, int nn){
  XPtr<glmm_nngp> ptr(xp);
  ptr->model.covariance.gen_NN(nn);
}

// [[Rcpp::export]]
void Model__ml_beta(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.ml_beta();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__ml_all(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.ml_all();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__laplace_ml_beta_u(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.laplace_ml_beta_u();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__laplace_ml_theta(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.laplace_ml_theta();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__laplace_ml_beta_theta(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.laplace_ml_beta_theta();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__nr_beta(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.nr_beta();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__laplace_nr_beta_u(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.laplace_nr_beta_u();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__Sigma(SEXP xp, bool inverse, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&inverse](auto ptr){return returnType(ptr->matrix.Sigma(inverse));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__information_matrix(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.information_matrix());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__obs_information_matrix(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.observed_information_matrix());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__u(SEXP xp, bool scaled_, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&scaled_](auto ptr){return returnType(ptr->re.u(scaled_));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__Zu(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->re.Zu());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__X(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.X());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
void Model__mcmc_sample(SEXP xp, SEXP warmup_, SEXP samples_, SEXP adapt_, int type = 0){
  int warmup = as<int>(warmup_);
  int samples = as<int>(samples_);
  int adapt = as<int>(adapt_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->mcmc.mcmc_sample(warmup,samples,adapt);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_trace(SEXP xp, SEXP trace_, int type = 0){
  int trace = as<int>(trace_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [&trace](auto ptr){ptr->set_trace(trace);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__get_beta(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.parameter_vector());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}

// [[Rcpp::export]]
SEXP Model__y(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.data.y);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}

// [[Rcpp::export]]
SEXP Model__get_theta(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.parameters_);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<double> >(S));
}

// [[Rcpp::export]]
SEXP Model__get_var_par(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.data.var_par);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
SEXP Model__get_variance(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.data.variance);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::ArrayXd>(S));
}

// [[Rcpp::export]]
void Model__set_var_par(SEXP xp, SEXP var_par_, int type = 0){
  double var_par = as<double>(var_par_);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [&var_par](auto ptr){ptr->model.data.set_var_par(var_par);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_trials(SEXP xp, SEXP trials, int type = 0){
  Eigen::ArrayXd var_par = as<Eigen::ArrayXd>(trials);
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {}, 
    [&var_par](auto ptr){ptr->model.data.set_variance(var_par);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__L(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.D(true,false));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__ZL(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.ZL());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__hessian_numerical(SEXP xp, double tol = 1e-4, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&tol](auto ptr){return returnType(ptr->optim.hessian_numerical(tol));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__xb(SEXP xp, int type = 0){
  glmmrType model(xp,type);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.xb());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}
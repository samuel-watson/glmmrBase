#include <glmmr.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP ModelBits__new(SEXP formula_, SEXP data_, SEXP colnames_,
                    SEXP family_, SEXP link_, SEXP beta_,
                    SEXP theta_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  std::vector<double> beta = as<std::vector<double> >(beta_);
  std::vector<double> theta = as<std::vector<double> >(theta_);
  XPtr<bits>ptr(new bits(formula,data,colnames,family,link),true);
  ptr->linear_predictor.update_parameters(beta);
  ptr->covariance.update_parameters(theta);
  return ptr;
}

// [[Rcpp::export]]
void ModelBits__update_beta(SEXP xp, SEXP beta_){
  std::vector<double> beta = as<std::vector<double> >(beta_);
  XPtr<bits> ptr(xp);
  ptr->linear_predictor.update_parameters(beta);
}

// [[Rcpp::export]]
void ModelBits__update_theta(SEXP xp, SEXP theta_){
  std::vector<double> theta = as<std::vector<double> >(theta_);
  XPtr<bits> ptr(xp);
  ptr->covariance.update_parameters(theta);
}

// [[Rcpp::export]]
SEXP Model__new_w_pars(SEXP formula_, SEXP data_, SEXP colnames_,
                    SEXP family_, SEXP link_, SEXP beta_,
                    SEXP theta_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  std::vector<double> beta = as<std::vector<double> >(beta_);
  std::vector<double> theta = as<std::vector<double> >(theta_);
  XPtr<glmm> ptr(new glmm(formula,data,colnames,family,link),true);
  ptr->model.linear_predictor.update_parameters(beta);
  ptr->model.covariance.update_parameters(theta);
  return ptr;
}

// [[Rcpp::export]]
SEXP Model__new(SEXP formula_, SEXP data_, SEXP colnames_,
                       SEXP family_, SEXP link_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  XPtr<glmm> ptr(new glmm(formula,data,colnames,family,link),true);
  return ptr;
}

// [[Rcpp::export]]
SEXP Model_nngp__new(SEXP formula_, SEXP data_, SEXP colnames_,
                     SEXP family_, SEXP link_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  XPtr<glmm_nngp> ptr(new glmm_nngp(formula,data,colnames,family,link),true);
  return ptr;
}

// [[Rcpp::export]]
SEXP Model_nngp__new_w_pars(SEXP formula_, SEXP data_, SEXP colnames_,
                       SEXP family_, SEXP link_, SEXP beta_,
                       SEXP theta_, int nn){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  std::vector<double> beta = as<std::vector<double> >(beta_);
  std::vector<double> theta = as<std::vector<double> >(theta_);
  XPtr<glmm_nngp> ptr(new glmm_nngp(formula,data,colnames,family,link),true);
  ptr->model.linear_predictor.update_parameters(beta);
  ptr->model.covariance.gen_NN(nn);
  ptr->model.covariance.update_parameters(theta);
  return ptr;
}

// [[Rcpp::export]]
SEXP Model_hsgp__new(SEXP formula_, SEXP data_, SEXP colnames_,
                     SEXP family_, SEXP link_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  XPtr<glmm_hsgp> ptr(new glmm_hsgp(formula,data,colnames,family,link),true);
  return ptr;
}

// [[Rcpp::export]]
SEXP Model_hsgp__new_w_pars(SEXP formula_, SEXP data_, SEXP colnames_,
                            SEXP family_, SEXP link_, SEXP beta_,
                            SEXP theta_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  std::vector<double> beta = as<std::vector<double> >(beta_);
  std::vector<double> theta = as<std::vector<double> >(theta_);
  XPtr<glmm_hsgp> ptr(new glmm_hsgp(formula,data,colnames,family,link),true);
  ptr->model.linear_predictor.update_parameters(beta);
  ptr->model.covariance.update_parameters(theta);
  return ptr;
}

// [[Rcpp::export]]
void Model__set_y(SEXP xp, SEXP y_, int type = 0){
  Eigen::VectorXd y = as<Eigen::VectorXd>(y_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&y](auto ptr){ptr->set_y(y);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_offset(SEXP xp, SEXP offset_, int type = 0){
  Eigen::VectorXd offset = as<Eigen::VectorXd>(offset_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&offset](auto ptr){ptr->set_offset(offset);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_weights(SEXP xp, SEXP weights_, int type = 0){
  Eigen::ArrayXd weights = as<Eigen::ArrayXd>(weights_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&weights](auto ptr){ptr->set_weights(weights);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__P(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.P());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<int>(S));
}

// [[Rcpp::export]]
SEXP Model__Q(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.Q());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<int>(S));
}

// [[Rcpp::export]]
SEXP Model__theta_size(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.npar());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<int>(S));
}

// [[Rcpp::export]]
void Model__update_beta(SEXP xp, SEXP beta_, int type = 0){
  // removed check within this function for beta size - checked elsewhere
  std::vector<double> beta = as<std::vector<double> >(beta_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&beta](auto ptr){ptr->update_beta(beta);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__update_theta(SEXP xp, SEXP theta_, int type = 0){
  std::vector<double> theta = as<std::vector<double> >(theta_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&theta](auto ptr){ptr->update_theta(theta);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__update_u(SEXP xp, SEXP u_, int type = 0){
  Eigen::MatrixXd u = as<Eigen::MatrixXd>(u_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&u](auto ptr){ptr->update_u(u);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__use_attenuation(SEXP xp, SEXP use_, int type = 0){
  bool use = as<bool>(use_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&use](auto ptr){ptr->matrix.W.attenuated = use;}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__update_W(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->matrix.W.update();}
  };
  std::visit(functor,model.ptr);
}




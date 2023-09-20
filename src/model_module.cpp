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
SEXP Covariance__get_ptr_model(SEXP xp){
  XPtr<glmm> ptr(xp);
  XPtr<glmmr::Covariance> cptr(&(ptr->model.covariance));
  return cptr;
}

// [[Rcpp::export]]
SEXP LinearPredictor__get_ptr_model(SEXP xp){
  XPtr<glmm> ptr(xp);
  XPtr<glmmr::LinearPredictor> cptr(&(ptr->model.linear_predictor));
  return cptr;
}

// [[Rcpp::export]]
void Model__set_y(SEXP xp, SEXP y_, int type = 0){
  Eigen::VectorXd y = as<Eigen::VectorXd>(y_);
  switch(type){
    case 0:
      {
        XPtr<glmm> ptr(xp);
        ptr->set_y(y);
        break;
      }
    case 1:
      {
        XPtr<glmm_nngp> ptr(xp);
        ptr->set_y(y);
        break; 
      }  
  }
}

// [[Rcpp::export]]
void Model__set_offset(SEXP xp, SEXP offset_, int type = 0){
  Eigen::VectorXd offset = as<Eigen::VectorXd>(offset_);
  switch(type){
  case 0:
    {
      XPtr<glmm> ptr(xp);
      ptr->set_offset(offset);
      break;
    }
  case 1:
    {
      XPtr<glmm_nngp> ptr(xp);
      ptr->set_offset(offset);
      break;
    }
  }
}

// [[Rcpp::export]]
void Model__set_weights(SEXP xp, SEXP weights_, int type = 0){
  Eigen::ArrayXd weights = as<Eigen::ArrayXd>(weights_);
  switch(type){
  case 0:
    {
      XPtr<glmm> ptr(xp);
      ptr->set_weights(weights);
      break;
    }
  case 1:
    {
      XPtr<glmm_nngp> ptr(xp);
      ptr->set_weights(weights);
      break;
    }
  }
}

// [[Rcpp::export]]
SEXP Model__P(SEXP xp, int type = 0){
  int u;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  u = ptr->model.linear_predictor.P();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  u = ptr->model.linear_predictor.P();
  break;
}
  }
  return wrap(u);
}

// [[Rcpp::export]]
SEXP Model__Q(SEXP xp, int type = 0){
  int Q;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Q = ptr->model.covariance.Q();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Q = ptr->model.covariance.Q();
  break;
}
  }
  return wrap(Q);
}

// [[Rcpp::export]]
SEXP Model__theta_size(SEXP xp, int type = 0){
  int Q;
  switch(type){
  case 0:
{
  XPtr<glmm> ptr(xp);
  Q = ptr->model.covariance.npar();
  break;
}
  case 1:
{
  XPtr<glmm_nngp> ptr(xp);
  Q = ptr->model.covariance.npar();
  break;
}
  }
  return wrap(Q);
}

// [[Rcpp::export]]
void Model__update_beta(SEXP xp, SEXP beta_, int type = 0){
  std::vector<double> beta = as<std::vector<double> >(beta_);
  switch(type){
  case 0:
  {
    XPtr<glmm> ptr(xp);
    if(beta.size() != ptr->model.linear_predictor.P()){
      Rcpp::stop("Wrong number of beta parameters");
    } else {
      ptr->update_beta(beta);
    }
    break;
  }
  case 1:
  {
    XPtr<glmm_nngp> ptr(xp);
    if(beta.size() != ptr->model.linear_predictor.P()){
      Rcpp::stop("Wrong number of beta parameters");
    } else {
      ptr->update_beta(beta);
    }
    break;
  }
  }
}

// [[Rcpp::export]]
void Model__update_theta(SEXP xp, SEXP theta_, int type = 0){
  std::vector<double> theta = as<std::vector<double> >(theta_);
  switch(type){
  case 0:
  {
    XPtr<glmm> ptr(xp);
    if(theta.size() != ptr->model.covariance.npar()){
      Rcpp::stop("Wrong number of covariance parameters");
    } else {
      ptr->update_theta(theta);
    }
    break;
  }
  case 1:
  {
    XPtr<glmm_nngp> ptr(xp);
    if(theta.size() != ptr->model.covariance.npar()){
      Rcpp::stop("Wrong number of covariance parameters");
    } else {
      ptr->update_theta(theta);
    }
    break;
  }
  }
}

// [[Rcpp::export]]
void Model__update_u(SEXP xp, SEXP u_, int type = 0){
  Eigen::MatrixXd u = as<Eigen::MatrixXd>(u_);
  switch(type){
  case 0:
  {
    XPtr<glmm> ptr(xp);
    if(u.rows() != ptr->model.covariance.Q()){
      Rcpp::stop("Wrong number of random effects");
    } else {
      ptr->update_u(u);
    }
    break;
  }
  case 1:
  {
    XPtr<glmm_nngp> ptr(xp);
    if(u.rows() != ptr->model.covariance.Q()){
      Rcpp::stop("Wrong number of random effects");
    } else {
      ptr->update_u(u);
    }
    break;
  }
  }
}

// [[Rcpp::export]]
void Model__use_attenuation(SEXP xp, SEXP use_, int type = 0){
  bool use = as<bool>(use_);
  switch(type){
  case 0:
    {
      XPtr<glmm> ptr(xp);
      ptr->matrix.W.attenuated = use;
      break;
    }
  case 1:
    {
      XPtr<glmm_nngp> ptr(xp);
      ptr->matrix.W.attenuated = use;
      break;
    }
  }
  
}

// [[Rcpp::export]]
void Model__update_W(SEXP xp, int type = 0){
  switch(type){
  case 0:
    {
      XPtr<glmm> ptr(xp);
      ptr->matrix.W.update();
      break;
    }
  case 1:
    {
      XPtr<glmm_nngp> ptr(xp);
      ptr->matrix.W.update();
      break;
    }
  }
}




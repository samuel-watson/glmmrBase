#include <glmmr.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP Covariance__new(SEXP form_,SEXP data_, SEXP colnames_){
  std::string form = as<std::string>(form_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  XPtr<covariance> ptr(new covariance(form,data,colnames),true);
  return ptr;
}

// [[Rcpp::export]]
SEXP Covariance_nngp__new(SEXP form_,SEXP data_, SEXP colnames_){
  std::string form = as<std::string>(form_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  XPtr<nngp> ptr(new nngp(form,data,colnames),true);
  return ptr;
}

// [[Rcpp::export]]
SEXP Covariance_hsgp__new(SEXP form_,SEXP data_, SEXP colnames_){
  std::string form = as<std::string>(form_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  XPtr<hsgp> ptr(new hsgp(form,data,colnames),true);
  return ptr;
}

// [[Rcpp::export]]
SEXP Covariance__Z(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  switch(type){
  case GLMM:
{
  XPtr<covariance> ptr(xp);
  Eigen::MatrixXd Z = ptr->Z();
  return wrap(Z);
  break;
}
  case GLMM_NNGP:
{
  XPtr<nngp> ptr(xp);
  Eigen::MatrixXd Z = ptr->Z();
  return wrap(Z);
  break;
}
  case GLMM_HSGP:
{
  XPtr<hsgp> ptr(xp);
  Eigen::MatrixXd Z = ptr->Z();
  return wrap(Z);
  break;
}
    default:
    {
      Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(1,1);
      return wrap(Z);
      break;
    }
  }
}

// [[Rcpp::export]]
SEXP Covariance__ZL(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  switch(type){
  case GLMM:
{
  XPtr<covariance> ptr(xp);
  Eigen::MatrixXd Z = ptr->ZL();
  return wrap(Z);
  break;
}
  case GLMM_NNGP:
{
  XPtr<nngp> ptr(xp);
  Eigen::MatrixXd Z = ptr->ZL();
  return wrap(Z);
  break;
}
  case GLMM_HSGP:
{
  XPtr<hsgp> ptr(xp);
  Eigen::MatrixXd Z = ptr->ZL();
  return wrap(Z);
  break;
}
  default:
{
  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(1,1);
  return wrap(Z);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Covariance__LZWZL(SEXP xp, SEXP w_, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  Eigen::VectorXd w = as<Eigen::VectorXd>(w_);
  switch(type){
  case GLMM:
  {
    XPtr<covariance> ptr(xp);
    Eigen::MatrixXd Z = ptr->LZWZL(w);
    return wrap(Z);
    break;
  }
  case GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    Eigen::MatrixXd Z = ptr->LZWZL(w);
    return wrap(Z);
    break;
  }
  case GLMM_HSGP:
  {
    XPtr<hsgp> ptr(xp);
    Eigen::MatrixXd Z = ptr->LZWZL(w);
    return wrap(Z);
    break;
  }
  default:
  {
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(1,1);
    return wrap(Z);
    break;
  }
  }
}

// [[Rcpp::export]]
void Covariance__Update_parameters(SEXP xp, SEXP parameters_, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  std::vector<double> parameters = as<std::vector<double> >(parameters_);
  switch(type){
  case GLMM:
    {
      XPtr<covariance> ptr(xp);
      ptr->update_parameters_extern(parameters);
      break;
    }
  case GLMM_NNGP:
    {
      XPtr<nngp> ptr(xp);
      ptr->update_parameters_extern(parameters);
      break;
    }
  case GLMM_HSGP:
    {
      XPtr<hsgp> ptr(xp);
      ptr->update_parameters_extern(parameters);
      break;
    }
  }
}

// [[Rcpp::export]]
SEXP Covariance__D(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  switch(type){
  case GLMM:
{
  XPtr<covariance> ptr(xp);
  Eigen::MatrixXd D = ptr->D(false,false);
  return wrap(D);
  break;
}
  case GLMM_NNGP:
{
  XPtr<nngp> ptr(xp);
  Eigen::MatrixXd D = ptr->D(false,false);
  return wrap(D);
  break;
}
  case GLMM_HSGP:
{
  XPtr<hsgp> ptr(xp);
  Eigen::MatrixXd D = ptr->D(false,false);
  return wrap(D);
  break;
}
  default:
{
  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(1,1);
  return wrap(Z);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Covariance__D_chol(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  switch(type){
  case GLMM:
{
  XPtr<covariance> ptr(xp);
  Eigen::MatrixXd D = ptr->D(true,false);
  return wrap(D);
  break;
}
  case GLMM_NNGP:
{
  XPtr<nngp> ptr(xp);
  Eigen::MatrixXd D = ptr->D(true,false);
  return wrap(D);
  break;
}
  case GLMM_HSGP:
{
  XPtr<hsgp> ptr(xp);
  Eigen::MatrixXd D = ptr->D(true,false);
  return wrap(D);
  break;
}
  default:
{
  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(1,1);
  return wrap(Z);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Covariance__B(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  int B;
  switch(type){
  case GLMM:
  {
    XPtr<covariance> ptr(xp);
    B = ptr->B();
    break;
  }
  case GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    B = ptr->B();
    break;
  }
  case GLMM_HSGP:
  {
    XPtr<hsgp> ptr(xp);
    B = ptr->B();
    break;
  }
  }
  return wrap(B);
}

// [[Rcpp::export]]
SEXP Covariance__Q(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  int Q;
  switch(type){
    case GLMM:
      {
        XPtr<covariance> ptr(xp);
        Q = ptr->Q();
        break;
      }
  case GLMM_NNGP:
    {
      XPtr<nngp> ptr(xp);
      Q = ptr->Q();
      break;
    }
  case GLMM_HSGP:
    {
      XPtr<hsgp> ptr(xp);
      Q = ptr->Q();
      break;
    }
  }
  
  return wrap(Q);
}

// [[Rcpp::export]]
SEXP Covariance__log_likelihood(SEXP xp, SEXP u_, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  double ll;
  Eigen::VectorXd u = as<Eigen::VectorXd>(u_);
  switch(type){
  case GLMM:
    {
    XPtr<covariance> ptr(xp);
    ll = ptr->log_likelihood(u);
    break;
    }
  case GLMM_NNGP:
    {
    XPtr<nngp> ptr(xp);
    ll = ptr->log_likelihood(u);
    break;
    }
  case GLMM_HSGP:
    {
      XPtr<hsgp> ptr(xp);
      ll = ptr->log_likelihood(u);
      break;
    }
  }
  return wrap(ll);
}

// [[Rcpp::export]]
SEXP Covariance__log_determinant(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  double ll;
  switch(type){
  case GLMM:
    {
    XPtr<covariance> ptr(xp);
    ll = ptr->log_determinant();
    break;
    }
  case GLMM_NNGP:
    {
    XPtr<nngp> ptr(xp);
    ll = ptr->log_determinant();
    break;
    }
  case GLMM_HSGP:
    {
      XPtr<hsgp> ptr(xp);
      ll = ptr->log_determinant();
      break;
    }
  }
  return wrap(ll);
}

// [[Rcpp::export]]
SEXP Covariance__n_cov_pars(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  int G;
  switch(type){
  case GLMM:
  {
    XPtr<covariance> ptr(xp);
    G = ptr->npar();
    break;
  }
  case GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    G = ptr->npar();
    break;
  }
  case GLMM_HSGP:
  {
    XPtr<hsgp> ptr(xp);
    G = ptr->npar();
    break;
  }
  }
  return wrap(G);
}

// [[Rcpp::export]]
SEXP Covariance__simulate_re(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  switch(type){
  case GLMM:
{
  XPtr<covariance> ptr(xp);
  Eigen::VectorXd rr = ptr->sim_re();
  return wrap(rr);
  break;
}
  case GLMM_NNGP:
{
  XPtr<nngp> ptr(xp);
  Eigen::VectorXd rr = ptr->sim_re();
  return wrap(rr);
  break;
}
  case GLMM_HSGP:
{
  XPtr<hsgp> ptr(xp);
  Eigen::VectorXd rr = ptr->sim_re();
  return wrap(rr);
  break;
}
  default:
{
  Eigen::VectorXd Z = Eigen::VectorXd::Zero(1);
  return wrap(Z);
  break;
}
  }
}

// [[Rcpp::export]]
void Covariance__make_sparse(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  switch(type){
  case GLMM:
    {
      XPtr<covariance> ptr(xp);
      ptr->set_sparse(true);
      break;
    }
  case GLMM_NNGP:
    {
      XPtr<nngp> ptr(xp);
      ptr->set_sparse(true);
      break;
    }
  case GLMM_HSGP:
    {
      XPtr<hsgp> ptr(xp);
      ptr->set_sparse(true);
      break;
    }
  }
}

// [[Rcpp::export]]
void Covariance__make_dense(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  switch(type){
  case GLMM:
    {
      XPtr<covariance> ptr(xp);
      ptr->set_sparse(false);
      break;
    }
  case GLMM_NNGP:
    {
      XPtr<nngp> ptr(xp);
      ptr->set_sparse(false);
      break;
    }
  case GLMM_HSGP:
    {
      XPtr<hsgp> ptr(xp);
      ptr->set_sparse(false);
      break;
    }
  }
}

// [[Rcpp::export]]
void Covariance__set_nn(SEXP xp, int nn){
  XPtr<nngp> ptr(xp);
  ptr->grid.genNN(nn);
}


// [[Rcpp::export]]
SEXP Covariance__any_gr(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  bool gr;
  switch(type){
  case GLMM:
    {
      XPtr<covariance> ptr(xp);
      gr = ptr->any_group_re();
      break;
    }
  case GLMM_NNGP:
    {
      gr = false;
      break;
    }
  case GLMM_HSGP:
    {
      gr = false;
      break;
    }
  }
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__get_val(SEXP xp, int i, int j, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  double gr;
  switch(type){
  case GLMM:
  {
    XPtr<covariance> ptr(xp);
    gr = ptr->get_val(0,i,j);
    break;
  }
  case GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    gr = ptr->get_val(0,i,j);
    break;
  }
  case GLMM_HSGP:
  {
    XPtr<hsgp> ptr(xp);
    gr = ptr->get_val(0,i,j);
    break;
  }
  }
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__parameter_fn_index(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  std::vector<int> gr;
  switch(type){
  case GLMM:
    {
      XPtr<covariance> ptr(xp);
      gr = ptr->parameter_fn_index();
      break;
    }
  case GLMM_NNGP:
    {
      XPtr<nngp> ptr(xp);
      gr = ptr->parameter_fn_index();
      break;
    }
  case GLMM_HSGP:
    {
      XPtr<hsgp> ptr(xp);
      gr = ptr->parameter_fn_index();
      break;
    }
  }
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__re_terms(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  std::vector<std::string> gr;
  switch(type){
  case GLMM:
   { 
    XPtr<covariance> ptr(xp);
    gr = ptr->form_.re_terms();
    break;
    }
  case GLMM_NNGP:
    {
     XPtr<nngp> ptr(xp);
      gr = ptr->form_.re_terms();
      break;
    }
  case GLMM_HSGP:
    {
      XPtr<hsgp> ptr(xp);
      gr = ptr->form_.re_terms();
      break;
    }
  }
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__re_count(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  using enum Type;
  std::vector<int> gr;
  switch(type){
  case GLMM:
    {
      XPtr<covariance> ptr(xp);
      gr = ptr->re_count();
      break;
    }
  case GLMM_NNGP:
    {
      XPtr<nngp> ptr(xp);
      gr = ptr->re_count();
      break;
    }
  case GLMM_HSGP:
    {
      XPtr<hsgp> ptr(xp);
      gr = ptr->re_count();
      break;
    }
  }
  
  return wrap(gr);
}
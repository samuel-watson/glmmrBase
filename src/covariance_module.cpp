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
SEXP Covariance__Z(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<covariance> ptr(xp);
  Eigen::MatrixXd Z = ptr->Z();
  return wrap(Z);
  break;
}
  case 1:
{
  XPtr<nngp> ptr(xp);
  Eigen::MatrixXd Z = ptr->Z();
  return wrap(Z);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Covariance__ZL(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<covariance> ptr(xp);
  Eigen::MatrixXd Z = ptr->ZL();
  return wrap(Z);
  break;
}
  case 1:
{
  XPtr<nngp> ptr(xp);
  Eigen::MatrixXd Z = ptr->ZL();
  return wrap(Z);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Covariance__LZWZL(SEXP xp, SEXP w_, int type = 0){
  Eigen::VectorXd w = as<Eigen::VectorXd>(w_);
  switch(type){
  case 0:
  {
    XPtr<covariance> ptr(xp);
    Eigen::MatrixXd Z = ptr->LZWZL(w);
    return wrap(Z);
    break;
  }
  case 1:
  {
    XPtr<nngp> ptr(xp);
    Eigen::MatrixXd Z = ptr->LZWZL(w);
    return wrap(Z);
    break;
  }
  }
}

// [[Rcpp::export]]
void Covariance__Update_parameters(SEXP xp, SEXP parameters_, int type = 0){
  std::vector<double> parameters = as<std::vector<double> >(parameters_);
  switch(type){
  case 0:
    {
      XPtr<covariance> ptr(xp);
      ptr->update_parameters_extern(parameters);
      break;
    }
  case 1:
    {
      XPtr<nngp> ptr(xp);
      ptr->update_parameters_extern(parameters);
      break;
    }
  }
}

// [[Rcpp::export]]
SEXP Covariance__D(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<covariance> ptr(xp);
  Eigen::MatrixXd D = ptr->D(false,false);
  return wrap(D);
  break;
}
  case 1:
{
  XPtr<nngp> ptr(xp);
  Eigen::MatrixXd D = ptr->D(false,false);
  return wrap(D);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Covariance__D_chol(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<covariance> ptr(xp);
  Eigen::MatrixXd D = ptr->D(true,false);
  return wrap(D);
  break;
}
  case 1:
{
  XPtr<nngp> ptr(xp);
  Eigen::MatrixXd D = ptr->D(true,false);
  return wrap(D);
  break;
}
  }
}

// [[Rcpp::export]]
SEXP Covariance__B(SEXP xp, int type = 0){
  int B;
  switch(type){
  case 0:
  {
    XPtr<covariance> ptr(xp);
    B = ptr->B();
    break;
  }
  case 1:
  {
    XPtr<nngp> ptr(xp);
    B = ptr->B();
    break;
  }
  }
  return wrap(B);
}

// [[Rcpp::export]]
SEXP Covariance__Q(SEXP xp, int type = 0){
  int Q;
  switch(type){
    case 0:
      {
        XPtr<covariance> ptr(xp);
        Q = ptr->Q();
        break;
      }
  case 1:
    {
      XPtr<nngp> ptr(xp);
      Q = ptr->Q();
      break;
    }
  }
  
  return wrap(Q);
}

// [[Rcpp::export]]
SEXP Covariance__log_likelihood(SEXP xp, SEXP u_, int type = 0){
  double ll;
  Eigen::VectorXd u = as<Eigen::VectorXd>(u_);
  switch(type){
  case 0:
    {
    XPtr<covariance> ptr(xp);
    ll = ptr->log_likelihood(u);
    break;
    }
  case 1:
    {
    XPtr<nngp> ptr(xp);
    ll = ptr->log_likelihood(u);
    break;
    }
  }
  return wrap(ll);
}

// [[Rcpp::export]]
SEXP Covariance__log_determinant(SEXP xp, int type = 0){
  double ll;
  switch(type){
  case 0:
    {
    XPtr<covariance> ptr(xp);
    ll = ptr->log_determinant();
    break;
    }
  case 1:
    {
    XPtr<nngp> ptr(xp);
    ll = ptr->log_determinant();
    break;
    }
  }
  return wrap(ll);
}

// [[Rcpp::export]]
SEXP Covariance__n_cov_pars(SEXP xp, int type = 0){
  int G;
  switch(type){
  case 0:
  {
    XPtr<covariance> ptr(xp);
    G = ptr->npar();
    break;
  }
  case 1:
  {
    XPtr<covariance> ptr(xp);
    G = ptr->npar();
    break;
  }
  }
  return wrap(G);
}

// [[Rcpp::export]]
SEXP Covariance__simulate_re(SEXP xp, int type = 0){
  switch(type){
  case 0:
{
  XPtr<covariance> ptr(xp);
  Eigen::VectorXd rr = ptr->sim_re();
  return wrap(rr);
  break;
}
  case 1:
{
  XPtr<nngp> ptr(xp);
  Eigen::VectorXd rr = ptr->sim_re();
  return wrap(rr);
  break;
}
  }
}

// [[Rcpp::export]]
void Covariance__make_sparse(SEXP xp, int type = 0){
  switch(type){
  case 0:
    {
      XPtr<covariance> ptr(xp);
      ptr->set_sparse(true);
      break;
    }
  case 1:
    {
      XPtr<nngp> ptr(xp);
      ptr->set_sparse(true);
      break;
    }
  }
}

// [[Rcpp::export]]
void Covariance__make_dense(SEXP xp, int type = 0){
  switch(type){
  case 0:
    {
      XPtr<covariance> ptr(xp);
      ptr->set_sparse(false);
      break;
    }
  case 1:
    {
      XPtr<nngp> ptr(xp);
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
SEXP Covariance__any_gr(SEXP xp, int type = 0){
  bool gr;
  switch(type){
  case 0:
    {
      XPtr<covariance> ptr(xp);
      gr = ptr->any_group_re();
      break;
    }
  case 1:
    {
      XPtr<nngp> ptr(xp);
      gr = ptr->any_group_re();
      break;
    }
  }
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__get_val(SEXP xp, int i, int j, int type = 0){
  double gr;
  switch(type){
  case 0:
  {
    XPtr<covariance> ptr(xp);
    gr = ptr->get_val(0,i,j);
    break;
  }
  case 1:
  {
    XPtr<nngp> ptr(xp);
    gr = ptr->get_val(0,i,j);
    break;
  }
  }
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__parameter_fn_index(SEXP xp, int type = 0){
  std::vector<int> gr;
  switch(type){
  case 0:
    {
      XPtr<covariance> ptr(xp);
      gr = ptr->parameter_fn_index();
      break;
    }
  case 1:
    {
      XPtr<nngp> ptr(xp);
      gr = ptr->parameter_fn_index();
      break;
    }
  }
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__re_terms(SEXP xp, int type = 0){
  std::vector<std::string> gr;
  switch(type){
  case 0:
   { 
    XPtr<covariance> ptr(xp);
    gr = ptr->form_.re_terms();
    break;
    }
  case 1:
    {
     XPtr<nngp> ptr(xp);
      gr = ptr->form_.re_terms();
      break;
    }
  }
  return wrap(gr);
}

// [[Rcpp::export]]
SEXP Covariance__re_count(SEXP xp, int type = 0){
  std::vector<int> gr;
  switch(type){
  case 0:
    {
      XPtr<covariance> ptr(xp);
      gr = ptr->re_count();
      break;
    }
  case 1:
    {
      XPtr<nngp> ptr(xp);
      gr = ptr->re_count();
      break;
    }
  }
  
  return wrap(gr);
}
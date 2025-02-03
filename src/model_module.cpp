#include <glmmr.h>

namespace Rcpp {
template<>
SEXP wrap(const VectorMatrix& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vec") = Rcpp::wrap(x.vec),
      Rcpp::Named("mat") = Rcpp::wrap(x.mat)
  ));
}

template<>
SEXP wrap(const MatrixMatrix& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("mat1") = Rcpp::wrap(x.mat1),
      Rcpp::Named("mat2") = Rcpp::wrap(x.mat2),
      Rcpp::Named("a") = Rcpp::wrap(x.a),
      Rcpp::Named("b") = Rcpp::wrap(x.b)
  ));
}

template<typename T1, typename T2> SEXP wrap( const std::pair<T1,T2>& _v ) {
  return Rcpp::List::create(
    Rcpp::Named("first")  = Rcpp::wrap<T1>( _v.first ),
    Rcpp::Named("second") = Rcpp::wrap<T2>( _v.second )
  );
};

template<glmmr::SE corr>
SEXP wrap(const CorrectionData<corr>& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vcov_beta") = Rcpp::wrap(x.vcov_beta),
      Rcpp::Named("vcov_theta") = Rcpp::wrap(x.vcov_theta),
      Rcpp::Named("dof") = Rcpp::wrap(x.dof)
  ));
}

template<>
SEXP wrap(const CorrectionData<glmmr::SE::KRBoth>& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vcov_beta") = Rcpp::wrap(x.vcov_beta),
      Rcpp::Named("vcov_beta_second") = Rcpp::wrap(x.vcov_beta_second),
      Rcpp::Named("vcov_theta") = Rcpp::wrap(x.vcov_theta),
      Rcpp::Named("dof") = Rcpp::wrap(x.dof)
  ));
}

template<>
SEXP wrap(const BoxResults& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("dof") = Rcpp::wrap(x.dof),
      Rcpp::Named("scale") = Rcpp::wrap(x.scale),
      Rcpp::Named("test_stat") = Rcpp::wrap(x.test_stat),
      Rcpp::Named("p_value") = Rcpp::wrap(x.p_value)
  ));
}


}

using namespace Rcpp;


// [[Rcpp::export]]
SEXP Linpred__new(SEXP formula_,
                  SEXP data_,
                  SEXP colnames_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  glmmr::Formula f1(formula);
  XPtr<glmmr::LinearPredictor> ptr(new glmmr::LinearPredictor(f1,data,colnames));
  return ptr;
}

// [[Rcpp::export]]
void Linpred__update_pars(SEXP xp,
                          SEXP parameters_){
  std::vector<double> parameters = as<std::vector<double>>(parameters_);
  XPtr<glmmr::LinearPredictor> ptr(xp);
  ptr->update_parameters(parameters);
}

// [[Rcpp::export]]
SEXP Linpred__xb(SEXP xp){
  XPtr<glmmr::LinearPredictor> ptr(xp);
  Eigen::VectorXd xb = ptr->xb();
  return wrap(xb);
}

// [[Rcpp::export]]
SEXP Linpred__x(SEXP xp){
  XPtr<glmmr::LinearPredictor> ptr(xp);
  Eigen::MatrixXd X = ptr->X();
  return wrap(X);
}

// [[Rcpp::export]]
SEXP Linpred__beta_names(SEXP xp){
  XPtr<glmmr::LinearPredictor> ptr(xp);
  std::vector<std::string> X = ptr->parameter_names();
  return wrap(X);
}

// [[Rcpp::export]]
SEXP Linpred__any_nonlinear(SEXP xp){
  XPtr<glmmr::LinearPredictor> ptr(xp);
  bool anl = ptr->any_nonlinear();
  return wrap(anl);
}


// [[Rcpp::export]]
SEXP Covariance__new(SEXP form_,SEXP data_, SEXP colnames_){
  std::string form = as<std::string>(form_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  glmmr::Formula f1(form);
  XPtr<covariance> ptr(new covariance(f1,data,colnames),true);
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
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    Eigen::MatrixXd Z = ptr->Z();
    return wrap(Z);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    Eigen::MatrixXd Z = ptr->Z();
    return wrap(Z);
    break;
  }
  case Type::GLMM_HSGP:
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
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    Eigen::MatrixXd Z = ptr->ZL();
    return wrap(Z);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    Eigen::MatrixXd Z = ptr->ZL();
    return wrap(Z);
    break;
  }
  case Type::GLMM_HSGP:
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
  Eigen::VectorXd w = as<Eigen::VectorXd>(w_);
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    Eigen::MatrixXd Z = ptr->LZWZL(w);
    return wrap(Z);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    Eigen::MatrixXd Z = ptr->LZWZL(w);
    return wrap(Z);
    break;
  }
  case Type::GLMM_HSGP:
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
  std::vector<double> parameters = as<std::vector<double> >(parameters_);
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    ptr->update_parameters_extern(parameters);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    ptr->update_parameters_extern(parameters);
    break;
  }
  case Type::GLMM_HSGP:
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
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    Eigen::MatrixXd D = ptr->D(false,false);
    return wrap(D);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    Eigen::MatrixXd D = ptr->D(false,false);
    return wrap(D);
    break;
  }
  case Type::GLMM_HSGP:
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
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    Eigen::MatrixXd D = ptr->D(true,false);
    return wrap(D);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    Eigen::MatrixXd D = ptr->D(true,false);
    return wrap(D);
    break;
  }
  case Type::GLMM_HSGP:
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
  int B;
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    B = ptr->B();
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    B = ptr->B();
    break;
  }
  case Type::GLMM_HSGP:
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
  int Q;
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    Q = ptr->Q();
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    Q = ptr->Q();
    break;
  }
  case Type::GLMM_HSGP:
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
  double ll;
  Eigen::VectorXd u = as<Eigen::VectorXd>(u_);
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    ll = ptr->log_likelihood(u);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    ll = ptr->log_likelihood(u);
    break;
  }
  case Type::GLMM_HSGP:
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
  double ll;
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    ll = ptr->log_determinant();
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    ll = ptr->log_determinant();
    break;
  }
  case Type::GLMM_HSGP:
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
  int G;
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    G = ptr->npar();
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    G = ptr->npar();
    break;
  }
  case Type::GLMM_HSGP:
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
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    Eigen::VectorXd rr = ptr->sim_re();
    return wrap(rr);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    Eigen::VectorXd rr = ptr->sim_re();
    return wrap(rr);
    break;
  }
  case Type::GLMM_HSGP:
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
void Covariance__make_sparse(SEXP xp, bool amd = true, int type_ = 0){
  Type type = static_cast<Type>(type_);
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    ptr->set_sparse(true, amd);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    ptr->set_sparse(true, amd);
    break;
  }
  case Type::GLMM_HSGP:
  {
    XPtr<hsgp> ptr(xp);
    ptr->set_sparse(true, amd);
    break;
  }
  }
}

// [[Rcpp::export]]
void Covariance__make_dense(SEXP xp, int type_ = 0){
  Type type = static_cast<Type>(type_);
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    ptr->set_sparse(false);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    ptr->set_sparse(false);
    break;
  }
  case Type::GLMM_HSGP:
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
  bool gr;
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    gr = ptr->any_group_re();
    break;
  }
  case Type::GLMM_NNGP:
  {
    gr = false;
    break;
  }
  case Type::GLMM_HSGP:
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
  double gr;
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    gr = ptr->get_val(0,i,j);
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    gr = ptr->get_val(0,i,j);
    break;
  }
  case Type::GLMM_HSGP:
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
  std::vector<int> gr;
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    gr = ptr->parameter_fn_index();
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    gr = ptr->parameter_fn_index();
    break;
  }
  case Type::GLMM_HSGP:
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
  std::vector<std::string> gr;
  switch(type){
  case Type::GLMM:
  { 
    XPtr<covariance> ptr(xp);
    gr = ptr->form_.re_terms();
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    gr = ptr->form_.re_terms();
    break;
  }
  case Type::GLMM_HSGP:
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
  std::vector<int> gr;
  switch(type){
  case Type::GLMM:
  {
    XPtr<covariance> ptr(xp);
    gr = ptr->re_count();
    break;
  }
  case Type::GLMM_NNGP:
  {
    XPtr<nngp> ptr(xp);
    gr = ptr->re_count();
    break;
  }
  case Type::GLMM_HSGP:
  {
    XPtr<hsgp> ptr(xp);
    gr = ptr->re_count();
    break;
  }
  }
  
  return wrap(gr);
}

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
  int covnpar = ptr->covariance.npar();
  std::vector<double> cov_pars(covnpar);
  if(theta.size() != covnpar){
    for(int i = 0; i < covnpar; i++) cov_pars[i] = Rcpp::runif(1)(0);
  } else {
    cov_pars = theta;
  }
  ptr->covariance.update_parameters(cov_pars);
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
                     SEXP family_, SEXP link_, int nn){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data = as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  std::string family = as<std::string>(family_);
  std::string link = as<std::string>(link_);
  XPtr<glmm_nngp> ptr(new glmm_nngp(formula,data,colnames,family,link),true);
  ptr->model.covariance.gen_NN(nn);
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

// // [[Rcpp::export]]
// SEXP Model__P(SEXP xp, int type = 0){
//   auto functor = overloaded {
//     [](int) {  return returnType(0);},
//     [](auto ptr){return returnType(ptr->model.linear_predictor.P());}
//   };
//   Fn<int> func(xp,type);
//   return wrap(func(functor));
// }

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
void Model__update_u(SEXP xp, SEXP u_, bool append = false, int type = 0){
  Eigen::MatrixXd u = as<Eigen::MatrixXd>(u_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->update_u(u, append);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_quantile(SEXP xp, double q, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&q](auto ptr){ptr->model.family.set_quantile(q);}
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


// [[Rcpp::export]]
SEXP Model__get_W(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.W.W());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}

// [[Rcpp::export]]
void Model__set_direct_control(SEXP xp, bool direct = false, double direct_range_beta = 3.0, int max_iter = 100, double epsilon = 1e-4, bool select_one = true, bool trisect_once = false, 
                               int max_eval = 0, bool mrdirect = false, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->optim.set_direct_control(direct, direct_range_beta, max_iter, epsilon, select_one, trisect_once, max_eval, mrdirect);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_lbfgs_control(SEXP xp, double g_epsilon = 1e-8, int past = 3, double delta = 1e-8, int max_linesearch = 64, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->optim.set_lbfgs_control(g_epsilon, past, delta, max_linesearch);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__use_reml(SEXP xp, bool reml = true, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&reml](auto ptr){ptr->optim.use_reml(reml);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_bound(SEXP xp, SEXP bound_, bool beta = true, bool lower = true, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  std::vector<double> bound = as<std::vector<double> >(bound_);
  if(beta){
    auto functor = overloaded {
      [](int) {}, 
      [&](auto ptr){ptr->optim.set_bound(bound,lower);}
    };
    std::visit(functor,model.ptr);
  } else {
    auto functor = overloaded {
      [](int) {}, 
      [&](auto ptr){ptr->optim.set_theta_bound(bound,lower);}
    };
    std::visit(functor,model.ptr);
  }
  
}

// [[Rcpp::export]]
void Model__print_instructions(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor1 = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->model.linear_predictor.calc.print_instructions();}
  };
  // auto functor3 = overloaded {
  //   [](int) {}, 
  //   [](auto ptr){ptr->model.calc.print_instructions();}
  // };
  Rcpp::Rcout << "\nLINEAR PREDICTOR:\n";
  std::visit(functor1,model.ptr);
  // if(loglik){
  //   Rcpp::Rcout << "\nLOG-LIKELIHOOD:\n";
  //   std::visit(functor3,model.ptr);
  // }
}

// [[Rcpp::export]]
SEXP Model__log_prob(SEXP xp, SEXP v_, int type = 0){
  Eigen::VectorXd v = as<Eigen::VectorXd>(v_);
  glmmrType model(xp,static_cast<Type>(type));
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
  glmmrType model(xp,static_cast<Type>(type));
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
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->matrix.log_gradient(v,beta));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}

// [[Rcpp::export]]
SEXP Model__linear_predictor(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.linpred());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__log_likelihood(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.log_likelihood());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
SEXP Model__n_cov_pars(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.npar());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<int>(S));
}

// [[Rcpp::export]]
SEXP Model__Z(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.Z());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__Z_needs_updating(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.z_requires_update);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<bool>(S));
}

// [[Rcpp::export]]
void Model__cov_set_nn(SEXP xp, int nn){
  XPtr<glmm_nngp> ptr(xp);
  ptr->model.covariance.gen_NN(nn);
}

// [[Rcpp::export]]
void Model__test_lbfgs(SEXP xp, SEXP x){
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd start = as<Eigen::VectorXd>(x);
  Eigen::VectorXd grad(start.size());
  grad.setZero();
  double ll = ptr->optim.log_likelihood_beta_with_gradient(start,grad);
  Rcpp::Rcout << "\nStart: " << start.transpose();
  Rcpp::Rcout << "\nGradient: " << grad.transpose();
  Rcpp::Rcout << "\nLog likelihood: " << ll;
}

// [[Rcpp::export]]
void Model__test_lbfgs_theta(SEXP xp, SEXP x){
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd start = as<Eigen::VectorXd>(x);
  Eigen::VectorXd grad(start.size());
  grad.setZero();
  if(ptr->re.scaled_u_.cols() != ptr->re.u_.cols())ptr->re.scaled_u_.conservativeResize(NoChange,ptr->re.u_.cols());
  ptr->re.scaled_u_ = ptr->model.covariance.Lu(ptr->re.u_);  
  double ll = ptr->optim.log_likelihood_theta_with_gradient(start,grad);
  Rcpp::Rcout << "\nStart: " << start.transpose();
  Rcpp::Rcout << "\nGradient: " << grad.transpose();
  Rcpp::Rcout << "\nLog likelihood: " << ll;
}

// [[Rcpp::export]]
void Model__test_lbfgs_laplace(SEXP xp, SEXP x){
  XPtr<glmm> ptr(xp);
  Eigen::VectorXd start = as<Eigen::VectorXd>(x);
  Eigen::VectorXd grad(start.size());
  grad.setZero();
  if(ptr->re.scaled_u_.cols() != ptr->re.u_.cols())ptr->re.scaled_u_.conservativeResize(NoChange,ptr->re.u_.cols());
  ptr->re.scaled_u_ = ptr->model.covariance.Lu(ptr->re.u_);
  double ll = ptr->optim.log_likelihood_laplace_beta_u_with_gradient(start,grad);
  Rcpp::Rcout << "\nStart: " << start.transpose();
  Rcpp::Rcout << "\nGradient: " << grad.transpose();
  Rcpp::Rcout << "\nLog likelihood: " << ll;
}

// [[Rcpp::export]]
void Model__ml_beta(SEXP xp, int algo = 0, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&algo](auto ptr){
      switch(algo){
      case 1:
        ptr->optim.template ml_beta<NEWUOA>();
        break;
      case 2:
        ptr->optim.template ml_beta<LBFGS>();
        break;
      case 3:
        ptr->optim.template ml_beta<DIRECT>();
        break;
      default:
        ptr->optim.template ml_beta<BOBYQA>();
      break;
      }
    }
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__ml_theta(SEXP xp, int algo = 0, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&algo](auto ptr){
      switch(algo){
      case 1:
        ptr->optim.template ml_theta<NEWUOA>();
        break;
      case 2:
        ptr->optim.template ml_theta<LBFGS>();
        break;
      case 3:
        ptr->optim.template ml_theta<DIRECT>();
        break;
      default:
        ptr->optim.template ml_theta<BOBYQA>();
      break;
      }
    }
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__ml_all(SEXP xp, int algo = 0, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&algo](auto ptr){
      switch(algo){
      case 1:
        ptr->optim.template ml_all<NEWUOA>();
        break;
      case 2:
        Rcpp::stop("L-BGFS not available for full likelihood beta-theta joint optimisation.");
        break;
      case 3:
        ptr->optim.template ml_all<DIRECT>();
        break;
      default:
        ptr->optim.template ml_all<BOBYQA>();
      break;
      }
    }
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__laplace_ml_beta_u(SEXP xp, int algo = 0, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&algo](auto ptr){
      switch(algo){
      case 1:
        ptr->optim.template laplace_ml_beta_u<NEWUOA>();
        break;
      case 2:
        ptr->optim.template laplace_ml_beta_u<LBFGS>();
        break;
      case 3:
        ptr->optim.template laplace_ml_beta_u<DIRECT>();
        break;
      default:
        ptr->optim.template laplace_ml_beta_u<BOBYQA>();
      break;
      }
    }
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__laplace_ml_theta(SEXP xp, int algo = 0, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&algo](auto ptr){
      switch(algo){
      case 1:
        ptr->optim.template laplace_ml_theta<NEWUOA>();
        break;
      case 2:
        ptr->optim.template laplace_ml_theta<LBFGS>();
        break;
      case 3:
        ptr->optim.template laplace_ml_theta<DIRECT>();
        break;
      default:
        ptr->optim.template laplace_ml_theta<BOBYQA>();
      break;
      }
    }
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__laplace_ml_beta_theta(SEXP xp, int algo = 0, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&algo](auto ptr){
      switch(algo){
      case 1:
        ptr->optim.template laplace_ml_beta_theta<NEWUOA>();
        break;
      case 2:
        Rcpp::stop("L-BGFS(-B) is not available for Laplace beta-theta optimisation");
        break;
      case 3:
        ptr->optim.template laplace_ml_beta_theta<DIRECT>();
        break;
      default:
        ptr->optim.template laplace_ml_beta_theta<BOBYQA>();
      break;
      }
    }
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__nr_beta(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.nr_beta();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__laplace_nr_beta_u(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.laplace_nr_beta_u();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__laplace_beta_u(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.laplace_beta_u();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__Sigma(SEXP xp, bool inverse, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&inverse](auto ptr){return returnType(ptr->matrix.Sigma(inverse));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__information_matrix(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.information_matrix());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__D(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.D(false,false));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__D_chol(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.D(true,false));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__u_log_likelihood(SEXP xp, SEXP u_, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  Eigen::VectorXd u = as<Eigen::VectorXd>(u_);
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&u](auto ptr){return returnType(ptr->model.covariance.log_likelihood(u));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
SEXP Model__simulate_re(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.sim_re());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}

// [[Rcpp::export]]
SEXP Model__re_terms(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.form_.re_terms());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<strvec>(S));
}

// [[Rcpp::export]]
SEXP Model__re_count(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.re_count());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<intvec>(S));
}

// [[Rcpp::export]]
SEXP Model__parameter_fn_index(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.parameter_fn_index());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<intvec>(S));
}

// [[Rcpp::export]]
SEXP Model__information_matrix_crude(SEXP xp, int type = 2){
  glmmrType model(xp,static_cast<Type>(type));
  auto functorS = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.Sigma(false));}
  };
  auto functorX = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.X());}
  };
  auto S = std::visit(functorS,model.ptr);
  auto X = std::visit(functorX,model.ptr);
  Eigen::MatrixXd Sigma = std::get<Eigen::MatrixXd>(S);
  Eigen::MatrixXd Xmat = std::get<Eigen::MatrixXd>(X);
  Eigen::MatrixXd SigmaInv = Sigma.llt().solve(Eigen::MatrixXd::Identity(Sigma.rows(),Sigma.cols()));
  Eigen::MatrixXd M = Xmat.transpose() * SigmaInv * Xmat;
  return wrap(M);
}

// [[Rcpp::export]]
SEXP Model__obs_information_matrix(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.template observed_information_matrix<glmmr::IM::EIM>());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__observed_information_matrix(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.template observed_information_matrix<glmmr::IM::OIM>());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__u(SEXP xp, bool scaled_, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&scaled_](auto ptr){return returnType(ptr->re.u(scaled_));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__Zu(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->re.Zu());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__X(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
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
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->mcmc.mcmc_sample(warmup,samples,adapt);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_trace(SEXP xp, SEXP trace_, int type = 0){
  int trace = as<int>(trace_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&trace](auto ptr){ptr->set_trace(trace);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__get_beta(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.parameter_vector());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}

// [[Rcpp::export]]
SEXP Model__y(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.data.y);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::VectorXd>(S));
}

// [[Rcpp::export]]
SEXP Model__get_theta(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.parameters_);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<double> >(S));
}

// [[Rcpp::export]]
SEXP Model__get_var_par(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.data.var_par);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
SEXP Model__get_variance(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
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
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&var_par](auto ptr){ptr->model.data.set_var_par(var_par);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_trials(SEXP xp, SEXP trials, int type = 0){
  Eigen::ArrayXd var_par = as<Eigen::ArrayXd>(trials);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&var_par](auto ptr){ptr->model.data.set_variance(var_par);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__L(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.D(true,false));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__ZL(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.ZL());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__xb(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.xb());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::ArrayXd>(S));
}


// [[Rcpp::export]]
SEXP near_semi_pd(SEXP mat_){
  Eigen::MatrixXd mat = as<Eigen::MatrixXd>(mat_);
  glmmr::Eigen_ext::near_semi_pd(mat);
  return wrap(mat);
}

// [[Rcpp::export]]
SEXP Covariance__submatrix(SEXP xp, int i){
  XPtr<nngp> ptr(xp);
  VectorMatrix result = ptr->submatrix(i);
  return wrap(result);
}

// [[Rcpp::export]]
void Model_hsgp__set_approx_pars(SEXP xp, SEXP m_, SEXP L_){
  std::vector<int> m = as<std::vector<int> >(m_);
  Eigen::ArrayXd L = as<Eigen::ArrayXd>(L_);
  XPtr<glmm_hsgp> ptr(xp);
  ptr->model.covariance.update_approx_parameters(m,L);
  ptr->reset_u();
  std::vector<double> theta = ptr->model.covariance.parameters_;
  ptr->model.covariance.update_parameters(theta);
}

// [[Rcpp::export]]
void Covariance_hsgp__set_approx_pars(SEXP xp, SEXP m_, SEXP L_){
  std::vector<int> m = as<std::vector<int> >(m_);
  Eigen::ArrayXd L = as<Eigen::ArrayXd>(L_);
  XPtr<hsgp> ptr(xp);
  ptr->update_approx_parameters(m,L);
}

// [[Rcpp::export]]
SEXP Model_hsgp__dim(SEXP xp){
  XPtr<glmm_hsgp> ptr(xp);
  int dim = ptr->model.covariance.dim;
  return wrap(dim);
}

// [[Rcpp::export]]
SEXP Model__aic(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.aic());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
SEXP Model__residuals(SEXP xp, int rtype = 2, bool conditional = true, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->matrix.residuals(rtype,conditional));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__get_log_likelihood_values(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.current_likelihood_values());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::pair<double,double> >(S));
}

// [[Rcpp::export]]
SEXP Model__u_diagnostic(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.u_diagnostic());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::pair<double,double> >(S));
}

// MarginType type, dydx, diff, ratio
// [[Rcpp::export]]
SEXP Model__marginal(SEXP xp, 
                     std::string x,
                     int margin = 0,
                     int re = 3,
                     int se = 0,
                     int oim = 0,
                     Nullable<std::vector<std::string> > at = R_NilValue,
                     Nullable<std::vector<std::string> > atmeans = R_NilValue,
                     Nullable<std::vector<std::string> > average = R_NilValue,
                     double xvals_first = 1,
                     double xvals_second = 0,
                     Nullable<std::vector<double> > atvals = R_NilValue,
                     Nullable<std::vector<double> > revals = R_NilValue,
                     int type = 0){
  
  glmmrType model(xp,static_cast<Type>(type));
  std::vector<std::string> atvar;
  std::vector<std::string> atmeansvar;
  std::vector<std::string> averagevar;
  std::vector<double> atxvals;
  std::vector<double> atrevals;
  if(at.isNotNull())atvar = as<std::vector<std::string> >(at);
  if(atmeans.isNotNull())atmeansvar = as<std::vector<std::string> >(atmeans);
  if(average.isNotNull())averagevar = as<std::vector<std::string> >(average);
  std::pair<double, double> xvals;
  xvals.first = xvals_first;
  xvals.second = xvals_second;
  if(atvals.isNotNull())atxvals = as<std::vector<double> >(atvals);
  if(revals.isNotNull())atrevals = as<std::vector<double> >(revals);
  
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->marginal(static_cast<glmmr::MarginType>(margin),
                                    x,atvar,atmeansvar,averagevar,
                                    static_cast<glmmr::RandomEffectMargin>(re),
                                    static_cast<glmmr::SE>(se),
                                    static_cast<glmmr::IM>(oim),
                                    xvals,atxvals,atrevals
    ));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::pair<double,double> >(S));
}

// [[Rcpp::export]]
void Model__mcmc_set_lambda(SEXP xp, SEXP lambda_, int type = 0){
  double lambda = as<double>(lambda_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&lambda](auto ptr){ptr->mcmc.mcmc_set_lambda(lambda);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__reset_fn_counter(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.reset_fn_counter();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__get_fn_counter(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.fn_counter);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::pair<int,int> >(S));
}

// [[Rcpp::export]]
void Model__print_names(SEXP xp, bool data, bool parameters, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->model.linear_predictor.calc.print_names(data,parameters);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__mcmc_set_max_steps(SEXP xp, SEXP max_steps_, int type = 0){
  int max_steps = as<int>(max_steps_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&max_steps](auto ptr){ptr->mcmc.mcmc_set_max_steps(max_steps);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_sml_parameters(SEXP xp, bool saem_, int block_size = 20, double alpha = 0.8, bool pr_average = true, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){
      ptr->optim.control.saem = saem_;
      ptr->optim.control.alpha = alpha;
      ptr->re.mcmc_block_size = block_size;
      ptr->optim.control.pr_average = pr_average;
      if(!saem_){
        ptr->optim.ll_current.resize(block_size,NoChange);
        // ptr->optim.ll_previous.resize(block_size,NoChange);
      }
    }
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__ll_diff_variance(SEXP xp, bool beta, bool theta, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {return returnType(0);}, 
    [&](auto ptr){
      return returnType(ptr->optim.ll_diff_variance(beta,theta));
    }
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
void Model__mcmc_set_refresh(SEXP xp, SEXP refresh_, int type = 0){
  int refresh = as<int>(refresh_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&refresh](auto ptr){ptr->mcmc.mcmc_set_refresh(refresh);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__mcmc_set_target_accept(SEXP xp, SEXP target_, int type = 0){
  double target = as<double>(target_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&target](auto ptr){ptr->mcmc.mcmc_set_target_accept(target);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__make_sparse(SEXP xp, bool amd = true, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->model.make_covariance_sparse(amd);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__make_dense(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->model.make_covariance_dense();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__beta_parameter_names(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.parameter_names());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<std::string> >(S));
}

// [[Rcpp::export]]
SEXP Model__theta_parameter_names(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.parameter_names());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<std::string> >(S));
}

// // [[Rcpp::export]]
// SEXP Model__hessian_correction(SEXP xp, int type = 0){
//   glmmrType model(xp,static_cast<Type>(type));
//   auto functor = overloaded {
//     [](int) {  return returnType(0);}, 
//     [](auto ptr){return returnType(ptr->matrix.hessian_nonlinear_correction());}
//   };
//   auto S = std::visit(functor,model.ptr);
//   return wrap(std::get<MatrixXd>(S));
// }

// [[Rcpp::export]]
SEXP Model__any_nonlinear(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.calc.any_nonlinear);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<bool>(S));
}

// [[Rcpp::export]]
SEXP Model__sandwich(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.sandwich_matrix());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__infomat_theta(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.template information_matrix_theta<glmmr::IM::EIM>());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__kenward_roger(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.template small_sample_correction<glmmr::SE::KR, glmmr::IM::EIM>());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<CorrectionData<glmmr::SE::KR> >(S));
}

// [[Rcpp::export]]
SEXP Model__small_sample_correction(SEXP xp, int ss_type = 0, bool oim = false, int type = 0){
  using namespace glmmr;
  glmmrType model(xp,static_cast<Type>(type));
  SE corr = static_cast<SE>(ss_type);
  switch(corr){
  case SE::KR:
  {
    auto functor = overloaded {
      [](int) {  return returnType(0);}, 
      [&](auto ptr){
        if(oim){
          return returnType(ptr->matrix.template small_sample_correction<SE::KR,IM::OIM>());
        } else {
          return returnType(ptr->matrix.template small_sample_correction<SE::KR,IM::EIM>());
        }}
    };
    auto S = std::visit(functor,model.ptr);
    return wrap(std::get<CorrectionData<SE::KR> >(S));
    break;
  }
  case SE::KR2:
  {
    auto functor = overloaded {
      [](int) {  return returnType(0);}, 
      [&](auto ptr){
        if(oim){
          return returnType(ptr->matrix.template small_sample_correction<SE::KR2,IM::OIM>());
        } else {
          return returnType(ptr->matrix.template small_sample_correction<SE::KR2,IM::EIM>());
        }}
    };
    auto S = std::visit(functor,model.ptr);
    return wrap(std::get<CorrectionData<SE::KR2> >(S));
    break;
  }
  case SE::KRBoth:
  {
    auto functor = overloaded {
      [](int) {  return returnType(0);}, 
      [&](auto ptr){
        if(oim){
          return returnType(ptr->matrix.template small_sample_correction<SE::KRBoth,IM::OIM>());
        } else {
          return returnType(ptr->matrix.template small_sample_correction<SE::KRBoth,IM::EIM>());
        }}
    };
    auto S = std::visit(functor,model.ptr);
    return wrap(std::get<CorrectionData<SE::KRBoth> >(S));
    break;
  }
  case SE::Sat:
  {
    auto functor = overloaded {
      [](int) {  return returnType(0);}, 
      [&](auto ptr){
        if(oim){
          return returnType(ptr->matrix.template small_sample_correction<SE::Sat,IM::OIM>());
        } else {
          return returnType(ptr->matrix.template small_sample_correction<SE::Sat,IM::EIM>());
        }}
    };
    auto S = std::visit(functor,model.ptr);
    return wrap(std::get<CorrectionData<SE::Sat> >(S));
    break;
  }
  default:
  {
    Rcpp::stop("Not a valid small sample correction type");
  }
  }
  
}

// [[Rcpp::export]]
SEXP Model__box(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.box());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<BoxResults>(S));
}

// [[Rcpp::export]]
SEXP Model__cov_deriv(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.sigma_derivatives());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<Eigen::MatrixXd> >(S));
}

// [[Rcpp::export]]
SEXP Model__predict(SEXP xp, SEXP newdata_,
                    SEXP newoffset_,
                    int m, int type = 0){
  Eigen::ArrayXXd newdata = Rcpp::as<Eigen::ArrayXXd>(newdata_);
  Eigen::ArrayXd newoffset = Rcpp::as<Eigen::ArrayXd>(newoffset_);
  
  glmmrType model(xp,static_cast<Type>(type));
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
  VectorMatrix res = std::get<VectorMatrix>(S_re);
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

// [[Rcpp::export]]
SEXP Model__predict_re(SEXP xp, SEXP newdata_,
                       SEXP newoffset_,
                       int m, int type = 0){
  Eigen::ArrayXXd newdata = Rcpp::as<Eigen::ArrayXXd>(newdata_);
  Eigen::ArrayXd newoffset = Rcpp::as<Eigen::ArrayXd>(newoffset_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor_re = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->re.predict_re(newdata,newoffset));}
  };
  auto S_re = std::visit(functor_re,model.ptr);
  VectorMatrix res = std::get<VectorMatrix>(S_re);
  
  return Rcpp::List::create(
    Rcpp::Named("re_parameters") = wrap(res)
  );
}

//' Automatic differentiation of formulae
 //' 
 //' Exposes the automatic differentiator. Allows for calculation of Jacobian and Hessian matrices 
 //' of formulae in terms of specified parameters. Formula specification is as a string. Data items are automatically 
 //' multiplied by a parameter unless enclosed in parentheses.
 //' @param form_ String. Formula to differentiate specified in terms of data items and parameters. Any string not identifying 
 //' a function or a data item names in `colnames` is assumed to be a parameter.
 //' @param data_ Matrix. A matrix including the data. Rows represent observations. The number of columns should match the number 
 //' of items in `colnames_`
 //' @param colnames_ Vector of strings. The names of the columns of `data_`, used to match data named in the formula.
 //' @param parameters_ Vector of doubles. The values of the parameters at which to calculate the derivatives. The parameters should be in the 
 //' same order they appear in the formula.
 //' @return A list including the jacobian and hessian matrices.
 //' @examples
 //' # obtain the Jacobian and Hessian of the log-binomial model log-likelihood. 
 //' # The model is of data from an intervention and control group
 //' # with n1 and n0 participants, respectively, with y1 and y0 the number of events in each group. 
 //' # The mean is exp(alpha) in the control 
 //' # group and exp(alpha + beta) in the intervention group, so that beta is the log relative risk.
 //' hessian_from_formula(
 //'   form_ = "(y1)*(a+b)+((n1)-(y1))*log((1-exp(a+b)))+(y0)*a+((n0)-(y0))*log((1-exp(a)))",
 //'   data_ = matrix(c(10,100,20,100), nrow = 1),
 //'   colnames_ = c("y1","n1","y0","n0"),
 //'   parameters_ = c(log(0.1),log(0.5)))
 //' @export
 // [[Rcpp::export]]
 SEXP hessian_from_formula(SEXP form_, 
                           SEXP data_,
                           SEXP colnames_,
                           SEXP parameters_){
   std::string form = Rcpp::as<std::string>(form_);
   Eigen::ArrayXXd data = Rcpp::as<Eigen::ArrayXXd>(data_);
   std::vector<std::string> colnames = Rcpp::as<std::vector<std::string> >(colnames_);
   std::vector<double> parameters = Rcpp::as<std::vector<double> >(parameters_);
   glmmr::calculator calc;
   calc.data.conservativeResize(data.rows(),NoChange);
   std::vector<char> formula_as_chars(form.begin(),form.end());
   bool outparse = glmmr::parse_formula(formula_as_chars,
                                        calc,
                                        data,
                                        colnames,
                                        calc.data);
   (void)outparse;
   std::reverse(calc.instructions.begin(),calc.instructions.end());
   std::reverse(calc.indexes.begin(),calc.indexes.end());
   if(calc.parameter_names.size() != parameters.size())throw std::runtime_error("Wrong number of parameters");
   calc.parameters = parameters;
   VectorMatrix result = calc.jacobian_and_hessian();
   return Rcpp::wrap(result);
 }

//' Disable or enable parallelised computing
 //' 
 //' By default, the package will use multithreading for many calculations if OpenMP is 
 //' available on the system. For multi-user systems this may not be desired, so parallel
 //' execution can be disabled with this function.
 //' 
 //' @param parallel_ Logical indicating whether to use parallel computation (TRUE) or disable it (FALSE)
 //' @param cores_ Number of cores for parallel execution
 //' @return None, called for effects
 // [[Rcpp::export]]
 void setParallel(SEXP parallel_, int cores_ = 2){
   bool parallel = as<bool>(parallel_);
   if(OMP_IS_USED){
     int a, b; // needed for defines on machines without openmp
     if(!parallel){
       a = 0;
       b = 1;
       omp_set_dynamic(a); 
       omp_set_num_threads(b);
       Eigen::setNbThreads(b);
     } else {
       a = 1;
       b = cores_;
       omp_set_dynamic(a); 
       omp_set_num_threads(b);
       Eigen::setNbThreads(b);
     }
   } 
 }


// [[Rcpp::export]]
std::vector<std::string> re_names(const std::string& formula,
                                  bool as_formula = true){
  glmmr::Formula form(formula);
  std::vector<std::string> re;
  if(as_formula){
    re.resize(form.re_.size());
    for(int i = 0; i < form.re_.size(); i++){
      re[i] = "("+form.z_[i]+"|"+form.re_[i]+")";
    }
  } else {
    for(int i = 0; i < form.re_.size(); i++){
      re.push_back(form.re_[i]);
      re.push_back(form.z_[i]);
    }
  }
  
  return re;
}

// [[Rcpp::export]]
Eigen::VectorXd attenuate_xb(const Eigen::VectorXd& xb,
                             const Eigen::MatrixXd& Z,
                             const Eigen::MatrixXd& D,
                             const std::string& link){
  Eigen::VectorXd linpred = glmmr::maths::attenuted_xb(xb,Z,D,glmmr::str_to_link.at(link));
  return linpred;
}

// [[Rcpp::export]]
Eigen::VectorXd dlinkdeta(const Eigen::VectorXd& xb,
                          const std::string& link){
  Eigen::VectorXd deta = glmmr::maths::detadmu(xb,glmmr::str_to_link.at(link));
  return deta;
}

// Access to this function is provided to the user in the 
// glmmrOptim package
// [[Rcpp::export]]
SEXP girling_algorithm(SEXP xp, 
                       SEXP N_,
                       SEXP C_,
                       SEXP tol_){
  double N = as<double>(N_);
  double tol = as<double>(tol_);
  Eigen::VectorXd C = as<Eigen::VectorXd>(C_);
  XPtr<glmm> ptr(xp);
  Eigen::ArrayXd w = ptr->optim.optimum_weights(N,C,tol);
  return wrap(w);
}

// [[Rcpp::export]]
SEXP get_variable_names(SEXP formula_, 
                        SEXP colnames_){
  std::string formula = as<std::string>(formula_);
  Eigen::ArrayXXd data(1,1);
  Eigen::MatrixXd Xdata(1,1);
  data.setZero();
  Xdata.setZero();
  std::vector<std::string> colnames = as<std::vector<std::string> >(colnames_);
  glmmr::Formula form(formula);
  glmmr::calculator calc;
  bool out = glmmr::parse_formula(form.linear_predictor_,calc,data,colnames,Xdata,false,false);
  (void)out;
  return wrap(calc.data_names);
}
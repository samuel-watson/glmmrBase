// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// Covariance__new
SEXP Covariance__new(SEXP form_, SEXP data_, SEXP colnames_);
RcppExport SEXP _glmmrBase_Covariance__new(SEXP form_SEXP, SEXP data_SEXP, SEXP colnames_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type form_(form_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type data_(data_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type colnames_(colnames_SEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__new(form_, data_, colnames_));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__Z
SEXP Covariance__Z(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__Z(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__Z(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__ZL
SEXP Covariance__ZL(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__ZL(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__ZL(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__LZWZL
SEXP Covariance__LZWZL(SEXP xp, SEXP w_);
RcppExport SEXP _glmmrBase_Covariance__LZWZL(SEXP xpSEXP, SEXP w_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type w_(w_SEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__LZWZL(xp, w_));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__Update_parameters
void Covariance__Update_parameters(SEXP xp, SEXP parameters_);
RcppExport SEXP _glmmrBase_Covariance__Update_parameters(SEXP xpSEXP, SEXP parameters_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type parameters_(parameters_SEXP);
    Covariance__Update_parameters(xp, parameters_);
    return R_NilValue;
END_RCPP
}
// Covariance__D
SEXP Covariance__D(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__D(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__D(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__D_chol
SEXP Covariance__D_chol(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__D_chol(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__D_chol(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__B
SEXP Covariance__B(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__B(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__B(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__Q
SEXP Covariance__Q(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__Q(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__Q(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__log_likelihood
SEXP Covariance__log_likelihood(SEXP xp, SEXP u_);
RcppExport SEXP _glmmrBase_Covariance__log_likelihood(SEXP xpSEXP, SEXP u_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type u_(u_SEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__log_likelihood(xp, u_));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__log_determinant
SEXP Covariance__log_determinant(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__log_determinant(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__log_determinant(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__n_cov_pars
SEXP Covariance__n_cov_pars(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__n_cov_pars(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__n_cov_pars(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__simulate_re
SEXP Covariance__simulate_re(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__simulate_re(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__simulate_re(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__make_sparse
void Covariance__make_sparse(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__make_sparse(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Covariance__make_sparse(xp);
    return R_NilValue;
END_RCPP
}
// Covariance__make_dense
void Covariance__make_dense(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__make_dense(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Covariance__make_dense(xp);
    return R_NilValue;
END_RCPP
}
// Covariance__any_gr
SEXP Covariance__any_gr(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__any_gr(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__any_gr(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__parameter_fn_index
SEXP Covariance__parameter_fn_index(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__parameter_fn_index(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__parameter_fn_index(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__re_terms
SEXP Covariance__re_terms(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__re_terms(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__re_terms(xp));
    return rcpp_result_gen;
END_RCPP
}
// Covariance__re_count
SEXP Covariance__re_count(SEXP xp);
RcppExport SEXP _glmmrBase_Covariance__re_count(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Covariance__re_count(xp));
    return rcpp_result_gen;
END_RCPP
}
// Linpred__new
SEXP Linpred__new(SEXP formula_, SEXP data_, SEXP colnames_);
RcppExport SEXP _glmmrBase_Linpred__new(SEXP formula_SEXP, SEXP data_SEXP, SEXP colnames_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type formula_(formula_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type data_(data_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type colnames_(colnames_SEXP);
    rcpp_result_gen = Rcpp::wrap(Linpred__new(formula_, data_, colnames_));
    return rcpp_result_gen;
END_RCPP
}
// Linpred__update_pars
void Linpred__update_pars(SEXP xp, SEXP parameters_);
RcppExport SEXP _glmmrBase_Linpred__update_pars(SEXP xpSEXP, SEXP parameters_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type parameters_(parameters_SEXP);
    Linpred__update_pars(xp, parameters_);
    return R_NilValue;
END_RCPP
}
// Linpred__xb
SEXP Linpred__xb(SEXP xp);
RcppExport SEXP _glmmrBase_Linpred__xb(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Linpred__xb(xp));
    return rcpp_result_gen;
END_RCPP
}
// Linpred__x
SEXP Linpred__x(SEXP xp);
RcppExport SEXP _glmmrBase_Linpred__x(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Linpred__x(xp));
    return rcpp_result_gen;
END_RCPP
}
// Linpred__beta_names
SEXP Linpred__beta_names(SEXP xp);
RcppExport SEXP _glmmrBase_Linpred__beta_names(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Linpred__beta_names(xp));
    return rcpp_result_gen;
END_RCPP
}
// Linpred__any_nonlinear
SEXP Linpred__any_nonlinear(SEXP xp);
RcppExport SEXP _glmmrBase_Linpred__any_nonlinear(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Linpred__any_nonlinear(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__new
SEXP Model__new(SEXP y_, SEXP formula_, SEXP data_, SEXP colnames_, SEXP family_, SEXP link_);
RcppExport SEXP _glmmrBase_Model__new(SEXP y_SEXP, SEXP formula_SEXP, SEXP data_SEXP, SEXP colnames_SEXP, SEXP family_SEXP, SEXP link_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type y_(y_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type formula_(formula_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type data_(data_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type colnames_(colnames_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type family_(family_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type link_(link_SEXP);
    rcpp_result_gen = Rcpp::wrap(Model__new(y_, formula_, data_, colnames_, family_, link_));
    return rcpp_result_gen;
END_RCPP
}
// Model__set_offset
void Model__set_offset(SEXP xp, SEXP offset_);
RcppExport SEXP _glmmrBase_Model__set_offset(SEXP xpSEXP, SEXP offset_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type offset_(offset_SEXP);
    Model__set_offset(xp, offset_);
    return R_NilValue;
END_RCPP
}
// Model__update_beta
void Model__update_beta(SEXP xp, SEXP beta_);
RcppExport SEXP _glmmrBase_Model__update_beta(SEXP xpSEXP, SEXP beta_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type beta_(beta_SEXP);
    Model__update_beta(xp, beta_);
    return R_NilValue;
END_RCPP
}
// Model__update_theta
void Model__update_theta(SEXP xp, SEXP theta_);
RcppExport SEXP _glmmrBase_Model__update_theta(SEXP xpSEXP, SEXP theta_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type theta_(theta_SEXP);
    Model__update_theta(xp, theta_);
    return R_NilValue;
END_RCPP
}
// Model__update_u
void Model__update_u(SEXP xp, SEXP u_);
RcppExport SEXP _glmmrBase_Model__update_u(SEXP xpSEXP, SEXP u_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type u_(u_SEXP);
    Model__update_u(xp, u_);
    return R_NilValue;
END_RCPP
}
// Model__predict
SEXP Model__predict(SEXP xp, SEXP newdata_, SEXP newoffset_, int m);
RcppExport SEXP _glmmrBase_Model__predict(SEXP xpSEXP, SEXP newdata_SEXP, SEXP newoffset_SEXP, SEXP mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type newdata_(newdata_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type newoffset_(newoffset_SEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__predict(xp, newdata_, newoffset_, m));
    return rcpp_result_gen;
END_RCPP
}
// Model__use_attenuation
void Model__use_attenuation(SEXP xp, SEXP use_);
RcppExport SEXP _glmmrBase_Model__use_attenuation(SEXP xpSEXP, SEXP use_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type use_(use_SEXP);
    Model__use_attenuation(xp, use_);
    return R_NilValue;
END_RCPP
}
// Model__update_W
void Model__update_W(SEXP xp);
RcppExport SEXP _glmmrBase_Model__update_W(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__update_W(xp);
    return R_NilValue;
END_RCPP
}
// Model__log_prob
SEXP Model__log_prob(SEXP xp, SEXP v_);
RcppExport SEXP _glmmrBase_Model__log_prob(SEXP xpSEXP, SEXP v_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type v_(v_SEXP);
    rcpp_result_gen = Rcpp::wrap(Model__log_prob(xp, v_));
    return rcpp_result_gen;
END_RCPP
}
// Model__log_gradient
SEXP Model__log_gradient(SEXP xp, SEXP v_, SEXP beta_);
RcppExport SEXP _glmmrBase_Model__log_gradient(SEXP xpSEXP, SEXP v_SEXP, SEXP beta_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type v_(v_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type beta_(beta_SEXP);
    rcpp_result_gen = Rcpp::wrap(Model__log_gradient(xp, v_, beta_));
    return rcpp_result_gen;
END_RCPP
}
// Model__linear_predictor
SEXP Model__linear_predictor(SEXP xp);
RcppExport SEXP _glmmrBase_Model__linear_predictor(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__linear_predictor(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__log_likelihood
SEXP Model__log_likelihood(SEXP xp);
RcppExport SEXP _glmmrBase_Model__log_likelihood(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__log_likelihood(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__ml_theta
void Model__ml_theta(SEXP xp);
RcppExport SEXP _glmmrBase_Model__ml_theta(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__ml_theta(xp);
    return R_NilValue;
END_RCPP
}
// Model__ml_beta
void Model__ml_beta(SEXP xp);
RcppExport SEXP _glmmrBase_Model__ml_beta(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__ml_beta(xp);
    return R_NilValue;
END_RCPP
}
// Model__ml_all
void Model__ml_all(SEXP xp);
RcppExport SEXP _glmmrBase_Model__ml_all(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__ml_all(xp);
    return R_NilValue;
END_RCPP
}
// Model__laplace_ml_beta_u
void Model__laplace_ml_beta_u(SEXP xp);
RcppExport SEXP _glmmrBase_Model__laplace_ml_beta_u(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__laplace_ml_beta_u(xp);
    return R_NilValue;
END_RCPP
}
// Model__laplace_ml_theta
void Model__laplace_ml_theta(SEXP xp);
RcppExport SEXP _glmmrBase_Model__laplace_ml_theta(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__laplace_ml_theta(xp);
    return R_NilValue;
END_RCPP
}
// Model__laplace_ml_beta_theta
void Model__laplace_ml_beta_theta(SEXP xp);
RcppExport SEXP _glmmrBase_Model__laplace_ml_beta_theta(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__laplace_ml_beta_theta(xp);
    return R_NilValue;
END_RCPP
}
// Model__nr_beta
void Model__nr_beta(SEXP xp);
RcppExport SEXP _glmmrBase_Model__nr_beta(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__nr_beta(xp);
    return R_NilValue;
END_RCPP
}
// Model__laplace_nr_beta_u
void Model__laplace_nr_beta_u(SEXP xp);
RcppExport SEXP _glmmrBase_Model__laplace_nr_beta_u(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__laplace_nr_beta_u(xp);
    return R_NilValue;
END_RCPP
}
// Model__Sigma
SEXP Model__Sigma(SEXP xp, bool inverse);
RcppExport SEXP _glmmrBase_Model__Sigma(SEXP xpSEXP, SEXP inverseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< bool >::type inverse(inverseSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__Sigma(xp, inverse));
    return rcpp_result_gen;
END_RCPP
}
// Model__information_matrix
SEXP Model__information_matrix(SEXP xp);
RcppExport SEXP _glmmrBase_Model__information_matrix(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__information_matrix(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__hessian
SEXP Model__hessian(SEXP xp);
RcppExport SEXP _glmmrBase_Model__hessian(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__hessian(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__obs_information_matrix
SEXP Model__obs_information_matrix(SEXP xp);
RcppExport SEXP _glmmrBase_Model__obs_information_matrix(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__obs_information_matrix(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__u
SEXP Model__u(SEXP xp, bool scaled_);
RcppExport SEXP _glmmrBase_Model__u(SEXP xpSEXP, SEXP scaled_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< bool >::type scaled_(scaled_SEXP);
    rcpp_result_gen = Rcpp::wrap(Model__u(xp, scaled_));
    return rcpp_result_gen;
END_RCPP
}
// Model__Zu
SEXP Model__Zu(SEXP xp);
RcppExport SEXP _glmmrBase_Model__Zu(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__Zu(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__P
SEXP Model__P(SEXP xp);
RcppExport SEXP _glmmrBase_Model__P(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__P(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__Q
SEXP Model__Q(SEXP xp);
RcppExport SEXP _glmmrBase_Model__Q(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__Q(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__X
SEXP Model__X(SEXP xp);
RcppExport SEXP _glmmrBase_Model__X(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__X(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__mcmc_sample
void Model__mcmc_sample(SEXP xp, SEXP warmup_, SEXP samples_, SEXP adapt_);
RcppExport SEXP _glmmrBase_Model__mcmc_sample(SEXP xpSEXP, SEXP warmup_SEXP, SEXP samples_SEXP, SEXP adapt_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type warmup_(warmup_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type samples_(samples_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type adapt_(adapt_SEXP);
    Model__mcmc_sample(xp, warmup_, samples_, adapt_);
    return R_NilValue;
END_RCPP
}
// Model__set_trace
void Model__set_trace(SEXP xp, SEXP trace_);
RcppExport SEXP _glmmrBase_Model__set_trace(SEXP xpSEXP, SEXP trace_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type trace_(trace_SEXP);
    Model__set_trace(xp, trace_);
    return R_NilValue;
END_RCPP
}
// Model__get_beta
SEXP Model__get_beta(SEXP xp);
RcppExport SEXP _glmmrBase_Model__get_beta(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__get_beta(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__y
SEXP Model__y(SEXP xp);
RcppExport SEXP _glmmrBase_Model__y(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__y(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__get_theta
SEXP Model__get_theta(SEXP xp);
RcppExport SEXP _glmmrBase_Model__get_theta(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__get_theta(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__get_var_par
SEXP Model__get_var_par(SEXP xp);
RcppExport SEXP _glmmrBase_Model__get_var_par(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__get_var_par(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__set_var_par
void Model__set_var_par(SEXP xp, SEXP var_par_);
RcppExport SEXP _glmmrBase_Model__set_var_par(SEXP xpSEXP, SEXP var_par_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type var_par_(var_par_SEXP);
    Model__set_var_par(xp, var_par_);
    return R_NilValue;
END_RCPP
}
// Model__L
SEXP Model__L(SEXP xp);
RcppExport SEXP _glmmrBase_Model__L(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__L(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__ZL
SEXP Model__ZL(SEXP xp);
RcppExport SEXP _glmmrBase_Model__ZL(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__ZL(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__xb
SEXP Model__xb(SEXP xp);
RcppExport SEXP _glmmrBase_Model__xb(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__xb(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__aic
SEXP Model__aic(SEXP xp);
RcppExport SEXP _glmmrBase_Model__aic(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__aic(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__mcmc_set_lambda
void Model__mcmc_set_lambda(SEXP xp, SEXP lambda_);
RcppExport SEXP _glmmrBase_Model__mcmc_set_lambda(SEXP xpSEXP, SEXP lambda_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type lambda_(lambda_SEXP);
    Model__mcmc_set_lambda(xp, lambda_);
    return R_NilValue;
END_RCPP
}
// Model__mcmc_set_max_steps
void Model__mcmc_set_max_steps(SEXP xp, SEXP max_steps_);
RcppExport SEXP _glmmrBase_Model__mcmc_set_max_steps(SEXP xpSEXP, SEXP max_steps_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type max_steps_(max_steps_SEXP);
    Model__mcmc_set_max_steps(xp, max_steps_);
    return R_NilValue;
END_RCPP
}
// Model__mcmc_set_refresh
void Model__mcmc_set_refresh(SEXP xp, SEXP refresh_);
RcppExport SEXP _glmmrBase_Model__mcmc_set_refresh(SEXP xpSEXP, SEXP refresh_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type refresh_(refresh_SEXP);
    Model__mcmc_set_refresh(xp, refresh_);
    return R_NilValue;
END_RCPP
}
// Model__mcmc_set_target_accept
void Model__mcmc_set_target_accept(SEXP xp, SEXP target_);
RcppExport SEXP _glmmrBase_Model__mcmc_set_target_accept(SEXP xpSEXP, SEXP target_SEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type target_(target_SEXP);
    Model__mcmc_set_target_accept(xp, target_);
    return R_NilValue;
END_RCPP
}
// Model__make_sparse
void Model__make_sparse(SEXP xp);
RcppExport SEXP _glmmrBase_Model__make_sparse(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__make_sparse(xp);
    return R_NilValue;
END_RCPP
}
// Model__make_dense
void Model__make_dense(SEXP xp);
RcppExport SEXP _glmmrBase_Model__make_dense(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Model__make_dense(xp);
    return R_NilValue;
END_RCPP
}
// Model__beta_parameter_names
SEXP Model__beta_parameter_names(SEXP xp);
RcppExport SEXP _glmmrBase_Model__beta_parameter_names(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__beta_parameter_names(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__theta_parameter_names
SEXP Model__theta_parameter_names(SEXP xp);
RcppExport SEXP _glmmrBase_Model__theta_parameter_names(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__theta_parameter_names(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__hess_and_grad
SEXP Model__hess_and_grad(SEXP xp);
RcppExport SEXP _glmmrBase_Model__hess_and_grad(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__hess_and_grad(xp));
    return rcpp_result_gen;
END_RCPP
}
// Model__sandwich
SEXP Model__sandwich(SEXP xp);
RcppExport SEXP _glmmrBase_Model__sandwich(SEXP xpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    rcpp_result_gen = Rcpp::wrap(Model__sandwich(xp));
    return rcpp_result_gen;
END_RCPP
}
// re_names
std::vector<std::string> re_names(const std::string& formula);
RcppExport SEXP _glmmrBase_re_names(SEXP formulaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type formula(formulaSEXP);
    rcpp_result_gen = Rcpp::wrap(re_names(formula));
    return rcpp_result_gen;
END_RCPP
}
// gen_dhdmu
Eigen::VectorXd gen_dhdmu(const Eigen::VectorXd& xb, std::string family, std::string link);
RcppExport SEXP _glmmrBase_gen_dhdmu(SEXP xbSEXP, SEXP familySEXP, SEXP linkSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type xb(xbSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    rcpp_result_gen = Rcpp::wrap(gen_dhdmu(xb, family, link));
    return rcpp_result_gen;
END_RCPP
}
// gen_sigma_approx
Eigen::MatrixXd gen_sigma_approx(const Eigen::VectorXd& xb, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& D, std::string family, std::string link, double var_par, bool attenuate);
RcppExport SEXP _glmmrBase_gen_sigma_approx(SEXP xbSEXP, SEXP ZSEXP, SEXP DSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP var_parSEXP, SEXP attenuateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type xb(xbSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type D(DSEXP);
    Rcpp::traits::input_parameter< std::string >::type family(familySEXP);
    Rcpp::traits::input_parameter< std::string >::type link(linkSEXP);
    Rcpp::traits::input_parameter< double >::type var_par(var_parSEXP);
    Rcpp::traits::input_parameter< bool >::type attenuate(attenuateSEXP);
    rcpp_result_gen = Rcpp::wrap(gen_sigma_approx(xb, Z, D, family, link, var_par, attenuate));
    return rcpp_result_gen;
END_RCPP
}
// attenuate_xb
Eigen::VectorXd attenuate_xb(const Eigen::VectorXd& xb, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& D, const std::string& link);
RcppExport SEXP _glmmrBase_attenuate_xb(SEXP xbSEXP, SEXP ZSEXP, SEXP DSEXP, SEXP linkSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type xb(xbSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type D(DSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type link(linkSEXP);
    rcpp_result_gen = Rcpp::wrap(attenuate_xb(xb, Z, D, link));
    return rcpp_result_gen;
END_RCPP
}
// dlinkdeta
Eigen::VectorXd dlinkdeta(const Eigen::VectorXd& xb, const std::string& link);
RcppExport SEXP _glmmrBase_dlinkdeta(SEXP xbSEXP, SEXP linkSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type xb(xbSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type link(linkSEXP);
    rcpp_result_gen = Rcpp::wrap(dlinkdeta(xb, link));
    return rcpp_result_gen;
END_RCPP
}
// girling_algorithm
SEXP girling_algorithm(SEXP xp, SEXP N_, SEXP sigma_sq_, SEXP C_, SEXP tol_);
RcppExport SEXP _glmmrBase_girling_algorithm(SEXP xpSEXP, SEXP N_SEXP, SEXP sigma_sq_SEXP, SEXP C_SEXP, SEXP tol_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type xp(xpSEXP);
    Rcpp::traits::input_parameter< SEXP >::type N_(N_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type sigma_sq_(sigma_sq_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type C_(C_SEXP);
    Rcpp::traits::input_parameter< SEXP >::type tol_(tol_SEXP);
    rcpp_result_gen = Rcpp::wrap(girling_algorithm(xp, N_, sigma_sq_, C_, tol_));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_glmmrBase_Covariance__new", (DL_FUNC) &_glmmrBase_Covariance__new, 3},
    {"_glmmrBase_Covariance__Z", (DL_FUNC) &_glmmrBase_Covariance__Z, 1},
    {"_glmmrBase_Covariance__ZL", (DL_FUNC) &_glmmrBase_Covariance__ZL, 1},
    {"_glmmrBase_Covariance__LZWZL", (DL_FUNC) &_glmmrBase_Covariance__LZWZL, 2},
    {"_glmmrBase_Covariance__Update_parameters", (DL_FUNC) &_glmmrBase_Covariance__Update_parameters, 2},
    {"_glmmrBase_Covariance__D", (DL_FUNC) &_glmmrBase_Covariance__D, 1},
    {"_glmmrBase_Covariance__D_chol", (DL_FUNC) &_glmmrBase_Covariance__D_chol, 1},
    {"_glmmrBase_Covariance__B", (DL_FUNC) &_glmmrBase_Covariance__B, 1},
    {"_glmmrBase_Covariance__Q", (DL_FUNC) &_glmmrBase_Covariance__Q, 1},
    {"_glmmrBase_Covariance__log_likelihood", (DL_FUNC) &_glmmrBase_Covariance__log_likelihood, 2},
    {"_glmmrBase_Covariance__log_determinant", (DL_FUNC) &_glmmrBase_Covariance__log_determinant, 1},
    {"_glmmrBase_Covariance__n_cov_pars", (DL_FUNC) &_glmmrBase_Covariance__n_cov_pars, 1},
    {"_glmmrBase_Covariance__simulate_re", (DL_FUNC) &_glmmrBase_Covariance__simulate_re, 1},
    {"_glmmrBase_Covariance__make_sparse", (DL_FUNC) &_glmmrBase_Covariance__make_sparse, 1},
    {"_glmmrBase_Covariance__make_dense", (DL_FUNC) &_glmmrBase_Covariance__make_dense, 1},
    {"_glmmrBase_Covariance__any_gr", (DL_FUNC) &_glmmrBase_Covariance__any_gr, 1},
    {"_glmmrBase_Covariance__parameter_fn_index", (DL_FUNC) &_glmmrBase_Covariance__parameter_fn_index, 1},
    {"_glmmrBase_Covariance__re_terms", (DL_FUNC) &_glmmrBase_Covariance__re_terms, 1},
    {"_glmmrBase_Covariance__re_count", (DL_FUNC) &_glmmrBase_Covariance__re_count, 1},
    {"_glmmrBase_Linpred__new", (DL_FUNC) &_glmmrBase_Linpred__new, 3},
    {"_glmmrBase_Linpred__update_pars", (DL_FUNC) &_glmmrBase_Linpred__update_pars, 2},
    {"_glmmrBase_Linpred__xb", (DL_FUNC) &_glmmrBase_Linpred__xb, 1},
    {"_glmmrBase_Linpred__x", (DL_FUNC) &_glmmrBase_Linpred__x, 1},
    {"_glmmrBase_Linpred__beta_names", (DL_FUNC) &_glmmrBase_Linpred__beta_names, 1},
    {"_glmmrBase_Linpred__any_nonlinear", (DL_FUNC) &_glmmrBase_Linpred__any_nonlinear, 1},
    {"_glmmrBase_Model__new", (DL_FUNC) &_glmmrBase_Model__new, 6},
    {"_glmmrBase_Model__set_offset", (DL_FUNC) &_glmmrBase_Model__set_offset, 2},
    {"_glmmrBase_Model__update_beta", (DL_FUNC) &_glmmrBase_Model__update_beta, 2},
    {"_glmmrBase_Model__update_theta", (DL_FUNC) &_glmmrBase_Model__update_theta, 2},
    {"_glmmrBase_Model__update_u", (DL_FUNC) &_glmmrBase_Model__update_u, 2},
    {"_glmmrBase_Model__predict", (DL_FUNC) &_glmmrBase_Model__predict, 4},
    {"_glmmrBase_Model__use_attenuation", (DL_FUNC) &_glmmrBase_Model__use_attenuation, 2},
    {"_glmmrBase_Model__update_W", (DL_FUNC) &_glmmrBase_Model__update_W, 1},
    {"_glmmrBase_Model__log_prob", (DL_FUNC) &_glmmrBase_Model__log_prob, 2},
    {"_glmmrBase_Model__log_gradient", (DL_FUNC) &_glmmrBase_Model__log_gradient, 3},
    {"_glmmrBase_Model__linear_predictor", (DL_FUNC) &_glmmrBase_Model__linear_predictor, 1},
    {"_glmmrBase_Model__log_likelihood", (DL_FUNC) &_glmmrBase_Model__log_likelihood, 1},
    {"_glmmrBase_Model__ml_theta", (DL_FUNC) &_glmmrBase_Model__ml_theta, 1},
    {"_glmmrBase_Model__ml_beta", (DL_FUNC) &_glmmrBase_Model__ml_beta, 1},
    {"_glmmrBase_Model__ml_all", (DL_FUNC) &_glmmrBase_Model__ml_all, 1},
    {"_glmmrBase_Model__laplace_ml_beta_u", (DL_FUNC) &_glmmrBase_Model__laplace_ml_beta_u, 1},
    {"_glmmrBase_Model__laplace_ml_theta", (DL_FUNC) &_glmmrBase_Model__laplace_ml_theta, 1},
    {"_glmmrBase_Model__laplace_ml_beta_theta", (DL_FUNC) &_glmmrBase_Model__laplace_ml_beta_theta, 1},
    {"_glmmrBase_Model__nr_beta", (DL_FUNC) &_glmmrBase_Model__nr_beta, 1},
    {"_glmmrBase_Model__laplace_nr_beta_u", (DL_FUNC) &_glmmrBase_Model__laplace_nr_beta_u, 1},
    {"_glmmrBase_Model__Sigma", (DL_FUNC) &_glmmrBase_Model__Sigma, 2},
    {"_glmmrBase_Model__information_matrix", (DL_FUNC) &_glmmrBase_Model__information_matrix, 1},
    {"_glmmrBase_Model__hessian", (DL_FUNC) &_glmmrBase_Model__hessian, 1},
    {"_glmmrBase_Model__obs_information_matrix", (DL_FUNC) &_glmmrBase_Model__obs_information_matrix, 1},
    {"_glmmrBase_Model__u", (DL_FUNC) &_glmmrBase_Model__u, 2},
    {"_glmmrBase_Model__Zu", (DL_FUNC) &_glmmrBase_Model__Zu, 1},
    {"_glmmrBase_Model__P", (DL_FUNC) &_glmmrBase_Model__P, 1},
    {"_glmmrBase_Model__Q", (DL_FUNC) &_glmmrBase_Model__Q, 1},
    {"_glmmrBase_Model__X", (DL_FUNC) &_glmmrBase_Model__X, 1},
    {"_glmmrBase_Model__mcmc_sample", (DL_FUNC) &_glmmrBase_Model__mcmc_sample, 4},
    {"_glmmrBase_Model__set_trace", (DL_FUNC) &_glmmrBase_Model__set_trace, 2},
    {"_glmmrBase_Model__get_beta", (DL_FUNC) &_glmmrBase_Model__get_beta, 1},
    {"_glmmrBase_Model__y", (DL_FUNC) &_glmmrBase_Model__y, 1},
    {"_glmmrBase_Model__get_theta", (DL_FUNC) &_glmmrBase_Model__get_theta, 1},
    {"_glmmrBase_Model__get_var_par", (DL_FUNC) &_glmmrBase_Model__get_var_par, 1},
    {"_glmmrBase_Model__set_var_par", (DL_FUNC) &_glmmrBase_Model__set_var_par, 2},
    {"_glmmrBase_Model__L", (DL_FUNC) &_glmmrBase_Model__L, 1},
    {"_glmmrBase_Model__ZL", (DL_FUNC) &_glmmrBase_Model__ZL, 1},
    {"_glmmrBase_Model__xb", (DL_FUNC) &_glmmrBase_Model__xb, 1},
    {"_glmmrBase_Model__aic", (DL_FUNC) &_glmmrBase_Model__aic, 1},
    {"_glmmrBase_Model__mcmc_set_lambda", (DL_FUNC) &_glmmrBase_Model__mcmc_set_lambda, 2},
    {"_glmmrBase_Model__mcmc_set_max_steps", (DL_FUNC) &_glmmrBase_Model__mcmc_set_max_steps, 2},
    {"_glmmrBase_Model__mcmc_set_refresh", (DL_FUNC) &_glmmrBase_Model__mcmc_set_refresh, 2},
    {"_glmmrBase_Model__mcmc_set_target_accept", (DL_FUNC) &_glmmrBase_Model__mcmc_set_target_accept, 2},
    {"_glmmrBase_Model__make_sparse", (DL_FUNC) &_glmmrBase_Model__make_sparse, 1},
    {"_glmmrBase_Model__make_dense", (DL_FUNC) &_glmmrBase_Model__make_dense, 1},
    {"_glmmrBase_Model__beta_parameter_names", (DL_FUNC) &_glmmrBase_Model__beta_parameter_names, 1},
    {"_glmmrBase_Model__theta_parameter_names", (DL_FUNC) &_glmmrBase_Model__theta_parameter_names, 1},
    {"_glmmrBase_Model__hess_and_grad", (DL_FUNC) &_glmmrBase_Model__hess_and_grad, 1},
    {"_glmmrBase_Model__sandwich", (DL_FUNC) &_glmmrBase_Model__sandwich, 1},
    {"_glmmrBase_re_names", (DL_FUNC) &_glmmrBase_re_names, 1},
    {"_glmmrBase_gen_dhdmu", (DL_FUNC) &_glmmrBase_gen_dhdmu, 3},
    {"_glmmrBase_gen_sigma_approx", (DL_FUNC) &_glmmrBase_gen_sigma_approx, 7},
    {"_glmmrBase_attenuate_xb", (DL_FUNC) &_glmmrBase_attenuate_xb, 4},
    {"_glmmrBase_dlinkdeta", (DL_FUNC) &_glmmrBase_dlinkdeta, 2},
    {"_glmmrBase_girling_algorithm", (DL_FUNC) &_glmmrBase_girling_algorithm, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_glmmrBase(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

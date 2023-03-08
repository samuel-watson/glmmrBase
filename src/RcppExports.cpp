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
// genX
Eigen::MatrixXd genX(const std::string& formula, const Eigen::ArrayXXd& data, const std::vector<std::string>& colnames);
RcppExport SEXP _glmmrBase_genX(SEXP formulaSEXP, SEXP dataSEXP, SEXP colnamesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type formula(formulaSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const std::vector<std::string>& >::type colnames(colnamesSEXP);
    rcpp_result_gen = Rcpp::wrap(genX(formula, data, colnames));
    return rcpp_result_gen;
END_RCPP
}
// x_names
std::vector<std::string> x_names(const std::string& formula);
RcppExport SEXP _glmmrBase_x_names(SEXP formulaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type formula(formulaSEXP);
    rcpp_result_gen = Rcpp::wrap(x_names(formula));
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

static const R_CallMethodDef CallEntries[] = {
    {"_glmmrBase_Covariance__new", (DL_FUNC) &_glmmrBase_Covariance__new, 3},
    {"_glmmrBase_Covariance__Z", (DL_FUNC) &_glmmrBase_Covariance__Z, 1},
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
    {"_glmmrBase_genX", (DL_FUNC) &_glmmrBase_genX, 3},
    {"_glmmrBase_x_names", (DL_FUNC) &_glmmrBase_x_names, 1},
    {"_glmmrBase_re_names", (DL_FUNC) &_glmmrBase_re_names, 1},
    {"_glmmrBase_gen_dhdmu", (DL_FUNC) &_glmmrBase_gen_dhdmu, 3},
    {"_glmmrBase_gen_sigma_approx", (DL_FUNC) &_glmmrBase_gen_sigma_approx, 7},
    {"_glmmrBase_attenuate_xb", (DL_FUNC) &_glmmrBase_attenuate_xb, 4},
    {"_glmmrBase_dlinkdeta", (DL_FUNC) &_glmmrBase_dlinkdeta, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_glmmrBase(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

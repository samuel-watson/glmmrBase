// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// genD
Eigen::MatrixXd genD(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::VectorXd& gamma);
RcppExport SEXP _glmmrBase_genD(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type gamma(gammaSEXP);
    rcpp_result_gen = Rcpp::wrap(genD(cov, data, eff_range, gamma));
    return rcpp_result_gen;
END_RCPP
}
// genCholD
Eigen::MatrixXd genCholD(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::VectorXd& gamma);
RcppExport SEXP _glmmrBase_genCholD(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type gamma(gammaSEXP);
    rcpp_result_gen = Rcpp::wrap(genCholD(cov, data, eff_range, gamma));
    return rcpp_result_gen;
END_RCPP
}
// sample_re
Eigen::VectorXd sample_re(const Eigen::ArrayXXi& cov, const Eigen::ArrayXd& data, const Eigen::ArrayXd& eff_range, const Eigen::VectorXd& gamma);
RcppExport SEXP _glmmrBase_sample_re(SEXP covSEXP, SEXP dataSEXP, SEXP eff_rangeSEXP, SEXP gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::ArrayXXi& >::type cov(covSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const Eigen::ArrayXd& >::type eff_range(eff_rangeSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type gamma(gammaSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_re(cov, data, eff_range, gamma));
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
Eigen::MatrixXd gen_sigma_approx(const Eigen::VectorXd& xb, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& D, std::string family, std::string link, double var_par, bool attenuate, bool qlik);
RcppExport SEXP _glmmrBase_gen_sigma_approx(SEXP xbSEXP, SEXP ZSEXP, SEXP DSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP var_parSEXP, SEXP attenuateSEXP, SEXP qlikSEXP) {
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
    Rcpp::traits::input_parameter< bool >::type qlik(qlikSEXP);
    rcpp_result_gen = Rcpp::wrap(gen_sigma_approx(xb, Z, D, family, link, var_par, attenuate, qlik));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_glmmrBase_genD", (DL_FUNC) &_glmmrBase_genD, 4},
    {"_glmmrBase_genCholD", (DL_FUNC) &_glmmrBase_genCholD, 4},
    {"_glmmrBase_sample_re", (DL_FUNC) &_glmmrBase_sample_re, 4},
    {"_glmmrBase_gen_dhdmu", (DL_FUNC) &_glmmrBase_gen_dhdmu, 3},
    {"_glmmrBase_gen_sigma_approx", (DL_FUNC) &_glmmrBase_gen_sigma_approx, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_glmmrBase(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

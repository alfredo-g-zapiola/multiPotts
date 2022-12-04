// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// gibbsPotts1d
SEXP gibbsPotts1d(SEXP yS, SEXP zS, SEXP betaS, SEXP muS, SEXP sdS, SEXP nS, SEXP bS, SEXP prS, SEXP itS);
RcppExport SEXP _multiPotts_gibbsPotts1d(SEXP ySSEXP, SEXP zSSEXP, SEXP betaSSEXP, SEXP muSSEXP, SEXP sdSSEXP, SEXP nSSEXP, SEXP bSSEXP, SEXP prSSEXP, SEXP itSSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type yS(ySSEXP);
    Rcpp::traits::input_parameter< SEXP >::type zS(zSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type betaS(betaSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type muS(muSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type sdS(sdSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nS(nSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type bS(bSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type prS(prSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type itS(itSSEXP);
    rcpp_result_gen = Rcpp::wrap(gibbsPotts1d(yS, zS, betaS, muS, sdS, nS, bS, prS, itS));
    return rcpp_result_gen;
END_RCPP
}
// mdgibbsPotts
SEXP mdgibbsPotts(SEXP yS, SEXP betaS, SEXP muS, SEXP sigmaS, SEXP nS, SEXP bS, SEXP prS, SEXP itS);
RcppExport SEXP _multiPotts_mdgibbsPotts(SEXP ySSEXP, SEXP betaSSEXP, SEXP muSSEXP, SEXP sigmaSSEXP, SEXP nSSEXP, SEXP bSSEXP, SEXP prSSEXP, SEXP itSSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type yS(ySSEXP);
    Rcpp::traits::input_parameter< SEXP >::type betaS(betaSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type muS(muSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type sigmaS(sigmaSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nS(nSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type bS(bSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type prS(prSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type itS(itSSEXP);
    rcpp_result_gen = Rcpp::wrap(mdgibbsPotts(yS, betaS, muS, sigmaS, nS, bS, prS, itS));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_hello_world
arma::mat rcpparma_hello_world();
RcppExport SEXP _multiPotts_rcpparma_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpparma_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_outerproduct
arma::mat rcpparma_outerproduct(const arma::colvec& x);
RcppExport SEXP _multiPotts_rcpparma_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_innerproduct
double rcpparma_innerproduct(const arma::colvec& x);
RcppExport SEXP _multiPotts_rcpparma_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_bothproducts
Rcpp::List rcpparma_bothproducts(const arma::colvec& x);
RcppExport SEXP _multiPotts_rcpparma_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_multiPotts_gibbsPotts1d", (DL_FUNC) &_multiPotts_gibbsPotts1d, 9},
    {"_multiPotts_mdgibbsPotts", (DL_FUNC) &_multiPotts_mdgibbsPotts, 8},
    {"_multiPotts_rcpparma_hello_world", (DL_FUNC) &_multiPotts_rcpparma_hello_world, 0},
    {"_multiPotts_rcpparma_outerproduct", (DL_FUNC) &_multiPotts_rcpparma_outerproduct, 1},
    {"_multiPotts_rcpparma_innerproduct", (DL_FUNC) &_multiPotts_rcpparma_innerproduct, 1},
    {"_multiPotts_rcpparma_bothproducts", (DL_FUNC) &_multiPotts_rcpparma_bothproducts, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_multiPotts(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// GibbsGMM
SEXP GibbsGMM(SEXP yS, SEXP prS, SEXP itS, SEXP biS, SEXP salitS);
RcppExport SEXP _multiPotts_GibbsGMM(SEXP ySSEXP, SEXP prSSEXP, SEXP itSSEXP, SEXP biSSEXP, SEXP salitSSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type yS(ySSEXP);
    Rcpp::traits::input_parameter< SEXP >::type prS(prSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type itS(itSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type biS(biSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type salitS(salitSSEXP);
    rcpp_result_gen = Rcpp::wrap(GibbsGMM(yS, prS, itS, biS, salitS));
    return rcpp_result_gen;
END_RCPP
}
// GibbPotts
SEXP GibbPotts(SEXP yS, SEXP betaS, SEXP muS, SEXP sigmaS, SEXP nS, SEXP bS, SEXP prS, SEXP itS, SEXP biS, SEXP salitS);
RcppExport SEXP _multiPotts_GibbPotts(SEXP ySSEXP, SEXP betaSSEXP, SEXP muSSEXP, SEXP sigmaSSEXP, SEXP nSSEXP, SEXP bSSEXP, SEXP prSSEXP, SEXP itSSEXP, SEXP biSSEXP, SEXP salitSSEXP) {
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
    Rcpp::traits::input_parameter< SEXP >::type biS(biSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type salitS(salitSSEXP);
    rcpp_result_gen = Rcpp::wrap(GibbPotts(yS, betaS, muS, sigmaS, nS, bS, prS, itS, biS, salitS));
    return rcpp_result_gen;
END_RCPP
}
// mcmcPotts1d
SEXP mcmcPotts1d(SEXP yS, SEXP nS, SEXP bS, SEXP itS, SEXP biS, SEXP prS, SEXP mhS, SEXP salitS);
RcppExport SEXP _multiPotts_mcmcPotts1d(SEXP ySSEXP, SEXP nSSEXP, SEXP bSSEXP, SEXP itSSEXP, SEXP biSSEXP, SEXP prSSEXP, SEXP mhSSEXP, SEXP salitSSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type yS(ySSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nS(nSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type bS(bSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type itS(itSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type biS(biSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type prS(prSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type mhS(mhSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type salitS(salitSSEXP);
    rcpp_result_gen = Rcpp::wrap(mcmcPotts1d(yS, nS, bS, itS, biS, prS, mhS, salitS));
    return rcpp_result_gen;
END_RCPP
}
// mcmcPottsmd
SEXP mcmcPottsmd(SEXP yS, SEXP nS, SEXP bS, SEXP itS, SEXP biS, SEXP prS, SEXP mhS, SEXP salitS);
RcppExport SEXP _multiPotts_mcmcPottsmd(SEXP ySSEXP, SEXP nSSEXP, SEXP bSSEXP, SEXP itSSEXP, SEXP biSSEXP, SEXP prSSEXP, SEXP mhSSEXP, SEXP salitSSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type yS(ySSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nS(nSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type bS(bSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type itS(itSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type biS(biSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type prS(prSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type mhS(mhSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type salitS(salitSSEXP);
    rcpp_result_gen = Rcpp::wrap(mcmcPottsmd(yS, nS, bS, itS, biS, prS, mhS, salitS));
    return rcpp_result_gen;
END_RCPP
}
// MCMCPotts
SEXP MCMCPotts(SEXP yS, SEXP nS, SEXP bS, SEXP itS, SEXP biS, SEXP prS, SEXP mhS, SEXP salitS);
RcppExport SEXP _multiPotts_MCMCPotts(SEXP ySSEXP, SEXP nSSEXP, SEXP bSSEXP, SEXP itSSEXP, SEXP biSSEXP, SEXP prSSEXP, SEXP mhSSEXP, SEXP salitSSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type yS(ySSEXP);
    Rcpp::traits::input_parameter< SEXP >::type nS(nSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type bS(bSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type itS(itSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type biS(biSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type prS(prSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type mhS(mhSSEXP);
    Rcpp::traits::input_parameter< SEXP >::type salitS(salitSSEXP);
    rcpp_result_gen = Rcpp::wrap(MCMCPotts(yS, nS, bS, itS, biS, prS, mhS, salitS));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_multiPotts_GibbsGMM", (DL_FUNC) &_multiPotts_GibbsGMM, 5},
    {"_multiPotts_GibbPotts", (DL_FUNC) &_multiPotts_GibbPotts, 10},
    {"_multiPotts_mcmcPotts1d", (DL_FUNC) &_multiPotts_mcmcPotts1d, 8},
    {"_multiPotts_mcmcPottsmd", (DL_FUNC) &_multiPotts_mcmcPottsmd, 8},
    {"_multiPotts_MCMCPotts", (DL_FUNC) &_multiPotts_MCMCPotts, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_multiPotts(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

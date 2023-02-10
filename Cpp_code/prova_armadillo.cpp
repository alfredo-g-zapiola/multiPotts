#include <RcppArmadillo.h>   

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
void hello_world() {
  Rcpp::Rcout << "Hello World!" << std::endl;  
}

// After compile, this function will be immediately called using
// the below snippet and results will be sent to the R console.

/*** R
hello_world() 
*/


//to test this in the R console do:
// library(Rccp)
// library(RccpArmadilo)
// sourceCpp(file = "/Users/macbookpro/Documents/Bayesian Statistics/Project/Cpp_code/prova_armadillo.cpp") your path
// you should see helo world

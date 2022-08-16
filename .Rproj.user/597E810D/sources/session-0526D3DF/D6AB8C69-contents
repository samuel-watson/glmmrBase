#include <RcppArmadillo.h>
#include "glmmr.h"
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

//' Combines a field of matrices into a block diagonal matrix
//' 
//' Combines a field of matrices into a block diagonal matrix. Used on
//' the output of `genD`
//' @param matfield A field of matrices
//' @return A block diagonal matrix
// [[Rcpp::export]]
arma::mat blockMat(arma::field<arma::mat> matfield){
  arma::uword nmat = matfield.n_rows;
  if(nmat==1){
    return matfield(0);
  } else {
    arma::mat mat1 = matfield(0);
    arma::mat mat2;
    if(nmat==2){
      mat2 = matfield(1);
    } else {
      mat2 = blockMat(matfield.rows(1,nmat-1));
    }
    arma::uword n1 = mat1.n_rows;
    arma::uword n2 = mat2.n_rows;
    arma::mat dmat(n1+n2,n1+n2);
    dmat.fill(0);
    dmat.submat(0,0,n1-1,n1-1) = mat1;
    dmat.submat(n1,n1,n1+n2-1,n1+n2-1) = mat2;
    return dmat;
  }
}
#ifndef GLMMRSPARSE_H
#define GLMMRSPARSE_H

#include "general.h"
// extends some of the SparseChol functions and operators to Eigen classes

namespace glmmr {

inline MatrixXd sparse_to_dense(const sparse& m,
                                bool symmetric = true){
  MatrixXd D = MatrixXd::Zero(m.n,m.m);
  for(int i = 0; i < m.n; i++){
    for(int j = m.Ap[i]; j < m.Ap[i+1]; j++){
      D(i,m.Ai[j]) = m.Ax[j];
      if(symmetric && m.Ai[j]!=i) D(m.Ai[j],i) = D(m.Ai[j],i);
    }
  }
  return D;
}

inline sparse dense_to_sparse(const MatrixXd& A,
                              bool symmetric = true){
  sparse As(A.rows(),A.cols(),A.data(),true); // this doesn't account for symmetric yet
  return As;
}

inline MatrixXd operator*(const sparse& A, const MatrixXd& B){
  MatrixXd AB(A.n,B.cols());
  AB.setZero();
  int i,j,k;
  for(i = 0; i < A.n; i++){
    for(j = A.Ap[i]; j < A.Ap[i+1]; j++){
      for(k = 0; k<B.cols(); k++){
        AB(i,k) += A.Ax[j]*B(A.Ai[j],k);
      }
    }
  }
  return AB;
}

inline VectorXd operator*(const sparse& A, const VectorXd& B){
  if(A.m != B.size())Rcpp::stop("wrong dimension in sparse-vectorxd multiplication");
  VectorXd AB = VectorXd::Zero(A.n);
  double val;
  int i,j;
  for(i = 0; i < A.n; i++){
    for(j = A.Ap[i]; j < A.Ap[i+1]; j++){
      AB(i) += A.Ax[j]*B(A.Ai[j]);
    }
  }
  return AB;
}

inline ArrayXd operator*(const sparse& A, const ArrayXd& B){
  ArrayXd AB = ArrayXd::Zero(A.n);
  int i,j;
  for(i = 0; i < A.n; i++){
    for(j = A.Ap[i]; j < A.Ap[i+1]; j++){
      AB(i) += A.Ax[j]*B(A.Ai[j]);
    }
  }
  return AB;
}

// multiplication of sparse and diagonal of a vector
inline sparse operator%(const sparse& A, const VectorXd& x){
  sparse Ax(A);
  for(int i = 0; i < A.Ax.size(); i++){
    Ax.Ax[i] *= x(Ax.Ai[i]);
  }
  return Ax;
}

}

#endif
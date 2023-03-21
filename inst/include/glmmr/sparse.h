#ifndef GLMMRSPARSE_H
#define GLMMRSPARSE_H

#define MIN(a,b) ((a) < (b) ? (a) : (b))

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


inline void mat_mat_mult(const double* a, const double* b, double* ab,
                         const int* Ai, const int* Ap, 
                         int n, int m){
  double val;
  for(int i = 0; i < n; i++){
    for(int j = Ap[i]; j < Ap[i+1]; j++){
      val = a[j];
      for(int k = 0; k<m; k++){
        ab[i+k*n] += val*b[Ai[j]+k*m];
      }
    }
  }

}

inline MatrixXd operator*(const sparse& A, const MatrixXd& B){
  if(A.m != B.rows())Rcpp::stop("Bad dimension");
  int m = B.cols();
  int n = A.n;
  // dblvec ab(A.n*m,0);
  // mat_mat_mult(&A.Ax[0],B.data(),&ab[0],&A.Ai[0],&A.Ap[0],n,m);
  MatrixXd AB(A.n,m);
  // AB = Map<MatrixXd>(ab.data(),A.n,m);
  AB.setZero();
  double val;
    for(int i = 0; i < A.n; i++){
      for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){
        val = A.Ax[j];
        for(int k = 0; k<m; k++){
          AB(i,k) += val*B(A.Ai[j],k);
        }
      }
    }
  return AB;
    
}

inline VectorXd operator*(const sparse& A, const VectorXd& B){
  if(A.m != B.size())Rcpp::stop("wrong dimension in sparse-vectorxd multiplication");
  VectorXd AB = VectorXd::Zero(A.n);
  double val;
  for(int i = 0; i < A.n; i++){
    for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){
      AB(i) += A.Ax[j]*B(A.Ai[j]);
    }
  }
  return AB;
}

inline ArrayXd operator*(const sparse& A, const ArrayXd& B){
  ArrayXd AB = ArrayXd::Zero(A.n);
  for(int i = 0; i < A.n; i++){
    for(int j = A.Ap[i]; j < A.Ap[i+1]; j++){
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
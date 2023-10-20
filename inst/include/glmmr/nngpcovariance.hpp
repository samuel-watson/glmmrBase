#pragma once

#include "covariance.hpp"
#include "griddata.hpp"

namespace glmmr {

using namespace Eigen;

class nngpCovariance : public Covariance {
public:
  glmmr::griddata grid;
  MatrixXd A;
  VectorXd Dvec;
  int m = 10;
  
  nngpCovariance(const glmmr::Formula& formula,
                 const ArrayXXd &data,
                 const strvec& colnames,
                 const dblvec& parameters) : Covariance(formula, data, colnames, parameters),  
                  A(10,data.rows()), Dvec(data.rows()) {
    isSparse = false;
    parse_grid_data(data);
    gen_AD();
  }
  
  nngpCovariance(const glmmr::Formula& formula,
                 const ArrayXXd &data,
                 const strvec& colnames) : Covariance(formula, data, colnames),  
                 A(10,data.rows()), Dvec(data.rows()) {
    isSparse = false;
    parse_grid_data(data);
  }
  
  nngpCovariance(const glmmr::nngpCovariance& cov) : Covariance(cov.form_, cov.data_, cov.colnames_, cov.parameters_),  
    grid(cov.grid), A(grid.m,grid.N), Dvec(grid.N), m(cov.m) {
    isSparse = false;
    grid.genNN(m);
    gen_AD();
  }
  
  MatrixXd D(bool chol = true, bool upper = false) override;
  MatrixXd ZL() override;
  MatrixXd LZWZL(const VectorXd& w) override;
  MatrixXd ZLu(const MatrixXd& u) override;
  MatrixXd Lu(const MatrixXd& u) override;
  sparse ZL_sparse() override;
  int Q() override;
  double log_likelihood(const VectorXd &u) override;
  double log_determinant() override;
  void gen_AD();
  void gen_NN(int nn);
  void update_parameters(const dblvec& parameters) override;
  void update_parameters(const ArrayXd& parameters) override;
  void update_parameters_extern(const dblvec& parameters) override;
  vector_matrix submatrix(int i);
  MatrixXd inv_ldlt_AD(const MatrixXd &A,const VectorXd &D,const ArrayXXi &NN);
  void parse_grid_data(const ArrayXXd &data);
  
};

}

inline void glmmr::nngpCovariance::parse_grid_data(const ArrayXXd &data){
  int dim = this->block_nvar[0];
  ArrayXXd grid_data(data.rows(),dim);
  for(int i = 0; i < dim; i++){
    grid_data.col(i) = data.col(this->block_nvar[0]);
  }
  grid.setup(grid_data,10);
}

inline void glmmr::nngpCovariance::gen_NN(int nn){
  m = nn;
  A.conservativeResize(nn,grid.N);
  grid.genNN(m);
}

inline MatrixXd glmmr::nngpCovariance::D(bool chol, bool upper){
  MatrixXd As = inv_ldlt_AD(A,Dvec,grid.NN);
  if(chol){
    if(upper){
      return As.transpose();
    } else {
      return As;
    }
  } else {
    return As * As.transpose();
  }
}

inline MatrixXd glmmr::nngpCovariance::ZL(){
  MatrixXd L = sparse_to_dense(this->matL);
  return L;
}

inline MatrixXd glmmr::nngpCovariance::LZWZL(const VectorXd& w){
  MatrixXd ZL = sparse_to_dense(this->matL);
  MatrixXd LZWZL = ZL.transpose() * w.asDiagonal() * ZL;
  LZWZL += MatrixXd::Identity(LZWZL.rows(), LZWZL.cols());
  return LZWZL;
}

inline MatrixXd glmmr::nngpCovariance::ZLu(const MatrixXd& u){
  MatrixXd ZLu = this->matL * u;
  return ZLu;
}

inline MatrixXd glmmr::nngpCovariance::Lu(const MatrixXd& u){
  MatrixXd L = sparse_to_dense(this->matL);
  return L*u;
}

inline sparse glmmr::nngpCovariance::ZL_sparse(){
  return this->matL;
}

inline int glmmr::nngpCovariance::Q(){
  return grid.N;
}

inline double glmmr::nngpCovariance::log_likelihood(const VectorXd &u){
  double ll1 = 0.0;
  double logdet = log_determinant();
  int idxlim;
  double qf = u(0)*u(0)/Dvec(0);

#pragma omp parallel for reduction (+:qf)
  for(int i = 1; i < grid.N; i++){
    idxlim = i <= m ? i : m;
    VectorXd usec(idxlim);
    for(int j = 0; j < idxlim; j++) usec(j) = u(grid.NN(j,i));
    double au = u(i) - (A.col(i).segment(0,idxlim).transpose() * usec)(0);
    qf += au*au/Dvec(i);
  }
  ll1 -= 0.5*qf + 0.5*grid.N*log(2*M_PI);
  ll1 -= 0.5*logdet;
  return ll1;
}

inline double glmmr::nngpCovariance::log_determinant(){
  return Dvec.array().log().sum();
}

inline void glmmr::nngpCovariance::gen_AD(){
  A.setZero();
  Dvec.setZero();
  
  int idxlim;
  double val = Covariance::get_val(0,0,0);
  Dvec(0) = val;
  
#pragma omp parallel for
  for(int i = 1; i < grid.N; i++){
    idxlim = i <= m ? i : m;
    MatrixXd S(idxlim,idxlim);
    VectorXd Sv(idxlim);
    for(int j = 0; j<idxlim; j++){
      S(j,j) = val;
    }
    if(idxlim > 1){
      for(int j = 0; j<(idxlim-1); j++){
        for(int k = j+1; k<idxlim; k++){
          S(j,k) = Covariance::get_val(0,grid.NN(j,i),grid.NN(k,i));
          S(k,j) = S(j,k);
        }
      }
    }
    for(int j = 0; j<idxlim; j++){
      Sv(j) = Covariance::get_val(0,i,grid.NN(j,i));
    }
    A.block(0,i,idxlim,1) = S.ldlt().solve(Sv);
    Dvec(i) = val - (A.col(i).segment(0,idxlim).transpose() * Sv)(0);
  }
  this->matL = dense_to_sparse(D(true,false),false);
}

inline vector_matrix glmmr::nngpCovariance::submatrix(int i){
  int idxlim = i <= m ? i : m;
  double val = Covariance::get_val(0,0,0);
  Dvec(0) = val;
  MatrixXd S(idxlim,idxlim);
  VectorXd Sv(idxlim);
  for(int j = 0; j<idxlim; j++){
    S(j,j) = val;
  }
  if(idxlim > 1){
    for(int j = 0; j<(idxlim-1); j++){
      for(int k = j+1; k<idxlim; k++){
        S(j,k) = Covariance::get_val(0,grid.NN(j,i),grid.NN(k,i));
        S(k,j) = S(j,k);
      }
    }
  }
  for(int j = 0; j<idxlim; j++){
    Sv(j) = Covariance::get_val(0,i,grid.NN(j,i));
  }
  vector_matrix result(idxlim);
  result.vec = Sv;
  result.mat = S;
  return result;
}

inline void glmmr::nngpCovariance::update_parameters(const dblvec& parameters){
  parameters_ = parameters;
  update_parameters_in_calculators();
  gen_AD();
}

inline void glmmr::nngpCovariance::update_parameters_extern(const dblvec& parameters){
  parameters_ = parameters;
  update_parameters_in_calculators();
  gen_AD();
}

inline void glmmr::nngpCovariance::update_parameters(const ArrayXd& parameters){
  if(parameters_.size()==0){
    for(unsigned int i = 0; i < parameters.size(); i++){
      parameters_.push_back(parameters(i));
    }
    update_parameters_in_calculators();
  } else if(parameters_.size() == parameters.size()){
    for(unsigned int i = 0; i < parameters.size(); i++){
      parameters_[i] = parameters(i);
    }
    update_parameters_in_calculators();
  } 
  gen_AD();
};

inline MatrixXd glmmr::nngpCovariance::inv_ldlt_AD(const MatrixXd &A, 
                                                 const VectorXd &D,
                                                 const ArrayXXi &NN){
  int n = A.cols();
  int m = A.rows();
  MatrixXd y = MatrixXd::Zero(n,n);
  ArrayXd dsqrt = Dvec.array().sqrt();
#pragma omp parallel for  
  for(int k=0; k<n; k++){
    int idxlim;
    for (int i = 0; i < n; i++) {
      idxlim = i<=m ? i : m;
      double lsum = 0;
      for (int j = 0; j < idxlim; j++) {
        lsum += A(j,i) * y(NN(j,i),k);
      }
      y(i,k) = i==k ? (1+lsum)*dsqrt(k)  : lsum*dsqrt(k) ;
    }
  }
  
  return y;
}
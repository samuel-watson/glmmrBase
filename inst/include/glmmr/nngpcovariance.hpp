#pragma once

#include "covariance.hpp"
#include "matrixfield.h"
#include "griddata.hpp"

namespace glmmr {

using namespace Eigen;

class nngpCovariance : public Covariance {
public:
  glmmr::griddata   grid;
  MatrixXd          A;
  VectorXd          Dvec;
  int               m = 10;

  nngpCovariance(const glmmr::Formula& formula,const ArrayXXd &data,const strvec& colnames,const dblvec& parameters);
  nngpCovariance(const glmmr::Formula& formula,const ArrayXXd &data,const strvec& colnames);
  nngpCovariance(const glmmr::nngpCovariance& cov);

  MatrixXd      D(bool chol = true, bool upper = false) override;
  MatrixXd      ZL() override;
  MatrixXd      LZWZL(const VectorXd& w) override;
  MatrixXd      ZLu(const MatrixXd& u) override;
  MatrixXd      Lu(const MatrixXd& u) override;
  VectorXd      sim_re() override;
  sparse        ZL_sparse() override;
  int           Q() const override;
  double        log_likelihood(const VectorXd &u) override;
  double        log_determinant() override;
  void          gen_AD();
  void          gen_NN(int nn);
  void          update_parameters(const dblvec& parameters) override;
  void          update_parameters(const ArrayXd& parameters) override;
  void          update_parameters_d(const ArrayXd& parameters);
  void          update_parameters_extern(const dblvec& parameters) override;
  VectorMatrix  submatrix(int i);
  MatrixXd      inv_ldlt_AD(const MatrixXd &A,const VectorXd &D,const ArrayXXi &NN);
  void          parse_grid_data(const ArrayXXd &data);
  void          gen_AD_derivatives(glmmr::MatrixField<VectorXd>& dD, glmmr::MatrixField<MatrixXd>& dA); 
  VectorXd      log_gradient(const MatrixXd& u, double& ll) override;
};

}

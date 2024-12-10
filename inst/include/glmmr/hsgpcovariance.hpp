#pragma once

#include "covariance.hpp"

namespace glmmr {

using namespace Eigen;

class hsgpCovariance : public Covariance {
public:
// data
  int       dim;
  intvec    m;
  ArrayXXd  hsgp_data;
  ArrayXd   L_boundary;
  //constructors
  hsgpCovariance(const std::string& formula,const ArrayXXd& data,const strvec& colnames);
  hsgpCovariance(const glmmr::Formula& formula,const ArrayXXd& data,const strvec& colnames);
  hsgpCovariance(const std::string& formula,const ArrayXXd& data,const strvec& colnames,const dblvec& parameters);
  hsgpCovariance(const glmmr::Formula& formula,const ArrayXXd& data,const strvec& colnames,const dblvec& parameters);
  hsgpCovariance(const glmmr::hsgpCovariance& cov);
  // functions
  double      spd_nD(int i);
  double      d_spd_nD(int i, int par, bool sqrt_lambda = true);
  ArrayXd     phi_nD(int i);
  MatrixXd    ZL() override;
  MatrixXd    ZL_deriv(int par);
  MatrixXd    D(bool chol = true, bool upper = false) override;
  MatrixXd    LZWZL(const VectorXd& w) override;
  MatrixXd    ZLu(const MatrixXd& u) override;
  MatrixXd    Lu(const MatrixXd& u) override;
  VectorXd    sim_re() override;
  sparse      ZL_sparse() override;
  int         Q() const override;
  double      log_likelihood(const VectorXd &u) override;
  double      log_determinant() override;
  void        update_parameters(const dblvec& parameters) override;
  void        update_parameters(const ArrayXd& parameters) override;
  void        update_parameters_extern(const dblvec& parameters) override;
  void        set_function(bool squared_exp);
  MatrixXd    PhiSPD(bool lambda = true, bool inverse = false);
  ArrayXd     LambdaSPD();
  void        update_approx_parameters(intvec m_, ArrayXd L_);
  void        update_approx_parameters();
protected:
//data
  int       total_m;
  MatrixXd  L; // Half-eigen decomposition of Lambda + PhiTPhi m^2 * m^2
  ArrayXd   Lambda;
  ArrayXXi  indices;
  MatrixXd  Phi;
  MatrixXd  PhiT;
  bool      sq_exp = false;
  //functions
  void      parse_hsgp_data();
  void      gen_indices();
  void      gen_phi_prod();
  void      update_lambda();
};

}



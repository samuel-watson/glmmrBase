#pragma once

#include "covariance.hpp"

namespace glmmr {

using namespace Eigen;
using namespace glmmr;

//kroenecker product
inline MatrixXd kronecker(const MatrixXd& A, const MatrixXd& B){
  MatrixXd result = MatrixXd::Zero(A.rows()*B.rows(), A.cols()*B.cols());
#pragma omp parallel for
  for(int i = 0; i < A.rows(); i++){
    for(int j = 0; j < A.cols(); j++){
      result.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j) * B;
    }
  }
  return result;
}

class ar1Covariance : public Covariance {
public:
  double rho;
  
  ar1Covariance(const str& formula,const ArrayXXd &data,const strvec& colnames, const int T_);  
  ar1Covariance(const glmmr::ar1Covariance& cov);
  
  MatrixXd  ZL() override;
  MatrixXd  ZLu(const MatrixXd& u) override;
  MatrixXd  Lu(const MatrixXd& u) override;
  int       Q() const override;
  double    log_likelihood(const VectorXd &u) override;
  double    log_determinant() override;
  void      update_rho(const double rho_);
  MatrixXd  ar_matrix(bool chol = false);
  void      nr_step(const MatrixXd &umat,const MatrixXd &vmat, ArrayXd& logl, ArrayXd& gradients, const ArrayXd& uweight) override;
  MatrixXd  solve(const MatrixXd& u) override;
  
protected:
  int T;
  MatrixXd ar_factor;    
  MatrixXd ar_factor_chol;
  MatrixXd ar_factor_deriv;  
  MatrixXd ar_factor_inverse;  
  VectorXd uquad;
  VectorXd vquad;
};

}


inline glmmr::ar1Covariance::ar1Covariance(const str& formula,
                                         const ArrayXXd &data,
                                         const strvec& colnames, int T_) : Covariance(formula, data, colnames), 
                                         T(T_), 
                                         ar_factor(T,T), ar_factor_chol(T,T), ar_factor_deriv(T,T), ar_factor_inverse(T,T),
                                         uquad(Covariance::Q()), vquad(Covariance::Q()){ 
  isSparse = false;
  update_rho(0.1);
};

inline glmmr::ar1Covariance::ar1Covariance(const glmmr::ar1Covariance& cov) : Covariance(cov.form_, cov.data_, cov.colnames_), 
T(cov.T), ar_factor(cov.ar_factor), ar_factor_chol(cov.ar_factor_chol), ar_factor_deriv(cov.ar_factor_deriv), ar_factor_inverse(cov.ar_factor_inverse),
uquad(cov.uquad), vquad(cov.vquad){
  update_rho(cov.rho);
};

inline MatrixXd glmmr::ar1Covariance::ar_matrix(bool chol)
{
  if(chol){
    return ar_factor_chol;
  } else {
    return ar_factor;
  }
}

inline MatrixXd glmmr::ar1Covariance::ZL()
{
  
  
  const int n_total = matZ.rows();
  const int n_A = Covariance::Q();
  const int n_t = n_total / T;
  
  MatrixXd Z_full = MatrixXd::Zero(n_total, n_A * T);
  
  for(int t = 0; t < T; t++){
    Z_full.block(t * n_t, t * n_A, n_t, n_A) = matZ.block(t * n_t, 0, n_t, n_A);
  }
  
  MatrixXd L_full = glmmr::kronecker(ar_factor_chol, MatrixXd(matL.matrixL()));
  
  MatrixXd result = Z_full * L_full;
  return result;
}

inline MatrixXd glmmr::ar1Covariance::ZLu(const MatrixXd& u)
{
  const int n_total = matZ.rows();
  const int n_A = Covariance::Q();
  const int n_t = n_total / T;
  const int ncols = u.cols();
  MatrixXd result = MatrixXd::Zero(n_total, ncols);
  
  for(int c = 0; c < ncols; c++){
    Map<const MatrixXd> U(u.col(c).data(), n_A, T);
    MatrixXd LU = matL.matrixL() * U * ar_factor_chol.transpose();
    for(int t = 0; t < T; t++){
      result.block(t * n_t, c, n_t, 1) = matZ.block(t * n_t, 0, n_t, n_A) * LU.col(t);
    }
  }
  return result;
}

inline MatrixXd glmmr::ar1Covariance::Lu(const MatrixXd& u)
{
  const int n_A = Covariance::Q();
  const int ncols = u.cols();
  MatrixXd result(n_A * T, ncols);
  
  for(int c = 0; c < ncols; c++){
    Map<const MatrixXd> U(u.col(c).data(), n_A, T);
    MatrixXd LU = matL.matrixL() * U * ar_factor_chol.transpose();
    result.col(c) = Map<const VectorXd>(LU.data(), n_A * T);
  }
  return result;
}

inline int glmmr::ar1Covariance::Q() const 
{
  //return Covariance::Q() * T;
  int parent_Q = Covariance::Q();
  int q = parent_Q * T;
  return q;
}

inline double glmmr::ar1Covariance::log_likelihood(const VectorXd &u)
{
  static const double LOG_2PI = log(2*M_PI);
  int N = Covariance::Q();
  int NT = N * T;
  Map<const MatrixXd> U(u.data(), N, T);
  MatrixXd W = matL.solve(MatrixXd(U));
  MatrixXd M = U.transpose() * W;
  double qf = (M.array() * ar_factor_inverse.array()).sum();
  
  double logdet_A = Covariance::log_determinant();
  double logdet_R = 2.0 * ar_factor_chol.diagonal().array().log().sum();
  double logdet = T * logdet_A + N * logdet_R;
  
  double ll = -0.5 * (NT * LOG_2PI + logdet + qf);
  
  return ll;
}

inline double glmmr::ar1Covariance::log_determinant()
{
  int N = Covariance::Q();
  double logdet_A = Covariance::log_determinant();
  double logdet_R = 2.0 * ar_factor_chol.diagonal().array().log().sum();
  return T * logdet_A + N * logdet_R;
}

inline void glmmr::ar1Covariance::update_rho(const double rho_)
{
  rho = rho_;
  ar_factor.setConstant(1.0);
  ar_factor_deriv.setZero();
  if(T > 1){
    for(int t = 0; t < T-1; t++){
      for(int s = t+1; s < T; s++){
        ar_factor(t,s) = pow(rho,s - t);
        ar_factor_deriv(t,s) = (s-t)*pow(rho,s-t-1);
        ar_factor(s,t) = ar_factor(t,s);
        ar_factor_deriv(s,t) = ar_factor_deriv(t,s);
      }
    }
  }
  ar_factor_chol = MatrixXd(ar_factor.llt().matrixL());
  ar_factor_inverse = ar_factor.llt().solve(MatrixXd::Identity(T, T));
}

inline void glmmr::ar1Covariance::nr_step(const MatrixXd &umat, const MatrixXd &vmat, ArrayXd& logl, ArrayXd& gradients, 
                                const ArrayXd& uweight){
  static const double LOG_2PI = log(2*M_PI);
  static const double NEG_HALF_LOG_2PI = -0.5 * LOG_2PI;
  
  // Get derivatives of A (spatial) from parent
  std::vector<MatrixXd> derivs_A;
  derivatives(derivs_A, 1);
  const int npars_A = derivs_A.size() - 1;
  const int npars = npars_A + 1;  // +1 for rho
  const int niter = umat.cols();
  const int n_A = Covariance::Q();  // spatial dimension
  
  VectorXd grad = VectorXd::Zero(npars);
  logl.setZero();
  
  if(!isSparse) make_sparse();
  
  // Log determinant: T * log|A| + n_A * log|R|
  double logdet_A = Covariance::log_determinant();  // just the spatial part
  double logdet_R = 2.0 * ar_factor_chol.diagonal().array().log().sum();
  double logdet_val = T * logdet_A + n_A * logdet_R;
  
  logl.array() += NEG_HALF_LOG_2PI * Q() - 0.5 * logdet_val;
  
  // S_A[j] = A^{-1} dA/dtheta_j
  std::vector<MatrixXd> S_A;
  for(int j = 0; j < npars_A; j++){
    S_A.emplace_back(matL.solve(derivs_A[j + 1]));
    // tr(I ⊗ S_A) = T * tr(S_A)
    grad(j) = -0.5 * T * S_A[j].trace();
  }
  
  // S_R = R^{-1} dR/drho
  MatrixXd S_R = ar_factor_inverse * ar_factor_deriv;
  // tr(S_R ⊗ I) = n_A * tr(S_R)
  grad(npars_A) = -0.5 * n_A * S_R.trace();
  
  // Information matrix - trace contributions
  MatrixXd M = MatrixXd::Zero(npars, npars);
  
  // A-A block: tr((I ⊗ S_A^j)(I ⊗ S_A^k)) = T * tr(S_A^j S_A^k)
  for(int j = 0; j < npars_A; j++){
    for(int k = j; k < npars_A; k++){
      double tr_val = -0.5 * T * (S_A[j] * S_A[k]).trace();
      M(j, k) = tr_val;
      if(j != k) M(k, j) = tr_val;
    }
  }
  
  // A-R cross: tr((I ⊗ S_A)(S_R ⊗ I)) = tr(S_R ⊗ S_A) = tr(S_R) * tr(S_A)
  double tr_S_R = S_R.trace();
  for(int j = 0; j < npars_A; j++){
    double tr_val = -0.5 * S_A[j].trace() * tr_S_R;
    M(j, npars_A) = tr_val;
    M(npars_A, j) = tr_val;
  }
  
  // R-R: tr((S_R ⊗ I)^2) = n_A * tr(S_R^2)
  MatrixXd S_R_sq = S_R * S_R;
  M(npars_A, npars_A) = -0.5 * n_A * S_R_sq.trace();
  
  // Monte Carlo contributions - quadratic form
#pragma omp parallel for
  for(int i = 0; i < niter; i++){
    double qf = vmat.col(i).dot(umat.col(i));
    logl(i) += -0.5 * qf;
  }
  
  // Gradient MC terms
  // Vectors reshaped as n_A x T matrices (spatial x temporal)
  // For D = R ⊗ A, D^{-1} dD/dθ_A = I ⊗ S_A, D^{-1} dD/dρ = S_R ⊗ I
  //
  // u^T (I ⊗ S_A) v: using (I ⊗ S_A) vec(V) = vec(S_A V)
  //   = vec(U)^T vec(S_A V) = tr(U^T S_A V)
  //
  // u^T (S_R ⊗ I) v: using (S_R ⊗ I) vec(V) = vec(V S_R^T)
  //   = vec(U)^T vec(V S_R^T) = tr(U^T V S_R^T)
  
  for(int j = 0; j < npars_A; j++){
    double grad_j = 0.0;
#pragma omp parallel for reduction(+:grad_j)
    for(int i = 0; i < niter; i++){
      Map<const MatrixXd> U(umat.col(i).data(), n_A, T);
      Map<const MatrixXd> V(vmat.col(i).data(), n_A, T);
      // tr(U^T S_A V)
      grad_j += uweight(i) * (U.transpose() * S_A[j] * V).trace();
    }
    grad(j) += 0.5 * grad_j;
  }
  
  double grad_rho = 0.0;
#pragma omp parallel for reduction(+:grad_rho)
  for(int i = 0; i < niter; i++){
    Map<const MatrixXd> U(umat.col(i).data(), n_A, T);
    Map<const MatrixXd> V(vmat.col(i).data(), n_A, T);
    // tr(U^T V S_R^T) = tr(S_R^T U^T V)
    grad_rho += uweight(i) * (U.transpose() * V * S_R.transpose()).trace();
  }
  grad(npars_A) += 0.5 * grad_rho;
  
  // Information matrix MC terms
  //
  // A-A block: u^T (I ⊗ S_A^j S_A^k) v = tr(U^T S_A^j S_A^k V)
  //
  // A-R cross: u^T (S_R ⊗ S_A) v: using (S_R ⊗ S_A) vec(V) = vec(S_A V S_R^T)
  //   = tr(U^T S_A V S_R^T)
  //
  // R-R: u^T (S_R^2 ⊗ I) v = tr(U^T V (S_R^2)^T)
  
  for(int j = 0; j < npars_A; j++){
    for(int k = j; k < npars_A; k++){
      MatrixXd Sprod = S_A[j] * S_A[k];
      double m_jk = 0.0;
#pragma omp parallel for reduction(+:m_jk)
      for(int i = 0; i < niter; i++){
        Map<const MatrixXd> U(umat.col(i).data(), n_A, T);
        Map<const MatrixXd> V(vmat.col(i).data(), n_A, T);
        // tr(U^T S_A^j S_A^k V)
        m_jk += uweight(i) * (U.transpose() * Sprod * V).trace();
      }
      M(j, k) += m_jk;
      if(j != k) M(k, j) += m_jk;
    }
  }
  
  // A-R cross
  for(int j = 0; j < npars_A; j++){
    double m_jr = 0.0;
#pragma omp parallel for reduction(+:m_jr)
    for(int i = 0; i < niter; i++){
      Map<const MatrixXd> U(umat.col(i).data(), n_A, T);
      Map<const MatrixXd> V(vmat.col(i).data(), n_A, T);
      // tr(U^T S_A V S_R^T)
      m_jr += uweight(i) * (U.transpose() * S_A[j] * V * S_R.transpose()).trace();
    }
    M(j, npars_A) += m_jr;
    M(npars_A, j) += m_jr;
  }
  
  // R-R
  double m_rr = 0.0;
#pragma omp parallel for reduction(+:m_rr)
  for(int i = 0; i < niter; i++){
    Map<const MatrixXd> U(umat.col(i).data(), n_A, T);
    Map<const MatrixXd> V(vmat.col(i).data(), n_A, T);
    // tr(U^T V (S_R^2)^T)
    m_rr += uweight(i) * (U.transpose() * V * S_R_sq.transpose()).trace();
  }
  M(npars_A, npars_A) += m_rr;
  
  gradients.tail(npars) = grad;
  infomat_theta = M;
  
  // Update parameters
  VectorXd theta_A = Map<VectorXd>(parameters_.data(), parameters_.size());
  
  // Solve for full step
  VectorXd step = M.llt().solve(grad);
  
  // Update A parameters (first npars_A elements of step)
  theta_A += step.head(npars_A);
  update_parameters(theta_A.array());
  
  // Update rho (last element of step)
  double newrho = rho + step(npars_A);
  
  // Constrain rho to (-1, 1) for stationarity
  if(newrho >= 1.0) newrho = 0.99;
  if(newrho <= -1.0) newrho = -0.99;
  
  // Recompute R matrices with new rho
  update_rho(newrho);
}

inline MatrixXd glmmr::ar1Covariance::solve(const MatrixXd& u){
  const int n_A = Covariance::Q();
  const int ncols = u.cols();
  
  MatrixXd result(u.rows(), ncols);
  
  for(int i = 0; i < ncols; i++){
    Map<const MatrixXd> U(u.col(i).data(), n_A, T);
    MatrixXd X = matL.solve(MatrixXd(U));
    MatrixXd V = X * ar_factor_inverse;
    result.col(i) = Map<const VectorXd>(V.data(), n_A * T);
  }
  
  return result;
}

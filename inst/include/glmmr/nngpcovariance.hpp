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
  void          nr_step(const MatrixXd &umat, ArrayXd& logl) override;
};

}

inline glmmr::nngpCovariance::nngpCovariance(const glmmr::Formula& formula,
               const ArrayXXd &data,
               const strvec& colnames,
               const dblvec& parameters) : Covariance(formula, data, colnames, parameters),  
               A(10,data.rows()), Dvec(data.rows()) {
  isSparse = false;
  parse_grid_data(data);
  gen_AD();
}

inline glmmr::nngpCovariance::nngpCovariance(const glmmr::Formula& formula,
               const ArrayXXd &data,
               const strvec& colnames) : Covariance(formula, data, colnames),  
               A(10,data.rows()), Dvec(data.rows()) {
  isSparse = false;
  parse_grid_data(data);
}

inline glmmr::nngpCovariance::nngpCovariance(const glmmr::nngpCovariance& cov) : Covariance(cov.form_, cov.data_, cov.colnames_, cov.parameters_),  
grid(cov.grid), A(grid.m,grid.N), Dvec(grid.N), m(cov.m) {
  isSparse = false;
  grid.genNN(m);
  gen_AD();
}

inline void glmmr::nngpCovariance::parse_grid_data(const ArrayXXd &data)
{
  int dim = this->block_nvar[0];
  ArrayXXd grid_data(data.rows(),dim);
  for(int i = 0; i < dim; i++){
    grid_data.col(i) = data.col(this->re_cols_data_[0][0][i]);
  }
  grid.setup(grid_data,10);
}

inline void glmmr::nngpCovariance::gen_NN(int nn){
  m = nn;
  A.conservativeResize(nn,NoChange);
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
  MatrixXd L = D(true,false);
  return L;
}

inline MatrixXd glmmr::nngpCovariance::LZWZL(const VectorXd& w){
  MatrixXd ZL = D(true,false);
  MatrixXd LZWZL = ZL.transpose() * w.asDiagonal() * ZL;
  LZWZL += MatrixXd::Identity(LZWZL.rows(), LZWZL.cols());
  return LZWZL;
}

inline MatrixXd glmmr::nngpCovariance::ZLu(const MatrixXd& u){
  MatrixXd L = D(true,false);
  MatrixXd ZLu = L * u;
  return ZLu;
}

inline MatrixXd glmmr::nngpCovariance::Lu(const MatrixXd& u){
  MatrixXd L = D(true,false);
  return L*u;
}

inline VectorXd glmmr::nngpCovariance::sim_re(){
  if(parameters_.size()==0)throw std::runtime_error("no parameters");
  VectorXd samps(this->Q_);
  MatrixXd L = D(true,false);
  std::random_device rd{};
  std::mt19937 gen{ rd() };
  std::normal_distribution d{ 0.0, 1.0 };
  auto random_norm = [&d, &gen] { return d(gen); };
  VectorXd zz(this->Q_);
  for (int j = 0; j < zz.size(); j++) zz(j) = random_norm();;
  samps = L*zz;
  return samps;
}

inline sparse glmmr::nngpCovariance::ZL_sparse(){
  MatrixXd L = D(true,false);
  return dense_to_sparse(L);
}

inline int glmmr::nngpCovariance::Q() const {
  return grid.N;
}

inline double glmmr::nngpCovariance::log_likelihood(const VectorXd &u){
  double logdet = log_determinant();
  int idxlim;
  double qf = u(0)*u(0)/Dvec(0);

//#pragma omp parallel for reduction(+:qf) private(idxlim) 
  for(int i = 1; i < grid.N; i++){
    idxlim = i <= m ? i : m;
    VectorXd usec(idxlim);
    for(int j = 0; j < idxlim; j++) usec(j) = u(grid.NN(j,i));
    double au = u(i) - (A.col(i).segment(0,idxlim).transpose() * usec)(0);
    qf += au*au/Dvec(i);
  }
  double ll1 = 0.5*qf + 0.5*grid.N*log(2*M_PI) + 0.5*logdet;
  return -1.0 * ll1;
}

inline double glmmr::nngpCovariance::log_determinant(){
  return Dvec.array().log().sum();
}

inline void glmmr::nngpCovariance::gen_AD()
{
  A.setZero();
  Dvec.setZero();
  double val = Covariance::get_val(0,0,0);
  Dvec(0) = val;
  
//#pragma omp parallel for
  for(int i = 1; i < grid.N; i++){
    int idxlim = i <= m ? i : m;
    MatrixXd S(idxlim,idxlim);
    VectorXd Sv(idxlim);
    for(int j = 0; j<idxlim; j++)
    {
      S(j,j) = val;
    }
    if(idxlim > 1)
    {
      for(int j = 0; j<(idxlim-1); j++)
      {
        for(int k = j+1; k<idxlim; k++)
        {
          S(j,k) = Covariance::get_val(0,grid.NN(j,i),grid.NN(k,i));
          S(k,j) = S(j,k);
        }
      }
    }
    for(int j = 0; j<idxlim; j++) Sv(j) = Covariance::get_val(0,i,grid.NN(j,i));
    A.col(i).head(idxlim) = S.ldlt().solve(Sv);
    Dvec(i) = val - (A.col(i).head(idxlim).transpose() * Sv)(0);
  }
}

inline VectorMatrix glmmr::nngpCovariance::submatrix(int i)
{
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
  VectorMatrix result(idxlim);
  result.vec = Sv;
  result.mat = S;
  return result;
}

inline void glmmr::nngpCovariance::update_parameters(const dblvec& parameters)
{
  parameters_ = parameters;
  update_parameters_in_calculators();
  gen_AD();
}

inline void glmmr::nngpCovariance::update_parameters_extern(const dblvec& parameters)
{
  parameters_ = parameters;
  update_parameters_in_calculators();
  gen_AD();
}

inline void glmmr::nngpCovariance::update_parameters(const ArrayXd& parameters)
{
  if(parameters_.size()==0){
    for(int i = 0; i < parameters.size(); i++){
      parameters_.push_back(parameters(i));
    }
    update_parameters_in_calculators();
  } else if(parameters_.size() == parameters.size()){
    for(int i = 0; i < parameters.size(); i++){
      parameters_[i] = parameters(i);
    }
    update_parameters_in_calculators();
  } 
  gen_AD();
};

inline void glmmr::nngpCovariance::update_parameters_d(const ArrayXd& parameters)
{
  if(parameters_.size()==0){
    for(int i = 0; i < parameters.size(); i++){
      parameters_.push_back(parameters(i));
    }
    update_parameters_in_calculators();
  } else if(parameters_.size() == parameters.size()){
    for(int i = 0; i < parameters.size(); i++){
      parameters_[i] = parameters(i);
    }
    update_parameters_in_calculators();
  } 
};

inline MatrixXd glmmr::nngpCovariance::inv_ldlt_AD(const MatrixXd &A, 
                                                 const VectorXd &D,
                                                 const ArrayXXi &NN)
                                                 {
  int n = A.cols();
  int m = A.rows();
  MatrixXd y = MatrixXd::Zero(n,n);
  ArrayXd dsqrt = Dvec.array().sqrt();
//#pragma omp parallel for  
  for(int k=0; k<n; k++){
    int idxlim;
    for (int i = 0; i < n; i++) {
      idxlim = i<=m ? i : m;
      double lsum = 0;
      for (int j = 0; j < idxlim; j++) {
        lsum += A(j,i) * y(NN(j,i),k);
      }
      y(i,k) = i==k ? (1+lsum) : lsum ;
    }
  }
  return y * dsqrt.matrix().asDiagonal();
}

inline void glmmr::nngpCovariance::nr_step(const MatrixXd &umat, ArrayXd& logl){

  static const double LOG_2PI = log(2*M_PI);
  static const double NEG_HALF_LOG_2PI = -0.5 * LOG_2PI;
  
  std::vector<MatrixXd> derivs;
  derivatives(derivs, 1);
  int npars = derivs.size() - 1;
  int niter = umat.cols();
  VectorXd grad(npars);
  grad.setZero();
  double logdet_val = 0.0;
  logl.setZero();
  dblvec dqf(npars, 0.0);
  int m = A.rows();
  sparse IA(umat.rows(),umat.rows(),true);
  
  int idxlim;
  for (int i = 0; i < umat.rows(); i++) {
    idxlim = i<=m ? i : m;
    for (int j = 0; j < idxlim; j++) {
      if(grid.NN(j,i)==i){
        IA.insert(grid.NN(j,i),i,1.0 - A(j,i));
      } else {
        IA.insert(grid.NN(j,i),i,-1.0 * A(j,i));
      }
    }
  }
  
  sparse IAt = IA;
  IAt.transpose();
  sparse IAD = IAt % Dvec;
  logdet_val = log_determinant();
  // Precompute S matrices and gradient
  glmmr::MatrixField<MatrixXd> S;
  for(int i = 0; i < npars; i++)
  {
    MatrixXd detmat2 = SparseOperators::operator*(
      SparseOperators::operator*(IAt, derivs[i+1]), 
      IAD
    );
    grad(i) = -0.5 * detmat2.trace();
    S.add(detmat2);
  }
  
  // Update logl with constant terms
  logl.array() += NEG_HALF_LOG_2PI * Q_ - 0.5 * logdet_val;
  // Main iteration loop - OPTIMIZED
  //#pragma omp parallel if(niter > 20)
  {
    dblvec dqf_local(npars, 0.0);

  //#pragma omp for
    for(int i = 0; i < niter; i++)
    {
      VectorXd IAu = IA * umat.col(i);
      VectorXd IAuD = (IAu.array() * Dvec.array()).matrix();
      // Quadratic form
      double qf = IAuD.dot(IAu);
  //#pragma omp atomic
      logl(i) += -0.5 * qf;

      // Gradient contribution - MUCH faster than outer product
      for(int j = 0; j < npars; j++)
      {
        dqf_local[j] += (S(j) * IAuD).dot(IAu);
      }
    }

    // Accumulate thread-local results
  //#pragma omp critical
    for(int j = 0; j < npars; j++) {
      dqf[j] += dqf_local[j];
    }
  }

  double niter_inv = 1.0 / (double)niter;
  for(int j = 0; j < npars; j++)
  {
    grad(j) += 0.5 * dqf[j] * niter_inv;
  }

  // Hessian computation
  MatrixXd M(npars, npars);
  for(int j = 0; j < npars; j++) {
    for(int k = j; k < npars; k++) {
      double val = (S(j) * S(k)).trace();
      M(j, k) = val;
      if(j != k) M(k, j) = val;
    }
  }
  // Newton-Raphson update
  M = M.llt().solve(MatrixXd::Identity(npars, npars));
  VectorXd theta_curr = Map<VectorXd>(parameters_.data(), parameters_.size());
  theta_curr += M * grad;
  update_parameters(theta_curr.array());


//   
//   static const double LOG_2PI = log(2*M_PI);
//   std::vector<MatrixXd> derivs;
//   derivatives(derivs,1);
//   int npars = derivs.size()-1;
//   int niter = umat.cols();
//   VectorXd grad(npars);
//   grad.setZero();
// 
//   double logdet_val=0.0;
//   logl = 0.0;
//   dblvec dqf(npars);
//   std::fill(dqf.begin(), dqf.end(), 0.0);
//   logl.setZero();
// 
//   // create sparse IA
//   sparse IA(umat.rows(),umat.rows(),true);
// 
//   int n = A.cols();
//   int m = A.rows();
//   double aval;
//   for(int k=0; k<n; k++){
//     int idxlim;
//     for (int i = 0; i < n; i++) {
//       idxlim = i<=m ? i : m;
//       double lsum = 0;
//       for (int j = 0; j < idxlim; j++) {
//         lsum += A(j,i) * IA(grid.NN(j,i),k);
//       }
//       aval = i==k ? (1+lsum) : lsum ;
//       IA.insert(i,k,aval);
//     }
//   }
// 
//   sparse IAt = IA;
//   IAt.transpose();
//   sparse IAD = IAt % Dvec;
// 
//   logdet_val = log_determinant();
//   glmmr::MatrixField<MatrixXd> S;
//   for(int i = 0; i < npars; i++)
//   {
//     MatrixXd detmat = SparseOperators::operator*(IAt, derivs[i+1]);
//     MatrixXd detmat2 = SparseOperators::operator*(detmat, IAD);
//     S.add(detmat2);
//     grad(i) += -0.5* detmat2.trace();
//   }
//   logl.array() += -0.5*(Q_ * LOG_2PI + logdet_val);
//   double qf = 0;
//   for(int i = 0; i < niter; i++)
//   {
//     VectorXd ucol = umat.col(i);
//     VectorXd IAu = IA * ucol;
//     VectorXd IAuD = (IAu.array() * Dvec.array()).matrix();
//     logl(i) += -0.5* IAuD.transpose() * IAu;
//     MatrixXd zzquadquad = IAu * IAu.transpose();
//     zzquadquad *= Dvec.asDiagonal();
//     for(int j = 0; j < npars; j++)
//     {
//       dqf[j] += (zzquadquad * S(j)).trace();
//     }
//   }
//   for(int j = 0; j < npars; j++)
//   {
//     grad(j) += 0.5* dqf[j] / (double) niter;
//   }
// 
//   MatrixXd M(npars,npars);
//   for(int j = 0; j < npars; j++){
//     for(int k = j; k < npars; k++){
//       M(j,k) = (S(j) * S(k)).trace();
//       if(j != k)M(k,j) = M(j,k);
//     }
//   }
//   M = M.llt().solve(MatrixXd::Identity(npars,npars));
//   VectorXd theta_curr = Map<VectorXd>(parameters_.data(), parameters_.size());
//   theta_curr += M * grad;
//   update_parameters(theta_curr.array());
}

inline void glmmr::nngpCovariance::gen_AD_derivatives(glmmr::MatrixField<VectorXd>& dD, glmmr::MatrixField<MatrixXd>& dA)
{
  A.setZero();
  Dvec.setZero();
  int idxlim;
  int npars = dD.size();  
  dblvec s0vals = calc_[0].calculate<CalcDyDx::BetaFirst>(0,0,0,0);
  Dvec(0) = s0vals[0];
  for(int i = 0; i < dD.size(); i++) dD(i)(0) = s0vals[i+1];

#pragma omp parallel for private(idxlim)
  for(int i = 1; i < grid.N; i++)
  {
    idxlim = i <= m ? i : m;
    MatrixXd S(idxlim, idxlim * (npars + 1));
    VectorXd Sv(idxlim * (npars + 1));
    for(int j = 0; j<idxlim; j++)
    {
      S(j,j) = s0vals[0];
      for(int k = 0; k < npars; k++) S(j,j+idxlim*(k+1)) = s0vals[k+1];
    }
    if(idxlim > 1)
    {
      for(int j = 0; j<(idxlim-1); j++)
      {
        for(int k = j+1; k<idxlim; k++)
        {
          dblvec svals = calc_[0].calculate<CalcDyDx::BetaFirst>(grid.NN(j,i),grid.NN(k,i),0,0);
          S(j,k) = svals[0];
          S(k,j) = S(j,k);
          for(int l = 0; l < npars; l++)
          {
            S(j, k+idxlim*(l+1)) = svals[l+1];
            S(k+idxlim*(l+1), j) = S(j, k+idxlim*(l+1));
          } 
        }
      }
    }
    for(int j = 0; j<idxlim; j++)
    {
      dblvec svals = calc_[0].calculate<CalcDyDx::BetaFirst>(i,grid.NN(j,i),0,0);
      Sv(j) = svals[0];
      for(int l = 0; l < npars; l++) Sv(j + idxlim*(l+1)) = svals[l+1];
    }
    A.block(0,i,idxlim,1) = S.block(0,0,idxlim,idxlim).ldlt().solve(Sv);
    for(int l = 0; l < npars; l++) dA(l).block(0,i,idxlim,1) = S.block(0,idxlim*(l+1),idxlim,idxlim).ldlt().solve(Sv.segment(idxlim*(l+1),idxlim) - S.block(0,idxlim*(l+1),idxlim,idxlim) * A.col(i).segment(0,idxlim)); 
    Dvec(i) = s0vals[0] - (A.col(i).segment(0,idxlim).transpose() * Sv.head(idxlim))(0);
    for(int l = 0; l < npars; l++) dD(l)(i) = s0vals[l+1] - (Sv.segment(idxlim*(l+1),idxlim).transpose()*A.col(i).segment(0,idxlim))(0) - (Sv.head(idxlim).transpose() * dA(l).col(i).segment(0,idxlim))(0);
  }
}

inline VectorXd glmmr::nngpCovariance::log_gradient(const MatrixXd& umat, double& ll)
{
 // #pragma omp declare reduction(vec_dbl_plus : std::vector<double> : \
  //                  std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
   //                 initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
  
  int npars = this->npar();
  VectorXd grad(npars);
  grad.setZero();  
  glmmr::MatrixField<VectorXd> dD;
  glmmr::MatrixField<MatrixXd> dA;
  for(int i = 0; i < npars; i++)
  {
    dD.add_new(grid.N);
    dA.add_new(m,grid.N);
    dD(i).setZero();
    dA(i).setZero();
  }
  gen_AD_derivatives(dD,dA);

  //log determinant derivatives
  dblvec dlogdet(npars,0.0);
  double logdet = log_determinant();
  
  for(int i = 0; i < grid.N; i++)
  {
    for(int j = 0; j < npars; j++)
    {
      dlogdet[j] = dD(j)(i)/Dvec(i);
      dD(j)(i) *= (1 / (Dvec(i) * Dvec(i)));
    }
  } 

  double au, av, qf;
  dblvec dau(npars,0.0);
  dblvec dqf(npars,0.0);
  dblvec dll(npars,0.0);
  int niter = umat.cols();
  int idxlim;
  ll = 0;

//#pragma omp parallel for reduction(vec_dbl_plus:dll) reduction(+:ll) private(au, av, qf, dau, dqf, idxlim) if(niter > 50)
  for(int k = 0; k < niter; k++)
  {
    qf = umat(0)*umat(0)/Dvec(0);
    for(int l = 0; l < npars; l++)
    {
      dqf[l] = -1.0 * umat(0,k)*umat(0,k)*dD(l)(0);
    }
    for(int i = 1; i < grid.N; i++)
    {
      idxlim = i <= m ? i : m;
      VectorXd usec(idxlim);
      for(int j = 0; j < idxlim; j++) 
      {
        usec(j) = umat(grid.NN(j,i),k);
      }
      au = umat(i,k) - (A.col(i).segment(0,idxlim).transpose() * usec)(0);
      qf += au*au/Dvec(i);
      for(int l = 0; l < npars; l++)
      {
        dau[l] = -1.0 * (dA(l).col(i).segment(0,idxlim).transpose() * usec)(0);
        dqf[l] += 2*dau[l]*au/Dvec(i) - au*au*dD(l)(i);
      }      
    }
    for(int l = 0; l < npars; l++) dll[l] -= 0.5*dqf[l]; 
    ll -= 0.5*qf;
  }  

  for(int l = 0; l < npars; l++) grad(l) = -0.5 * dlogdet[l] + dll[l] / (double)niter;
  ll *= 1.0/(double)niter;
  ll -= 0.5*logdet + 0.5*grid.N*log(2*M_PI);

  return grad;
}

#pragma once

#include <unordered_set>
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
  double    nu = 0.5;
  bool      sq_exp = false;
  //constructors
  hsgpCovariance(const std::string& formula,const ArrayXXd& data,const strvec& colnames);
  hsgpCovariance(const glmmr::Formula& formula,const ArrayXXd& data,const strvec& colnames);
  hsgpCovariance(const std::string& formula,const ArrayXXd& data,const strvec& colnames,const dblvec& parameters);
  hsgpCovariance(const glmmr::Formula& formula,const ArrayXXd& data,const strvec& colnames,const dblvec& parameters);
  hsgpCovariance(const glmmr::hsgpCovariance& cov);
  // functions
  double      spd_nD(int i);
  dblvec      d_spd_nD(int i);
  dblvec      d_sqrt_spd_nD(int i);
  ArrayXd     phi_nD(int i);
  MatrixXd    ZPhi();
  MatrixXd    ZL() override;
  MatrixXd    ZL_deriv(int par);
  MatrixXd    D(bool chol = true, bool upper = false) override;
  MatrixXd    ZLu(const MatrixXd& u) override;
  MatrixXd    Lu(const MatrixXd& u) override;
  VectorXd    sim_re() override;
  MatrixXd    solve(const MatrixXd& u) override;
  void        nr_step(const MatrixXd &umat, const MatrixXd &vmat, 
               ArrayXd& logl, ArrayXd& gradients, 
               const ArrayXd& uweight) override;
  SparseMatrix<double>   ZL_sparse_new() override;
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
  MatrixXd    solve_spectral(const MatrixXd& u);
protected:
//data
  int       total_m;
  MatrixXd  L; // Half-eigen decomposition of Lambda + PhiTPhi m^2 * m^2
  ArrayXd   Lambda;
  ArrayXXi  indices;
  MatrixXd  Phi;
  MatrixXd  PhiT;
  
  //functions
  void      parse_hsgp_data();
  void      gen_indices();
  void      gen_phi_prod();
  void      update_lambda();
};

}


inline glmmr::hsgpCovariance::hsgpCovariance(const std::string& formula,
               const ArrayXXd& data,
               const strvec& colnames) : Covariance(formula, data, colnames),
               dim(this->re_cols_data_[0][0].size()),
               m(dim),
               hsgp_data(this->matZ.cols(),dim),
               L_boundary(dim),
               L(this->matZ.cols(),1), 
               Lambda(1), 
               indices(1,dim), 
               Phi(this->matZ.cols(),1), 
               PhiT(2,2) {
  isSparse = false;
  for(int i = 0; i < dim; i++)L_boundary(i) = 1.5;
  std::fill(m.begin(),m.end(),10);
  parse_hsgp_data();
  update_approx_parameters();
};

inline glmmr::hsgpCovariance::hsgpCovariance(const glmmr::Formula& formula,
               const ArrayXXd& data,
               const strvec& colnames) : Covariance(formula, data, colnames),
               dim(this->re_cols_data_[0][0].size()),
               m(dim),
               hsgp_data(this->matZ.cols(),dim),
               L_boundary(dim),
               L(this->matZ.cols(),1), 
               Lambda(1), 
               indices(1,dim), 
               Phi(this->matZ.cols(),1), 
               PhiT(2,2) {
  isSparse = false;
  for(int i = 0; i < dim; i++)L_boundary(i) = 1.5;
  std::fill(m.begin(),m.end(),10);
  parse_hsgp_data();
  update_approx_parameters();
};

inline glmmr::hsgpCovariance::hsgpCovariance(const std::string& formula,
               const ArrayXXd& data,
               const strvec& colnames,
               const dblvec& parameters) : Covariance(formula, data, colnames, parameters),
               dim(this->re_cols_data_[0][0].size()),
               m(dim),
               hsgp_data(this->matZ.cols(),dim),
               L_boundary(dim),
               L(this->matZ.cols(),1), 
               Lambda(1),
               indices(1,dim), 
               Phi(this->matZ.cols(),1), 
               PhiT(2,2) {
  isSparse = false;
  for(int i = 0; i < dim; i++)L_boundary(i) = 1.5;
  std::fill(m.begin(),m.end(),10);
  parse_hsgp_data();
  update_approx_parameters();
  update_lambda();
};

inline glmmr::hsgpCovariance::hsgpCovariance(const glmmr::Formula& formula,
               const ArrayXXd& data,
               const strvec& colnames,
               const dblvec& parameters) : Covariance(formula, data, colnames, parameters),
               dim(this->re_cols_data_[0][0].size()),
               m(dim),
               hsgp_data(this->matZ.cols(),dim),
               L_boundary(dim),
               L(this->matZ.cols(),1), 
               Lambda(1),
               indices(1,dim), 
               Phi(this->matZ.cols(),1), 
               PhiT(2,2) {
  isSparse = false;
  for(int i = 0; i < dim; i++)L_boundary(i) = 1.5;
  std::fill(m.begin(),m.end(),10);
  parse_hsgp_data();
  update_approx_parameters();
  update_lambda();
};

inline glmmr::hsgpCovariance::hsgpCovariance(const glmmr::hsgpCovariance& cov) : Covariance(cov.form_, cov.data_, cov.colnames_, cov.parameters_), 
dim(cov.dim),m(cov.m), hsgp_data(cov.hsgp_data),
L_boundary(cov.L_boundary), L(cov.L), Lambda(cov.Lambda), 
indices(cov.indices), Phi(cov.Phi), PhiT(cov.PhiT) {
  isSparse = false;
};

inline void glmmr::hsgpCovariance::parse_hsgp_data(){
  std::unordered_set<int> uz;
  hsgp_data.setZero();
  for (int k=0; k<matD.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(this->matZ,k); it; ++it)
    {
      if(uz.find(it.col()) == uz.end()){
        for(int i = 0; i < dim; i++){
          hsgp_data(it.col(),i) = this->data_(it.row(),this->re_cols_data_[0][0][i]);
        }
        uz.insert(it.col());
      }
    }
    
    sq_exp = false;
  nu = 0.5; // default
  for(const auto& fn : this->fn_[0]){
    if(fn == CovFunc::sqexp || fn == CovFunc::sqexp0){
      sq_exp = true;
      return;
    } else if(fn == CovFunc::matern || fn == CovFunc::matern1log){
      nu = 1.0;
      return;
    } else if(fn == CovFunc::matern2log){
      nu = 2.0;
      return;
    } else if(fn == CovFunc::fexp || fn == CovFunc::fexp0 || fn == CovFunc::fexplog){
      nu = 0.5;
      return;
    }
  }
  throw std::runtime_error("HSGP only allows exp, matern (nu=1,2), and sqexp currently.");
}

inline void glmmr::hsgpCovariance::update_approx_parameters(intvec m_, ArrayXd L_){
  m = m_;
  L_boundary = L_;
  total_m = glmmr::algo::prod_vec(m);
  this->Q_ = total_m;
  indices.resize(total_m,NoChange);
  Phi.resize(NoChange,total_m);
  PhiT.resize(total_m,total_m);
  Lambda.resize(total_m);
  L.resize(NoChange,total_m);
  gen_indices();
  gen_phi_prod();
}

inline void glmmr::hsgpCovariance::update_approx_parameters(){
  total_m = glmmr::algo::prod_vec(m);
  this->Q_ = total_m;
  indices.resize(total_m,NoChange);
  Phi.resize(NoChange,total_m);
  PhiT.resize(total_m,total_m);
  Lambda.resize(total_m);
  L.resize(NoChange,total_m);
  gen_indices();
  gen_phi_prod();
}

inline MatrixXd glmmr::hsgpCovariance::solve(const MatrixXd& u){
  return u.array().colwise() / Lambda;
}

inline double glmmr::hsgpCovariance::spd_nD(int i){
  double wprod = 0;
  for(int d = 0; d < dim; d++){
    double w = (indices(i,d) * M_PI) / (2 * L_boundary(d));
    wprod += w * w;
  }
  
  bool logpars = all_log_re();
  double sigma2 = logpars ? exp(parameters_[0]) : parameters_[0];
  double ell    = logpars ? exp(parameters_[1]) : parameters_[1];
  
  if(sq_exp){
    double phisq = ell * ell;
    return sigma2 * pow(2 * M_PI, dim / 2.0) * pow(ell, dim) * exp(-0.5 * phisq * wprod);
  }
  
  double ell2 = ell * ell;
  double two_nu = 2.0 * nu;
  double alpha = nu + dim / 2.0;
  double A = pow(4.0 * M_PI, dim / 2.0) * tgamma(alpha) / tgamma(nu);
  double B = pow(two_nu, nu) / pow(ell, two_nu);
  double C = pow(two_nu / ell2 + wprod, -alpha);
  
  return sigma2 * A * B * C;
}

inline dblvec glmmr::hsgpCovariance::d_spd_nD(int i){
  // Always returns ∂S/∂log(σ²) and ∂S/∂log(ℓ)
  dblvec result(2);
  
  double wprod = 0;
  for(int d = 0; d < dim; d++){
    double w = (indices(i,d) * M_PI) / (2 * L_boundary(d));
    wprod += w * w;
  }
  
  bool logpars = all_log_re();
  double sigma2 = logpars ? exp(parameters_[0]) : parameters_[0];
  double ell    = logpars ? exp(parameters_[1]) : parameters_[1];
  double S = spd_nD(i);
  
  if(sq_exp){
    double phisq = ell * ell;
    result[0] = S;
    result[1] = S * (dim - phisq * wprod);
    return result;
  }
  
  double ell2 = ell * ell;
  double two_nu = 2.0 * nu;
  double E = two_nu / ell2 + wprod;
  double alpha = nu + dim / 2.0;
  
  result[0] = S;
  result[1] = S * (-two_nu + two_nu * 2.0 * alpha / (ell2 * E));
  
  return result;
}

inline dblvec glmmr::hsgpCovariance::d_sqrt_spd_nD(int i){
  dblvec dS = d_spd_nD(i);
  double S = spd_nD(i);
  double inv_2sqrtS = 0.5 / sqrt(S);
  dblvec result(2);
  result[0] = dS[0] * inv_2sqrtS;
  result[1] = dS[1] * inv_2sqrtS;
  return result;
}

inline MatrixXd glmmr::hsgpCovariance::ZPhi(){
  return this->Z() * Phi;
}

inline ArrayXd glmmr::hsgpCovariance::phi_nD(int i){
  ArrayXd fi1(hsgp_data.rows());
  ArrayXd fi2(hsgp_data.rows());
  
  fi1 = (1/sqrt(L_boundary(0))) * sin(indices(i,0)*M_PI*(hsgp_data.col(0)+L_boundary(0))/(2*L_boundary(0)));
  if(dim > 1){
    for(int d = 1; d < dim; d++){
      fi2 = (1/sqrt(L_boundary(d))) * sin(indices(i,d)*M_PI*(hsgp_data.col(d)+L_boundary(d))/(2*L_boundary(d)));
      fi1 *= fi2;
    }
  }
  return fi1;
}

inline MatrixXd glmmr::hsgpCovariance::D(bool chol, bool upper){
  if(chol){
    return Lambda.sqrt().matrix().asDiagonal();
  } else {
    return Lambda.matrix().asDiagonal();
  }
}

inline VectorXd glmmr::hsgpCovariance::sim_re(){
  if(parameters_.size()==0) throw std::runtime_error("no parameters");
  std::random_device rd{};
  std::mt19937 gen{ rd() };
  std::normal_distribution d{ 0.0, 1.0 };
  VectorXd samps(this->Q_);
  for(int j = 0; j < samps.size(); j++){
    samps(j) = d(gen) * sqrt(Lambda(j));  // u ~ N(0, diag(Lambda))
  }
  return samps;
}

inline void glmmr::hsgpCovariance::update_parameters_extern(const dblvec& parameters){
  parameters_ = parameters;
  update_lambda();
};

inline void glmmr::hsgpCovariance::update_parameters(const dblvec& parameters){
  parameters_ = parameters;
  update_lambda();
};

inline void glmmr::hsgpCovariance::update_parameters(const ArrayXd& parameters){
  if(parameters_.size()==0){
    for(int i = 0; i < parameters.size(); i++){
      parameters_.push_back(parameters(i));
    }
  } else {
    for(int i = 0; i < parameters.size(); i++){
      parameters_[i] = parameters(i);
    }
  }
  update_lambda();
};

inline MatrixXd glmmr::hsgpCovariance::ZL(){
  return ZPhi();
}

inline MatrixXd glmmr::hsgpCovariance::ZL_deriv(int par)
{
  throw std::runtime_error("ZL deriv (HSGP) has been disabled");
}

inline MatrixXd glmmr::hsgpCovariance::ZLu(const MatrixXd& u){
  return ZPhi() * u;
}


inline MatrixXd glmmr::hsgpCovariance::Lu(const MatrixXd& u){
  return Phi * u;
}

inline SparseMatrix<double> glmmr::hsgpCovariance::ZL_sparse_new(){
  return ZPhi().sparseView();
}

inline int glmmr::hsgpCovariance::Q() const{
  return this->Q_;
}

inline double glmmr::hsgpCovariance::log_likelihood(const VectorXd &u){
  static const double LOG_2PI = log(2*M_PI);
  double logdet = Lambda.log().sum();
  double qf = (u.array().square() / Lambda).sum();
  return -0.5 * total_m * LOG_2PI - 0.5 * logdet - 0.5 * qf;
}

inline double glmmr::hsgpCovariance::log_determinant(){
  return Lambda.log().sum();
}

inline void glmmr::hsgpCovariance::set_function(bool squared_exp){
  sq_exp = squared_exp;
}

inline void glmmr::hsgpCovariance::gen_indices(){
  intvec2d indices_vec;
  intvec ind_buffer(dim);
  intvec2d linspace_vec;
  for(int i = 0; i < dim; i++){
    intvec linspaced(m[i]);
    for(int k = 1; k <= m[i]; k++)linspaced[k-1] = k;
    linspace_vec.push_back(linspaced);
  }
  for(int i = 0; i < linspace_vec[0].size(); i++){
    glmmr::algo::combinations(linspace_vec,0,i,ind_buffer,indices_vec);
  }
  // copy into array
  for(int i = 0; i < indices_vec.size(); i++){
    for(int j = 0; j < indices_vec[0].size(); j++){
      indices(i,j) = indices_vec[i][j];
    }
  }
}

inline void glmmr::hsgpCovariance::gen_phi_prod(){
  for(int i = 0; i < total_m; i++){
    ArrayXd phi = phi_nD(i);
    Phi.col(i) = phi.matrix();
  }
  PhiT = Phi.transpose() * Phi;
}

inline MatrixXd glmmr::hsgpCovariance::PhiSPD(bool lambda, bool inverse){
  MatrixXd pnew = Phi;
  if(lambda){
    if(!inverse){
      pnew *= Lambda.sqrt().matrix().asDiagonal();
    } else {
      pnew *= Lambda.sqrt().inverse().matrix().asDiagonal();
    }
  }
  return pnew;
}

inline ArrayXd glmmr::hsgpCovariance::LambdaSPD(){
  return Lambda;
}

inline void glmmr::hsgpCovariance::update_lambda(){
  for(int i = 0; i < total_m; i++){
    Lambda(i) = spd_nD(i);
  }
}

inline void glmmr::hsgpCovariance::nr_step(
    const MatrixXd &umat, const MatrixXd &vmat,
    ArrayXd& logl, ArrayXd& gradients, const ArrayXd& uweight)
{
  // This function is NOT called directly — the bits_hsgp nr_theta specialisation
  // handles both the prior and observation-space contributions.
  // Redirect there via the existing machinery.
  throw std::runtime_error("HSGP nr_step should not be called directly — use the nr_theta specialisation");

}

inline MatrixXd glmmr::hsgpCovariance::solve_spectral(const MatrixXd& u){
  return u;
}
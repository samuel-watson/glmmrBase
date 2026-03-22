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
  ArrayXd   raw_half_range_;   // per-dimension half-range (always stored)
  ArrayXd   shift_;            // per-dimension center
  ArrayXd   scale_factors_;    // what's currently applied to hsgp_data
  double    L_factor_ = 1.5;
  
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
  dblvec      get_parameters_extern() const;
  void        set_function(bool squared_exp);
  MatrixXd    PhiSPD(bool lambda = true, bool inverse = false);
  ArrayXd     LambdaSPD();
  void        update_approx_parameters(intvec m_, double L_factor);
  void        update_approx_parameters();
  void        set_anisotropic(bool aniso);
  int         npar() const override;
  intvec      parameter_fn_index() const override;
protected:
//data
  int       total_m;
  MatrixXd  L; // Half-eigen decomposition of Lambda + PhiTPhi m^2 * m^2
  ArrayXd   Lambda;
  ArrayXXi  indices;
  MatrixXd  Phi;
  MatrixXd  PhiT;
  bool      anisotropic = false;
  
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
  for(int i = 0; i < dim; i++)L_boundary(i) = 1.05;
  std::fill(m.begin(),m.end(),10);
  raw_half_range_ = ArrayXd::Ones(dim);
  shift_ = ArrayXd::Zero(dim);
  scale_factors_ = ArrayXd::Ones(dim);
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
  for(int i = 0; i < dim; i++)L_boundary(i) = 1.05;
  std::fill(m.begin(),m.end(),10);
  raw_half_range_ = ArrayXd::Ones(dim);
  shift_ = ArrayXd::Zero(dim);
  scale_factors_ = ArrayXd::Ones(dim);
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
  for(int i = 0; i < dim; i++)L_boundary(i) = 1.05;
  std::fill(m.begin(),m.end(),10);
  raw_half_range_ = ArrayXd::Ones(dim);
  shift_ = ArrayXd::Zero(dim);
  scale_factors_ = ArrayXd::Ones(dim);
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
  for(int i = 0; i < dim; i++)L_boundary(i) = 1.05;
  std::fill(m.begin(),m.end(),10);
  parse_hsgp_data();
  raw_half_range_ = ArrayXd::Ones(dim);
  shift_ = ArrayXd::Zero(dim);
  scale_factors_ = ArrayXd::Ones(dim);
  update_approx_parameters();
  update_lambda();
};

inline glmmr::hsgpCovariance::hsgpCovariance(const glmmr::hsgpCovariance& cov) 
  : Covariance(cov.form_, cov.data_, cov.colnames_, cov.parameters_), 
    dim(cov.dim), m(cov.m), hsgp_data(cov.hsgp_data),
    L_boundary(cov.L_boundary), 
    anisotropic(cov.anisotropic),
    raw_half_range_(cov.raw_half_range_), 
    shift_(cov.shift_),
    scale_factors_(cov.scale_factors_),
    L(cov.L), Lambda(cov.Lambda), 
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
    
    shift_.resize(dim);
    raw_half_range_.resize(dim);
    scale_factors_.resize(dim);
    
    for(int d = 0; d < dim; d++){
      double mn = hsgp_data.col(d).minCoeff();
      double mx = hsgp_data.col(d).maxCoeff();
      shift_(d) = 0.5 * (mn + mx);
      raw_half_range_(d) = 0.5 * (mx - mn);
      if(raw_half_range_(d) < 1e-10) raw_half_range_(d) = 1.0;
    }
    
    double max_range = raw_half_range_.maxCoeff();
    scale_factors_.setConstant(max_range);
    
    for(int d = 0; d < dim; d++){
      hsgp_data.col(d) = (hsgp_data.col(d) - shift_(d)) / scale_factors_(d);
    }
    
    for(int d = 0; d < dim; d++){
      double max_abs = hsgp_data.col(d).abs().maxCoeff();
      L_boundary(d) = L_factor_ * std::max(max_abs, 0.1);
    }
    
    sq_exp = false;
    nu = 0.5; // default
    for(const auto& fn : this->fn_[0]){
      if(fn == CovFunc::sqexp || fn == CovFunc::sqexp0){
        sq_exp = true;
        return;
      } else if(fn == CovFunc::matern || fn == CovFunc::matern1log){
        nu = 1.5;
        return;
      } else if(fn == CovFunc::matern2log){
        nu = 2.5;
        return;
      } else if(fn == CovFunc::fexp || fn == CovFunc::fexp0 || fn == CovFunc::fexplog){
        nu = 0.5;
        return;
      }
  }
  throw std::runtime_error("HSGP only allows exp, matern (nu=1,2), and sqexp currently.");
}

inline void glmmr::hsgpCovariance::update_approx_parameters(intvec m_, double L_factor){
  m = m_;
  L_factor_ = L_factor;
  // Actual boundary = factor * max(|scaled data|) per dimension
  for(int d = 0; d < dim; d++){
    double max_abs = hsgp_data.col(d).abs().maxCoeff();
    L_boundary(d) = L_factor_ * std::max(max_abs, 0.1);
  }
  total_m = glmmr::algo::prod_vec(m);
  this->Q_ = total_m;
  indices.resize(total_m, NoChange);
  Phi.resize(NoChange, total_m);
  PhiT.resize(total_m, total_m);
  Lambda.resize(total_m);
  L.resize(NoChange, total_m);
  gen_indices();
  gen_phi_prod();
}

// Keep the explicit version for internal use, but have it 
// also recompute from scaled data
inline void glmmr::hsgpCovariance::update_approx_parameters(){
  for(int d = 0; d < dim; d++){
    double max_abs = hsgp_data.col(d).abs().maxCoeff();
    L_boundary(d) = L_factor_ * std::max(max_abs, 0.1);
  }
  total_m = glmmr::algo::prod_vec(m);
  this->Q_ = total_m;
  indices.resize(total_m, NoChange);
  Phi.resize(NoChange, total_m);
  PhiT.resize(total_m, total_m);
  Lambda.resize(total_m);
  L.resize(NoChange, total_m);
  gen_indices();
  gen_phi_prod();
}

inline MatrixXd glmmr::hsgpCovariance::solve(const MatrixXd& u){
  return u.array().colwise() / Lambda;
}

inline int glmmr::hsgpCovariance::npar() const {
  int np = 2;
  if(anisotropic) np += dim - 1;
  return np;
}

inline intvec glmmr::hsgpCovariance::parameter_fn_index() const {
  intvec idx = re_fn_par_link_;
  if(!anisotropic) return idx;
  int max = *std::max_element(idx.begin(), idx.end());
  for(int i = 0; i < (dim - 1); i++) idx.push_back(max + i + 1);
  return idx;
}

inline void glmmr::hsgpCovariance::set_anisotropic(bool aniso){
  if(aniso == anisotropic) return;
  anisotropic = aniso;
  
  // Undo current scaling, apply new
  for(int d = 0; d < dim; d++){
    hsgp_data.col(d) *= scale_factors_(d);  // back to centred original
  }
  
  for(int d = 0; d < dim; d++){
    double max_abs = hsgp_data.col(d).abs().maxCoeff();
    L_boundary(d) = L_factor_ * std::max(max_abs, 0.1);
  }
  
  // Recompute basis functions with rescaled data
  gen_phi_prod();
  if(parameters_.size() > 1) update_lambda();
  
  if(aniso){
    scale_factors_ = raw_half_range_;
    // Resize parameters: sigma^2 + dim length scales
    dblvec newpars(1 + dim, 0.0);
    if(!parameters_.empty()) newpars[0] = parameters_[0];
    // Initialise each ell_d from existing ell (if present), 
    // converting from old to new scale
    if(parameters_.size() > 1){
      bool logpars = all_log_re();
      double old_scale = scale_factors_.maxCoeff(); // previous isotropic scale
      // Actually old scale is still in scale_factors_ before we overwrote...
      // We already set scale_factors_ = raw_half_range_ above, so reconstruct:
      double iso_scale = raw_half_range_.maxCoeff();
      for(int d = 0; d < dim; d++){
        if(logpars){
          // old internal: log(ell_orig) - log(iso_scale)
          // new internal: log(ell_orig) - log(raw_half_range_(d))
          newpars[1 + d] = parameters_[1] - log(raw_half_range_(d)) + log(iso_scale);
        } else {
          newpars[1 + d] = parameters_[1] * iso_scale / raw_half_range_(d);
        }
      }
    }
    parameters_ = newpars;
  } else {
    double max_range = raw_half_range_.maxCoeff();
    double old_scale_0 = scale_factors_(0); // any dim, since about to go isotropic
    scale_factors_.setConstant(max_range);
    // Collapse to sigma^2 + single ell (use geometric mean of scaled ells)
    dblvec newpars(2, 0.0);
    if(!parameters_.empty()) newpars[0] = parameters_[0];
    if(parameters_.size() > 1){
      bool logpars = all_log_re();
      double avg = 0;
      for(int d = 0; d < dim; d++){
        if(logpars){
          avg += parameters_[1 + d] + log(raw_half_range_(d)) - log(max_range);
        } else {
          avg += log(parameters_[1 + d] * raw_half_range_(d) / max_range);
        }
      }
      avg /= dim;
      newpars[1] = logpars ? avg : exp(avg);
    }
    parameters_ = newpars;
  }
  
  for(int d = 0; d < dim; d++){
    hsgp_data.col(d) /= scale_factors_(d);  // apply new scaling
  }
  
  // Recompute basis functions with rescaled data
  gen_phi_prod();
  if(parameters_.size() > 1) update_lambda();
}

// In hsgpcovariance.hpp — replace spd_nD

inline double glmmr::hsgpCovariance::spd_nD(int i){
  bool logpars = all_log_re();
  double sigma2 = logpars ? exp(parameters_[0]) : parameters_[0];
  
  // Anisotropic: one length scale per dimension
  // parameters_[1..dim] are the per-dimension length scales
  
  if(sq_exp){
    double result = sigma2 * pow(2 * M_PI, dim / 2.0);
    for(int d = 0; d < dim; d++){
      double ell_d = anisotropic 
      ? (logpars ? exp(parameters_[1 + d]) : parameters_[1 + d])
        : (logpars ? exp(parameters_[1])     : parameters_[1]);
      double w = (indices(i,d) * M_PI) / (2 * L_boundary(d));
      result *= ell_d * exp(-0.5 * ell_d * ell_d * w * w);
    }
    return result;
  }
  
  // Matérn / exponential: product of 1D spectral densities
  // S_1d(w; nu, ell) = C(nu) * (2*nu/ell^2 + w^2)^{-(nu + 0.5)}
  // For separable: S(w) = sigma^2 * prod_d S_1d(w_d; nu, ell_d)
  double result = sigma2;
  double alpha_1d = nu + 0.5;
  double C_1d = pow(4.0 * M_PI, 0.5) * tgamma(alpha_1d) / tgamma(nu);
  
  for(int d = 0; d < dim; d++){
    double ell_d = anisotropic
    ? (logpars ? exp(parameters_[1 + d]) : parameters_[1 + d])
      : (logpars ? exp(parameters_[1])     : parameters_[1]);
    double w = (indices(i,d) * M_PI) / (2 * L_boundary(d));
    double two_nu = 2.0 * nu;
    double ell2 = ell_d * ell_d;
    double B = pow(two_nu, nu) / pow(ell_d, two_nu);
    double C = pow(two_nu / ell2 + w * w, -alpha_1d);
    result *= C_1d * B * C;
  }
  return result;
}

inline dblvec glmmr::hsgpCovariance::d_spd_nD(int i){
  // Returns: d/dlog(sigma^2), d/dlog(ell_1), ..., d/dlog(ell_dim)
  bool logpars = all_log_re();
  double sigma2 = logpars ? exp(parameters_[0]) : parameters_[0];
  bool anisotropic = (parameters_.size() >= 1 + dim);
  int npars = anisotropic ? 1 + dim : 2;
  dblvec result(npars, 0.0);
  
  double S = spd_nD(i);
  result[0] = S;  // dS/dlog(sigma^2) = S always
  
  if(sq_exp){
    for(int d = 0; d < dim; d++){
      int par_idx = anisotropic ? 1 + d : 1;
      double ell_d = logpars ? exp(parameters_[par_idx]) : parameters_[par_idx];
      double w = (indices(i,d) * M_PI) / (2 * L_boundary(d));
      // dS/dlog(ell_d) = S * (1 - ell_d^2 * w^2)
      // but if isotropic, accumulate over all dims into result[1]
      double contrib = S * (1.0 - ell_d * ell_d * w * w);
      if(anisotropic){
        // Each dimension contributes independently via product rule
        // d/dlog(ell_d) of prod = S/S_d * dS_d/dlog(ell_d)
        // Simpler: dlog(S)/dlog(ell_d) = 1 - ell_d^2 w_d^2
        result[1 + d] = S * (1.0 - ell_d * ell_d * w * w);
      } else {
        result[1] += contrib;  // isotropic: chain rule sum
      }
    }
    return result;
  }
  
  // Matérn separable: product rule
  // d log(S) / d log(ell_d) = d log(S_d) / d log(ell_d)
  //   = -2*nu + 2*nu * (nu+0.5) * 2 / (ell_d^2 * (2*nu/ell_d^2 + w_d^2)) ... 
  // but cleaner: for the 1D factor with params (nu, ell_d):
  //   dlog(S_1d)/dlog(ell_d) = -2*nu + 2*(nu+0.5)*2*nu / (2*nu + ell_d^2 * w_d^2)
  for(int d = 0; d < dim; d++){
    int par_idx = anisotropic ? 1 + d : 1;
    double ell_d = logpars ? exp(parameters_[par_idx]) : parameters_[par_idx];
    double w = (indices(i,d) * M_PI) / (2 * L_boundary(d));
    double ell2 = ell_d * ell_d;
    double two_nu = 2.0 * nu;
    double alpha_1d = nu + 0.5;
    double E = two_nu / ell2 + w * w;
    double dlogS_d = -two_nu + two_nu * 2.0 * alpha_1d / (ell2 * E);
    
    if(anisotropic){
      result[1 + d] = S * dlogS_d;
    } else {
      result[1] += S * dlogS_d;
    }
  }
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
  if(static_cast<int>(parameters.size())!=npar())throw std::runtime_error(std::to_string(parameters.size())+" covariance parameters provided, "+std::to_string(npar())+" required");
  bool logpars = all_log_re();
  parameters_ = parameters;
  
  if(anisotropic){
    for(int d = 0; d < dim; d++){
      int idx = 1 + d;
      if(idx < (int)parameters_.size()){
        if(logpars) parameters_[idx] -= log(scale_factors_(d));
        else        parameters_[idx] /= scale_factors_(d);
      }
    }
  } else if(parameters_.size() > 1){
    if(logpars) parameters_[1] -= log(scale_factors_(0));
    else        parameters_[1] /= scale_factors_(0);
  }
  
  update_lambda();
};

inline dblvec glmmr::hsgpCovariance::get_parameters_extern() const {
  bool logpars = all_log_re();
  dblvec pars = parameters_;
  
  if(anisotropic){
    for(int d = 0; d < dim; d++){
      int idx = 1 + d;
      if(idx < (int)pars.size()){
        if(logpars) pars[idx] += log(scale_factors_(d));
        else        pars[idx] *= scale_factors_(d);
      }
    }
  } else if(pars.size() > 1){
    if(logpars) pars[1] += log(scale_factors_(0));
    else        pars[1] *= scale_factors_(0);
  }
  return pars;
}

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


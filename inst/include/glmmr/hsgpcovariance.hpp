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
  //constructors
  hsgpCovariance(const std::string& formula,const ArrayXXd& data,const strvec& colnames);
  hsgpCovariance(const glmmr::Formula& formula,const ArrayXXd& data,const strvec& colnames);
  hsgpCovariance(const std::string& formula,const ArrayXXd& data,const strvec& colnames,const dblvec& parameters);
  hsgpCovariance(const glmmr::Formula& formula,const ArrayXXd& data,const strvec& colnames,const dblvec& parameters);
  hsgpCovariance(const glmmr::hsgpCovariance& cov);
  // functions
  double      spd_nD(int i);
  dblvec      d_spd_nD(int i);
  ArrayXd     phi_nD(int i);
  MatrixXd    ZL() override;
  MatrixXd    ZL_deriv(int par);
  MatrixXd    D(bool chol = true, bool upper = false) override;
  MatrixXd    ZLu(const MatrixXd& u) override;
  MatrixXd    Lu(const MatrixXd& u) override;
  VectorXd    sim_re() override;
  SparseMatrix<double>   ZL_sparse() override;
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
  // 
  // 
  // for(int j = 0; j < this->matZ.Ai.size(); j++){
  //   if(uz.find(this->matZ.Ai[j]) == uz.end()){
  //     for(int i = 0; i < dim; i++){
  //       hsgp_data(this->matZ.Ai[j],i) = this->data_(j,this->re_cols_data_[0][0][i]);
  //     }
  //     uz.insert(this->matZ.Ai[j]);
  //   } 
  // }
  
    
  auto sqexpidx = std::find(this->fn_[0].begin(),this->fn_[0].end(),CovFunc::sqexp);
  if(!(sqexpidx == this->fn_[0].end())){
    sq_exp = true;
  } else {
    auto expidx = std::find(this->fn_[0].begin(),this->fn_[0].end(),CovFunc::fexp);
    if(!(expidx == this->fn_[0].end())){
      sq_exp = false;
    } else {
      throw std::runtime_error("HSGP only allows exp and sqexp currently.");
    }
  }
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

inline double glmmr::hsgpCovariance::spd_nD(int i){
  double wprod = 0;
  const static double M_PI4 = 4 * M_PI;
  const static double M_PI2 = 2 * M_PI;
  const static double TGAM2 = tgamma(0.5);
  for(int d = 0; d < dim; d++) {
    double w = (indices(i,d)*M_PI)/(2*L_boundary(d));
    wprod += w*w;
  }
  double S;
  double phisq = parameters_[1] * parameters_[1];
  if(sq_exp){
    S = parameters_[0] * pow(M_PI2, dim/2.0) * pow(parameters_[1],dim) * exp(-0.5 * phisq * wprod);
  } else {
    double S1 = parameters_[0] * pow(M_PI4, dim/2.0) * (tgamma(0.5*(dim+1))/(TGAM2*parameters_[1]));
    double S2 = 1/phisq + wprod;
    S = S1 * pow(S2,-1*(dim+1)/2.0);
  }
  return S;
}

inline dblvec glmmr::hsgpCovariance::d_spd_nD(int i)
{
  dblvec result(2);
  if(sq_exp){
    throw std::runtime_error("HSGP derivatives only available for exponential covariance");
  } else {
    double wprod = 0;
    const static double M_PI4 = 4 * M_PI;
    const static double M_PI2 = 2 * M_PI;
    const static double TGAM2 = tgamma(0.5);
    
    double c_val = -1*(dim+1)/4.0; // c/2
    double a = pow(M_PI4, dim/2.0) * (tgamma(0.5*(dim+1))/TGAM2);
    
    for(int d = 0; d < dim; d++) {
      double w = (indices(i,d)*M_PI)/(2*L_boundary(d));
      wprod += w*w;
    }
    
    // this is on the log scale
    double log_par0 = log(parameters_[0]);
    double log_par1 = log(parameters_[1]);
    double exp1_w = (exp(-2.0 * log_par1) + wprod);
    double exp1_pow = pow(exp1_w,c_val);
    double b = a * exp(-0.5*log_par1) * exp1_pow;
    result[0] = 0.5* exp(0.5 * log_par0)* b;
    //result[2] = 0.5* exp(0.5 * log_par0)* b;
    double d_inter = exp1_pow * ((-2.0 * c_val * exp(-2.5 * log_par1))/exp1_w - 0.5*exp(-0.5*log_par1));
    result[1] = exp(0.5 * log_par0) * a * d_inter;
    //result[4] = 0.5 * exp(0.5 * log_par0) * a * d_inter;
    // double d2_inter = (6.0 * c_val + 0.5) * wprod * exp(4.5 * log_par1);
    // double d3_inter = 4 * (c_val + 0.25) * (c_val + 0.25) * exp(2.5 * log_par1);
    // double d4_inter = 0.25 * wprod * wprod * exp(6.5 * log_par1);
    // double d5_inter = wprod * exp(2.0 * log_par1 + 1);
    // double d_inter_sum = d2_inter + d3_inter + d4_inter;
    // result[3] = exp(0.5 * log_par0) * a * (exp(-3.0 * log_par1) * exp1_pow * d_inter_sum)/(d5_inter * d5_inter);
    // 
  }
  return result;
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
  MatrixXd As = PhiSPD();
 
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

inline VectorXd glmmr::hsgpCovariance::sim_re(){
  if(parameters_.size()==0)throw std::runtime_error("no parameters");
  VectorXd samps(this->Q_);
  std::random_device rd{};
  std::mt19937 gen{ rd() };
  std::normal_distribution d{ 0.0, 1.0 };
  auto random_norm = [&d, &gen] { return d(gen); };
  VectorXd zz(this->Q_);      
  for (int j = 0; j < zz.size(); j++) zz(j) = random_norm();
  samps = glmmr::hsgpCovariance::ZL()*zz;
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
  MatrixXd Z = this->Z();
  MatrixXd P = PhiSPD();
  MatrixXd ZL = Z * P;
  return ZL;
}

inline MatrixXd glmmr::hsgpCovariance::ZL_deriv(int par)
{
  throw std::runtime_error("ZL deriv (HSGP) has been disabled");
  // ArrayXd Lambda_deriv(Lambda.size());
  // #pragma omp parallel for
  //   for(int i = 0; i < total_m; i++)
  //   {
  //     Lambda_deriv(i) = d_spd_nD(i,par);
  //   }
    MatrixXd pnew = Phi;
    //pnew *= Lambda_deriv.matrix().asDiagonal();
    return pnew; 
}

inline MatrixXd glmmr::hsgpCovariance::ZLu(const MatrixXd& u){
  MatrixXd ZL = glmmr::hsgpCovariance::ZL();
  return ZL * u;
}

inline MatrixXd glmmr::hsgpCovariance::Lu(const MatrixXd& u){
  MatrixXd ZLu = L * u;
  return ZLu;
}

inline SparseMatrix<double> glmmr::hsgpCovariance::ZL_sparse(){
  MatrixXd ZLmat = this->ZL();
  return ZLmat.sparseView();
}

inline int glmmr::hsgpCovariance::Q() const{
  return this->Q_;
}

inline double glmmr::hsgpCovariance::log_likelihood(const VectorXd &u){
  if(u.size() != L.rows())throw std::runtime_error("hsgp problem u dim wrong");
  static const double LOG_2PI = log(2*M_PI);
  double ll = 0;
  double logdet = log_determinant();
  VectorXd uquad = u * L;
  ll += -0.5*hsgp_data.rows() * LOG_2PI - 0.5*uquad.transpose()*uquad - 0.5*logdet;
  return ll;
}

inline double glmmr::hsgpCovariance::log_determinant(){
  double logdet = 0;
  for(int i = 0; i < indices.rows(); i++){
    logdet += log(Lambda(i));
  }
  return logdet;
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
  L = PhiSPD(true,true);
}

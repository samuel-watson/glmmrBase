#pragma once

#include <boost/math/distributions/fisher_f.hpp>
#include "modelbits.hpp"
#include "matrixw.hpp"
#include "randomeffects.hpp"
#include "matrixfield.h"

namespace glmmr {

enum class SE {
  GLS = 0,
    KR = 1,
    Robust = 2,
    BW = 3,
    KR2 = 4,
    Sat = 5,
    KRBoth = 6 // used for when two types of correction are required
};

enum class IM {
  EIM = 0,
    OIM = 1,
    GEE_IND = 2,
    EIM2 = 3
};

}

struct BoxResults {
  dblvec dof;
  dblvec scale;
  dblvec test_stat;
  dblvec p_value;
  BoxResults(const int r) : dof(r), scale(r), test_stat(r), p_value(r) {};
};

struct CorrectionDataBase {
public:
  MatrixXd vcov_beta;
  MatrixXd vcov_theta;
  VectorXd dof;
  VectorXd lambda;
  CorrectionDataBase(int n1, int m1, int n2, int m2): vcov_beta(n1,m1), vcov_theta(n2,m2), dof(n1), lambda(n1) {};
  CorrectionDataBase(const MatrixXd& vcov_beta_, const MatrixXd& vcov_theta_, const MatrixXd& dof_, const MatrixXd& lambda_) : 
    vcov_beta(vcov_beta_), vcov_theta(vcov_theta_), dof(dof_), lambda(lambda_)  {};
  CorrectionDataBase(const CorrectionDataBase& x) : vcov_beta(x.vcov_beta), vcov_theta(x.vcov_theta), dof(x.dof), lambda(x.lambda) {};
  CorrectionDataBase& operator=(const CorrectionDataBase& x) = default;
};

template<glmmr::SE corr>
struct CorrectionData : public CorrectionDataBase {
public:
  CorrectionData(int n1, int m1, int n2, int m2): CorrectionDataBase(n1,m1,n2,m2) {};
  CorrectionData(const CorrectionData& x) : CorrectionDataBase(x.vcov_beta, x.vcov_theta, x.dof, x.lambda) {};
  CorrectionData& operator=(const CorrectionData& x){
    CorrectionDataBase::operator=(x);
    return *this;
  };
};

template<>
struct CorrectionData<glmmr::SE::KRBoth> : public CorrectionDataBase {
public:
  MatrixXd vcov_beta_second;
  CorrectionData(int n1, int m1, int n2, int m2): CorrectionDataBase(n1,m1,n2,m2), vcov_beta_second(n1,m1) {};
  CorrectionData(const CorrectionData& x) : CorrectionDataBase(x.vcov_beta, x.vcov_theta, x.dof, x.lambda), vcov_beta_second(x.vcov_beta_second) {};
  CorrectionData& operator=(const CorrectionData& x){
    CorrectionDataBase::operator=(x);
    vcov_beta_second = x.vcov_beta_second;
    return *this;
  };
};

namespace glmmr {

using namespace Eigen;

template<typename modeltype>
class ModelMatrix{
public:
  modeltype&                        model;
  glmmr::MatrixW<modeltype>         W;
  glmmr::RandomEffects<modeltype>&  re;
  // constructors
  ModelMatrix(modeltype& model_, glmmr::RandomEffects<modeltype>& re_);
  ModelMatrix(const glmmr::ModelMatrix<modeltype>& matrix);
  ModelMatrix(modeltype& model_, glmmr::RandomEffects<modeltype>& re_, bool useBlock_, bool useSparse_);
  // functions
  MatrixXd                information_matrix();
  MatrixXd                Sigma(bool inverse = false);
  template<IM imtype>
  MatrixXd                observed_information_matrix();
  MatrixXd                sandwich_matrix(); 
  std::vector<MatrixXd>   sigma_derivatives();
  template<IM imtype>
  MatrixXd                information_matrix_theta();
  template<SE corr, IM imtype>
  CorrectionData<corr>    small_sample_correction();
  MatrixXd                linpred();
  VectorMatrix            b_score();
  VectorMatrix            re_score();
  MatrixXd                hessian_nonlinear_correction();
  VectorXd                log_gradient(const VectorXd &v,bool beta = false);
  void                    gradient_eta(const VectorXd &v,ArrayXd& size_n_array);
  std::vector<glmmr::SigmaBlock> get_sigma_blocks();
  BoxResults              box();
  int                     P() const;
  int                     Q() const;
  MatrixXd                residuals(const int type, bool conditional = true);
  
private:
  std::vector<glmmr::SigmaBlock>  sigma_blocks;
  void                            gen_sigma_blocks();
  MatrixXd                        sigma_block(int b, bool inverse = false);
  MatrixXd                        sigma_builder(int b, bool inverse = false);
  MatrixXd                        information_matrix_by_block(int b);
  bool                            useBlock = true;
  bool                            useSparse = true;
};

}

template<typename modeltype>
inline glmmr::ModelMatrix<modeltype>::ModelMatrix(modeltype& model_, glmmr::RandomEffects<modeltype>& re_): model(model_), W(model_), re(re_) { gen_sigma_blocks();};

template<typename modeltype>
inline glmmr::ModelMatrix<modeltype>::ModelMatrix(const glmmr::ModelMatrix<modeltype>& matrix) : model(matrix.model), W(matrix.W), re(matrix.re) { gen_sigma_blocks();};

template<typename modeltype>
inline glmmr::ModelMatrix<modeltype>::ModelMatrix(modeltype& model_, glmmr::RandomEffects<modeltype>& re_, bool useBlock_, bool useSparse_): model(model_), W(model_), re(re_) { 
  useBlock = useBlock_;
  useSparse = useSparse_;
  if(useBlock)gen_sigma_blocks();};

template<typename modeltype>
inline std::vector<glmmr::SigmaBlock> glmmr::ModelMatrix<modeltype>::get_sigma_blocks(){
  return sigma_blocks;
}

template<typename modeltype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::information_matrix_by_block(int b){
  ArrayXi rows = Map<ArrayXi,Unaligned>(sigma_blocks[b].RowIndexes.data(),sigma_blocks[b].RowIndexes.size());
  MatrixXd X = glmmr::Eigen_ext::submat(model.linear_predictor.X(),rows,ArrayXi::LinSpaced(P(),0,P()-1));
  MatrixXd S = sigma_block(b,true);
  MatrixXd M = X.transpose()*S*X;
  return M;
}

template<typename modeltype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::information_matrix(){
  W.update();
  MatrixXd M = MatrixXd::Zero(P(),P());
  for(int i = 0; i< sigma_blocks.size(); i++){
    M += information_matrix_by_block(i);
  }
  return M;
}

template<typename modeltype>
inline void glmmr::ModelMatrix<modeltype>::gen_sigma_blocks(){
  int block_counter = 0;
  intvec2d block_ids(model.n());
  int block_size;
  sparse Z = model.covariance.Z_sparse();
  int i,j,k;
  auto it_begin = Z.Ai.begin();
  for(int b = 0; b < model.covariance.B(); b++){
    block_size = model.covariance.block_dim(b);
    for(i = 0; i < block_size; i++){
#pragma omp parallel for shared(it_begin, i)
      for(j = 0; j < model.n(); j++){
        auto it = std::find(it_begin + Z.Ap[j], it_begin + Z.Ap[j+1], (i+block_counter));
        if(it != (it_begin + Z.Ap[j+1])){
          block_ids[j].push_back(b);
        }
      }
    }
    block_counter += block_size;
  }
  intvec idx_matches;
  int n_matches;
  for(i = 0; i < model.n(); i++){
    if(sigma_blocks.size() == 0){
      glmmr::SigmaBlock newblock(block_ids[i]);
      newblock.add_row(i);
      sigma_blocks.push_back(newblock);
    } else {
      for(j = 0; j < sigma_blocks.size(); j++){
        if(sigma_blocks[j] == block_ids[i]){
          idx_matches.push_back(j);
        }
      }
      n_matches = idx_matches.size();
      if(n_matches==0){
        glmmr::SigmaBlock newblock(block_ids[i]);
        newblock.add_row(i);
        sigma_blocks.push_back(newblock);
      } else if(n_matches==1){
        sigma_blocks[idx_matches[0]].add(block_ids[i]);
        sigma_blocks[idx_matches[0]].add_row(i);
      } else if(n_matches>1){
        std::reverse(idx_matches.begin(),idx_matches.end());
        for(k = 0; k < (n_matches-1); k++){
          sigma_blocks[idx_matches[n_matches-1]].merge(sigma_blocks[idx_matches[k]]);
          sigma_blocks[idx_matches[n_matches-1]].add_row(i);
          sigma_blocks.erase(sigma_blocks.begin()+idx_matches[k]);
        }
      }
    }
    idx_matches.clear();
  }
}
template<typename modeltype>
inline int glmmr::ModelMatrix<modeltype>::P() const {
  return model.linear_predictor.P();
}


template<typename modeltype>
inline int glmmr::ModelMatrix<modeltype>::Q() const {
  return model.covariance.Q();
}

template<typename modeltype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::Sigma(bool inverse){
  W.update();
  MatrixXd S(model.n(), model.n());
  if(useBlock){
    S = sigma_builder(0,inverse);
  } else {
    MatrixXd ZL = model.covariance.ZL();
    S = ZL * ZL.transpose();
    S += W.W().array().inverse().matrix().asDiagonal();
    if(inverse){
      S = S.llt().solve(MatrixXd::Identity(S.rows(),S.cols()));
    }
  }
  return S;
}

template<typename modeltype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::sigma_block(int b,
                                                           bool inverse){
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
  if(b >= sigma_blocks.size())Rcpp::stop("Index out of range");
#endif
  // UPDATE THIS TO NOT USE SPARSE IF DESIRED
  sparse ZLs = submat_sparse(model.covariance.ZL_sparse(),sigma_blocks[b].RowIndexes);
  MatrixXd ZL = sparse_to_dense(ZLs,false);
  MatrixXd S = ZL * ZL.transpose();
  for(int i = 0; i < S.rows(); i++){
    S(i,i)+= 1/W.W()(sigma_blocks[b].RowIndexes[i]);
  }
  if(inverse){
    S = S.llt().solve(MatrixXd::Identity(S.rows(),S.cols()));
  }
  return S;
}

template<typename modeltype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::sigma_builder(int b,
                                                             bool inverse){
  int B_ = sigma_blocks.size();
  if (b == B_ - 1) {
    return sigma_block(b,inverse);
  }
  else {
    MatrixXd mat1 = sigma_block(b,inverse);
    MatrixXd mat2;
    if (b == B_ - 2) {
      mat2 = sigma_block(b+1,inverse);
    }
    else {
      mat2 = sigma_builder(b + 1,  inverse);
    }
    int n1 = mat1.rows();
    int n2 = mat2.rows();
    MatrixXd dmat = MatrixXd::Zero(n1+n2, n1+n2);
    dmat.block(0,0,n1,n1) = mat1;
    dmat.block(n1, n1, n2, n2) = mat2;
    return dmat;
  }
}

template<typename modeltype>
template<glmmr::IM imtype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::observed_information_matrix(){
  MatrixXd X = model.linear_predictor.X();
  bool nonlinear_w = model.family.family != Fam::gaussian || (model.data.weights != 1).any();
  if constexpr (imtype == IM::EIM){
    W.update();
    MatrixXd WX = X;
    if (nonlinear_w) WX.applyOnTheLeft(W.W_.asDiagonal());
    MatrixXd XtXW = X.transpose() * WX;
    MatrixXd ZL = model.covariance.ZL();
    MatrixXd WZL = ZL;
    if (nonlinear_w) WZL.applyOnTheLeft(W.W_.asDiagonal());
    MatrixXd XtWZL = X.transpose() * WZL;
    MatrixXd ZLWLZ = ZL.transpose() * WZL;
    if (!nonlinear_w) {
        XtXW *= (1.0 / model.data.var_par);
        XtWZL *= (1.0 / model.data.var_par);
        ZLWLZ *= (1.0 / model.data.var_par);
    }
    ZLWLZ += MatrixXd::Identity(Q(),Q());
    MatrixXd infomat(P()+Q(),P()+Q());
    infomat.topLeftCorner(P(),P()) = XtXW;
    infomat.topRightCorner(P(),Q()) = XtWZL;
    infomat.bottomLeftCorner(Q(),P()) = XtWZL.transpose();
    infomat.bottomRightCorner(Q(),Q()) = ZLWLZ;
    return infomat;
  } else if constexpr (imtype == IM::OIM){
    MatrixXd M = information_matrix();
    MatrixXd Mt = information_matrix_theta<IM::OIM>();
    int npar = Mt.cols();
    MatrixXd infomat(P()+npar,P()+npar);
    infomat.topLeftCorner(P(),P()) = -1.0*M;
    infomat.topRightCorner(P(),npar) = -1.0*Mt.bottomRows(P());
    infomat.bottomLeftCorner(npar,P()) = -1.0*Mt.bottomRows(P()).transpose();
    infomat.bottomRightCorner(npar,npar) = Mt.topRows(npar);
    return -1.0*infomat;
  } else if constexpr (imtype == IM::EIM2) {
    W.update();
    MatrixXd WX = X;
    if (nonlinear_w) WX.applyOnTheLeft(W.W_.asDiagonal());
    MatrixXd Z = model.covariance.Z();
    MatrixXd WZ = Z;
    if (nonlinear_w) WZ.applyOnTheLeft(W.W_.asDiagonal());

    MatrixXd D = model.covariance.D();
    if (model.covariance.all_group_re()) {
        for (int i = 0; i < D.rows(); i++)D(i, i) = 1.0 / D(i, i);
    } else {
        D = D.llt().solve(MatrixXd::Identity(D.rows(), D.cols()));
    }    
    MatrixXd infomat(P()+Q(),P()+Q());
    infomat.topLeftCorner(P(),P()) = X.transpose() * WX;
    infomat.topRightCorner(P(),Q()) = X.transpose() * WZ;
    infomat.bottomLeftCorner(Q(),P()) = infomat.topRightCorner(P(), Q()).transpose();
    infomat.bottomRightCorner(Q(),Q()) = WZ.transpose() * Z + D;
    return infomat;
  } else {
    MatrixXd M = information_matrix();
    M = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
    MatrixXd XtXW = X.transpose() * W.W_.asDiagonal() * X;
    M = XtXW * M * XtXW;
    return M;
  }
}

template<typename modeltype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::sandwich_matrix(){
  // none of these produce great results! Not sure what the best strategy is here.
  
  // MatrixXd XandZ(model.n(),P()+Q());
  // XandZ.leftCols(P()) = model.linear_predictor.Xdata;
  // XandZ.rightCols(Q()) = model.covariance.ZL();
  // dblvec bu;
  // for(const auto& b: model.linear_predictor.parameters) bu.push_back(b);
  // VectorXd umean = re.u(false).rowwise().mean();
  // for(int i = 0; i < umean.size(); i++) bu.push_back(umean(i));
  // MatrixMatrix result = model.vcalc.jacobian_and_hessian(bu,XandZ,model.data.offset);
  // MatrixXd JmatFull = result.mat2 * result.mat2.transpose();
  // MatrixXd Jmat = JmatFull.block(0,0,P(),P());
  // MatrixXd invHFull = -1.0*result.mat1;
  // invHFull = invHFull.llt().solve(MatrixXd::Identity(invHFull.rows(),invHFull.cols()));
  // MatrixXd invH = invHFull.block(0,0,P(),P());
  // MatrixXd sandwich = invH * Jmat * invH;
  // 
  // 
  // #if defined(R_BUILD) && defined(ENABLE_DEBUG)
  //   if(result.mat1.rows() <= P())Rcpp::stop("mat1 <= p");
  //   Rcpp::Rcout << "\nSANDWICH\n";
  //   Rcpp::Rcout << "\nHessian: \n" << -1.0*result.mat1.block(0,0,P(),P());
  //   Rcpp::Rcout << "\nInverse Hessian: \n" << invH;
  //   Rcpp::Rcout << "\nGradient: \n" << Jmat;
  //   Rcpp::Rcout << "\nSandwich: \n" << sandwich;
  // #endif
  // 
  // return sandwich;
  
  MatrixXd infomat = information_matrix();
  infomat = infomat.llt().solve(MatrixXd::Identity(P(),P()));
  MatrixXd X = model.linear_predictor.X();
  MatrixXd S = Sigma(true);
  MatrixXd SX = S*X;
  // MatrixXd resid_sum = MatrixXd::Zero(X.rows(),X.rows());
  // MatrixXd zd = linpred();
  VectorXd resid = model.linear_predictor.xb()+model.data.offset;
  resid = model.data.y - glmmr::maths::mod_inv_func(resid, model.family.link);
  MatrixXd resid_sum = resid * resid.transpose();
  // int niter = zd.cols();
  // for(int i = 0; i < niter; ++i){
  //   zd.col(i) = glmmr::maths::mod_inv_func(zd.col(i), model.family.link);
  //   if(model.family.family == Fam::binomial){
  //     zd.col(i) = zd.col(i).cwiseProduct(model.data.variance.matrix());
  //   }
  //   zd.col(i) = (model.data.y - zd.col(i))/((double)niter);
  // }
  // MatrixXd resid_sum = zd * zd.transpose();//*= niterinv;
  
#if defined(R_BUILD) && defined(ENABLE_DEBUG)
  Rcpp::Rcout << "\nSANDWICH\n";
  int size = resid_sum.rows() < 10 ? resid_sum.rows() : 10;
  Rcpp::Rcout << "\nResidual cross prod: \n" << resid_sum.block(0,0,size,size).transpose();
#endif
  
  MatrixXd robust = infomat * SX.transpose() * resid_sum * SX * infomat;//(infomat.rows(),infomat.cols());
  
  // if(type == SandwichType::CR2){
  //   MatrixXd H = MatrixXd::Identity(X.rows(),X.rows()) - X*infomat * SX.transpose();
  //   H = H.llt().solve(MatrixXd::Identity(X.rows(),X.rows()));
  //   MatrixXd HL(H.llt().matrixL());
  //   robust = infomat * SX.transpose()  * HL * resid_sum * HL * SX * infomat;
  // } else {
  //   robust = infomat * SX.transpose() * resid_sum * SX * infomat;
  // }
  return robust;
}

template<typename modeltype>
inline std::vector<MatrixXd> glmmr::ModelMatrix<modeltype>::sigma_derivatives()
{
  std::vector<MatrixXd> derivs;
  model.covariance.derivatives(derivs,2);
  return derivs;
}

template<typename modeltype>
template<glmmr::IM imtype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::information_matrix_theta()
{
  int n = model.n();
  std::vector<MatrixXd> derivs;
  model.covariance.derivatives(derivs,1);
  int R = model.covariance.npar();
  int Rmod = model.family.family==Fam::gaussian ? R+1 : R;
  MatrixXd SigmaInv = Sigma(true);
  MatrixXd Z = model.covariance.Z();
  MatrixXd M_theta = MatrixXd::Zero(Rmod,Rmod);
  glmmr::MatrixField<MatrixXd> S;
  VectorXd resid(1);
  
  // residuals in case of OIM
  if constexpr (imtype == IM::OIM){
    if(model.data.y.size() != model.n()) throw std::runtime_error("y data not correct size");
    resid.resize(model.n());
    resid = model.linear_predictor.xb()+model.data.offset;
    resid = model.data.y - glmmr::maths::mod_inv_func(resid, model.family.link);
  }
  
  for(int i = 0; i < Rmod; i++){
    MatrixXd partial0(model.n(),model.n());
    if(i < R){
      partial0 = Z*derivs[1+i]*Z.transpose();
    } else {
      partial0 = MatrixXd::Identity(n,n);
      if((model.data.weights != 1).any())partial0 = model.data.weights.inverse().matrix().asDiagonal();
    }
    S.add(partial0);
  }
  
  if(useBlock){
    for(int b = 0; b< sigma_blocks.size(); b++){
      ArrayXi rows = Map<ArrayXi,Unaligned>(sigma_blocks[b].RowIndexes.data(),sigma_blocks[b].RowIndexes.size());
      MatrixXd SigmaInvsub = glmmr::Eigen_ext::submat(SigmaInv,rows,rows);
      VectorXd residsub(1);
      if constexpr (imtype == IM::OIM){
        residsub.resize(rows.size());
        for(int r = 0; r < rows.size(); r++) residsub(r) = resid(rows(r));
      }
      for(int i = 0; i < Rmod; i++){
        MatrixXd Ssub1 = glmmr::Eigen_ext::submat(S(i),rows,rows);
        MatrixXd SPS = SigmaInvsub * Ssub1 * SigmaInvsub;
        for(int j = i; j < Rmod; j++){
          MatrixXd Ssub2 = glmmr::Eigen_ext::submat(S(j),rows,rows);
          double oim_adj;
          if constexpr (imtype == IM::OIM){
            oim_adj = (residsub.transpose() * SPS * Ssub2 * SigmaInvsub * residsub)(0);
          }
          for(int k = 0; k < rows.size(); k++){
            for(int l = 0; l < rows.size(); l++){
              M_theta(i,j) += 0.5*SPS(k,l)*Ssub2(l,k);
              if constexpr (imtype == IM::OIM) M_theta(i,j) -= oim_adj;
            }
          }
          if(i!=j)M_theta(j,i)=M_theta(i,j);
        }
      }
    }
  } else {
    for(int i = 0; i < Rmod; i++){
      MatrixXd SPS = SigmaInv * S(i) * SigmaInv;
      for(int j = i; j < Rmod; j++){
        MatrixXd Ssub2 = S(j);
        double oim_adj;
        if constexpr (imtype == IM::OIM){
          oim_adj = (resid.transpose() * SPS * Ssub2 * SigmaInv * resid)(0);
        }
        M_theta(i,j) = 0;
        for(int k = 0; k < model.n(); k++){
          for(int l = 0; l <  model.n(); l++){
            M_theta(i,j) += 0.5*SPS(k,l)*Ssub2(l,k);
            if constexpr (imtype == IM::OIM) M_theta(i,j) -= oim_adj;
          }
        }
        if(i!=j)M_theta(j,i)=M_theta(i,j);
      }
    }
  }
  
  if constexpr (imtype == IM::OIM){
    // add in the beta-theta part of the matrix
    MatrixXd X = model.linear_predictor.X();
    MatrixXd Mbt(X.cols(),M_theta.cols());
    if(useBlock){
      for(int b = 0; b< sigma_blocks.size(); b++){
        Mbt.setZero();
        ArrayXi rows = Map<ArrayXi,Unaligned>(sigma_blocks[b].RowIndexes.data(),sigma_blocks[b].RowIndexes.size());
        MatrixXd Xsub = glmmr::Eigen_ext::submat(X,rows,ArrayXi::LinSpaced(P(),0,P()-1));
        MatrixXd SigmaInvsub = glmmr::Eigen_ext::submat(SigmaInv,rows,rows);
        VectorXd residsub(rows.size());
        for(int r = 0; r < rows.size(); r++) residsub(r) = resid(rows(r));
        for(int i =0; i < M_theta.cols(); i++){
          MatrixXd Ssub1 = glmmr::Eigen_ext::submat(S(i),rows,rows);
          Mbt.col(i) += Xsub.transpose() * SigmaInvsub * Ssub1 * SigmaInvsub * residsub;
        }
      }
    } else {
      for(int i =0; i < M_theta.cols(); i++) Mbt.col(i) = X.transpose() * SigmaInv * S(i) * SigmaInv * resid;
    }
    M_theta.conservativeResize(M_theta.cols()+X.cols(),M_theta.cols());
    M_theta.bottomRows(X.cols()) = Mbt;
  }
  return M_theta;
}

template<typename modeltype>
template<glmmr::SE corr, glmmr::IM imtype>
inline CorrectionData<corr> glmmr::ModelMatrix<modeltype>::small_sample_correction()
{
  using namespace glmmr;
  static_assert(corr == SE::KR || corr == SE::KR2 || corr == SE::Sat || corr == SE::KRBoth,"Only Kenward-Roger or Satterthwaite allowed for small sample correction");
  
  int n = model.n();
  std::vector<MatrixXd> derivs;
  model.covariance.derivatives(derivs,2);
  int R = model.covariance.npar();
  int Rmod = model.family.family==Fam::gaussian ? R+1 : R;
  
  MatrixXd M = information_matrix();
  M = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
  MatrixXd M_new(M);
  MatrixXd SigmaInv = Sigma(true);
  MatrixXd Z = model.covariance.Z();
  MatrixXd X = model.linear_predictor.X();
  MatrixXd M_theta = MatrixXd::Zero(Rmod,Rmod);
  MatrixXd meat = MatrixXd::Zero(X.cols(),X.cols());
  MatrixField<MatrixXd> Pmat;
  MatrixField<MatrixXd> Q;
  MatrixField<MatrixXd> RR;
  MatrixField<MatrixXd> S;
  // in case of OIM
  MatrixXd M_theta_oim = MatrixXd::Zero(Rmod,Rmod);
  MatrixXd M_theta_kr = MatrixXd::Zero(Rmod,Rmod);
  VectorXd resid(1);
  int counter, counter_rr;
  
  // residuals in case of OIM
  if constexpr (imtype == IM::OIM){
    if(model.data.y.size() != model.n()) throw std::runtime_error("y data not correct size");
    resid.resize(model.n());
    resid = model.linear_predictor.xb()+model.data.offset;
    resid = model.data.y - glmmr::maths::mod_inv_func(resid, model.family.link);
  }
  
  for(int i = 0; i < Rmod; i++){
    MatrixXd partial0(model.n(),model.n());
    if(i < R){
      partial0 = Z*derivs[1+i]*Z.transpose();
    } else {
      partial0 = MatrixXd::Identity(n,n);
      if((model.data.weights != 1).any())partial0 = model.data.weights.inverse().matrix().asDiagonal();
    }
    S.add(partial0);
  }
  
  for(int i = 0; i < Rmod; i++){
    for(int j = i; j < Rmod; j++){
      if(i < R && j < R){
        int scnd_idx = i + j*(R-1) - j*(j-1)/2;
        S.add(Z*derivs[R+1+scnd_idx]*Z.transpose());
      }
    }
  }
  
  if(useBlock){
    for(int b = 0; b< sigma_blocks.size(); b++){
      ArrayXi rows = Map<ArrayXi,Unaligned>(sigma_blocks[b].RowIndexes.data(),sigma_blocks[b].RowIndexes.size());
      MatrixXd SigmaInvsub = glmmr::Eigen_ext::submat(SigmaInv,rows,rows);
      MatrixXd Xsub = glmmr::Eigen_ext::submat(X,rows,ArrayXi::LinSpaced(P(),0,P()-1));
      VectorXd residsub(1);
      if constexpr (imtype == IM::OIM){
        residsub.resize(rows.size());
        for(int r = 0; r < rows.size(); r++) residsub(r) = resid(rows(r));
      }
      counter = 0;
      counter_rr = 0;
      for(int i = 0; i < Rmod; i++){
        MatrixXd Ssub1 = glmmr::Eigen_ext::submat(S(i),rows,rows);
        MatrixXd SPS = SigmaInvsub * Ssub1 * SigmaInvsub;
        // create P matrix
        if(b==0){
          Pmat.add(MatrixXd::Zero(P(),P()));
        }
        Pmat.sum(i,-1.0*Xsub.transpose()*SPS*Xsub);
        
        for(int j = i; j < Rmod; j++){
          MatrixXd Ssub2 = glmmr::Eigen_ext::submat(S(j),rows,rows);
          
          // Q and R matrices
          if constexpr (corr == SE::KR || corr == SE::KR2 || corr == SE::KRBoth){
            if(b==0){
              Q.add(MatrixXd::Zero(P(),P()));
              if(i < R && j < R) RR.add(MatrixXd::Zero(P(),P()));
            }
            Q.sum(counter,Xsub.transpose()*SPS*Ssub2*SigmaInvsub*Xsub);
            counter++;
            // RR here
            if(i < R && j < R){
              int scnd_idx = i + j*(R-1) - j*(j-1)/2;
              MatrixXd Ssub3 = glmmr::Eigen_ext::submat(S(Rmod+scnd_idx),rows,rows);
              RR.sum(counter_rr,Xsub.transpose() * SigmaInvsub * Ssub3 * SigmaInvsub * Xsub);
              counter_rr++;
            }
          }
          
          double oim_adj;
          if constexpr (imtype == IM::OIM){
            oim_adj = (residsub.transpose() * SPS * Ssub2 * SigmaInvsub * residsub)(0);
          }
          for(int k = 0; k < rows.size(); k++){
            for(int l = 0; l < rows.size(); l++){
              M_theta(i,j) += 0.5*SPS(k,l)*Ssub2(l,k);
              if constexpr (imtype == IM::OIM) M_theta_oim(i,j) -= oim_adj;
            }
          }
          if(i!=j)M_theta(j,i)=M_theta(i,j);
          if constexpr (imtype == IM::OIM) if(i!=j)M_theta_oim(j,i)=M_theta_oim(i,j);
        }
      }
    }
  } else {
    for(int i = 0; i < Rmod; i++){
      MatrixXd SPS = SigmaInv * S(i) * SigmaInv;
      Pmat.add(X.transpose() * SPS * X);
      for(int j = i; j < Rmod; j++){
        MatrixXd Ssub2 = S(j);
        
        if constexpr (corr == SE::KR || corr == SE::KR2 || corr == SE::KRBoth){
          Q.add(X.transpose()*SPS*Ssub2*SigmaInv*X);
          // RR here
          if(i < R && j < R){
            int scnd_idx = i + j*(R-1) - j*(j-1)/2;
            RR.add(X.transpose() * SigmaInv * S(Rmod+scnd_idx) * SigmaInv * X);
          }
        }
        
        double oim_adj;
        if constexpr (imtype == IM::OIM){
          oim_adj = (resid.transpose() * SPS * Ssub2 * SigmaInv * resid)(0);
        }
        M_theta(i,j) = 0;
        for(int k = 0; k < model.n(); k++){
          for(int l = 0; l <  model.n(); l++){
            M_theta(i,j) += 0.5*SPS(k,l)*Ssub2(l,k);
            if constexpr (imtype == IM::OIM) M_theta_oim(i,j) -= oim_adj;
          }
        }
        if(i!=j)M_theta(j,i)=M_theta(i,j);
        if constexpr (imtype == IM::OIM) if(i!=j)M_theta_oim(j,i)=M_theta_oim(i,j);
      }
    }
  }
  
  if constexpr (corr == SE::KR || corr == SE::KR2 || corr == SE::KRBoth){
    counter = 0;
    for(int i = 0; i < Rmod; i++){
      for(int j = i; j < Rmod; j++){
        M_theta_kr(i,j) += -1.0*(M*Q(counter)).trace() + 0.5*((M*Pmat(i)*M*Pmat(j)).trace());
        if(i!=j)M_theta_kr(j,i)=M_theta_kr(i,j);
        counter++;
      }
    }
  }
  
  if constexpr (imtype == IM::OIM){
    // add in the beta-theta part of the matrix
    MatrixXd Mbt(X.cols(),M_theta.cols());
    if(useBlock){
      for(int b = 0; b< sigma_blocks.size(); b++){
        Mbt.setZero();
        ArrayXi rows = Map<ArrayXi,Unaligned>(sigma_blocks[b].RowIndexes.data(),sigma_blocks[b].RowIndexes.size());
        MatrixXd Xsub = glmmr::Eigen_ext::submat(X,rows,ArrayXi::LinSpaced(P(),0,P()-1));
        MatrixXd SigmaInvsub = glmmr::Eigen_ext::submat(SigmaInv,rows,rows);
        VectorXd residsub(rows.size());
        for(int r = 0; r < rows.size(); r++) residsub(r) = resid(rows(r));
        for(int i =0; i < M_theta.cols(); i++){
          MatrixXd Ssub1 = glmmr::Eigen_ext::submat(S(i),rows,rows);
          Mbt.col(i) += Xsub.transpose() * SigmaInvsub * Ssub1 * SigmaInvsub * residsub;
        }
      }
    } else {
      for(int i =0; i < M_theta.cols(); i++) Mbt.col(i) = X.transpose() * SigmaInv * S(i) * SigmaInv * resid;
    }
    
    MatrixXd Moim(this->P()+Rmod,this->P()+Rmod);
    Moim.topLeftCorner(this->P(),this->P()) = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
    Moim.topRightCorner(this->P(),Rmod) = Mbt;
    Moim.bottomLeftCorner(Rmod,this->P()) = Mbt.transpose();
    Moim.bottomRightCorner(Rmod,Rmod) = -1.0*(M_theta + M_theta_oim);
    Moim = Moim.llt().solve(MatrixXd::Identity(Moim.rows(),Moim.cols()));
    M = Moim.topLeftCorner(this->P(),this->P());
    M_theta = -1.0*(M_theta + M_theta_oim);
  }
  
  if constexpr (corr == SE::KR || corr == SE::KR2 || corr == SE::KRBoth ) M_theta += M_theta_kr;
  M_theta = M_theta.llt().solve(MatrixXd::Identity(Rmod,Rmod));
  
  if constexpr (corr == SE::KR || corr == SE::KR2 || corr == SE::KRBoth ){
    for(int i = 0; i < (Rmod-1); i++){
      for(int j = (i+1); j < Rmod; j++){
        int scnd_idx = i + j*(Rmod-1) - j*(j-1)/2;
        meat += M_theta(i,j)*(Q(scnd_idx) + Q(scnd_idx).transpose() - Pmat(i)*M*Pmat(j) -Pmat(j)*M*Pmat(i));//(SigX.transpose()*partial1*PG*partial2*SigX);//
        if(i < R && j < R){
          scnd_idx = i + j*(R-1) - j*(j-1)/2;
          meat -= 0.5*M_theta(i,j)*(RR(scnd_idx));
        }
      }
    }
    for(int i = 0; i < Rmod; i++){
      int scnd_idx = i + i*(Rmod-1) - i*(i-1)/2;
      meat += M_theta(i,i)*(Q(scnd_idx) - Pmat(i)*M*Pmat(i));
      if(i < R){
        scnd_idx = i + i*(R-1) - i*(i-1)/2;
        meat -= 0.25*M_theta(i,i)*RR(scnd_idx);
      }
    }
    M_new = M + 2*M*meat*M;
    
#if defined(R_BUILD) && defined(ENABLE_DEBUG)
    Rcpp::Rcout << "\n(K-R) First correction matrix: \n" << 2*M*meat*M;
#endif
  } else {
    M_new = M;
  }
  
  CorrectionData<corr> out(this->P(),this->P(),Rmod,Rmod);
  out.vcov_beta = M_new;
  out.vcov_theta = M_theta;
  if constexpr (corr == SE::KRBoth) out.vcov_beta_second = M_new;
  
  // new improved correction
  // this code can also be sped up by doing it by blocks
  if constexpr (corr == SE::KR2 || corr == SE::KRBoth){
    MatrixXd SigX = SigmaInv*X;
    MatrixXd SS = MatrixXd::Zero(SigmaInv.rows(),SigmaInv.cols());
    for(int i = 0; i < Rmod; i++){
      for(int j = i; j < Rmod; j++){
        if(i < R && j < R){
          int scnd_idx = i + j*(R-1) - j*(j-1)/2;
          double rep = i == j ? 1.0 : 2.0;
          SS += rep * M_theta(i,j)*S(Rmod+scnd_idx)*SigmaInv;
        }
      }
    }
    SS.applyOnTheLeft(SigmaInv);
    MatrixXd XSXM = X.transpose()*SS*X*M;
    dblvec V(Rmod,0.0);
    for(int i = 0; i < Rmod; i++){
      V[i] += (SS*S(i)).trace();
      V[i] -= 2*((SigX.transpose()*S(i)*SS*X*M).trace());
      V[i] += ((XSXM*SigX.transpose()*S(i)*SigX*M).trace());
    }
    MatrixXd M_correction = MatrixXd::Zero(M.rows(),M.cols());
    for(int i = 0; i < Rmod; i++){
      for(int j = i; j < Rmod; j++){
        double rep = i == j ? 0.25 : 0.5;
        M_correction += rep*M_theta(i,j)*V[j]*M*Pmat(i)*M;
      }
    }
    
#if defined(R_BUILD) && defined(ENABLE_DEBUG)
    Rcpp::Rcout << "\n(K-R) Improved correction matrix: \n" << M_correction;
    Rcpp::Rcout << "\nV: ";
    for(const auto& v: V)Rcpp::Rcout << v << " ";
#endif
    
    if constexpr (corr == SE::KR2) {
      out.vcov_beta -= M_correction;
    } else {
      out.vcov_beta_second -= M_correction;
    }
  }
  
  // degrees of freedom correction
  
  double a1, a2, B, g, c1, c2, c3, v0, v1, v2, rhotop, rho;
  int mult = 1;
  VectorXd L = VectorXd::Zero(this->P());
  MatrixXd Theta(this->P(),this->P());
  for(int p = 0; p < L.size(); p++){
    L.setZero();
    L(p) = 1;
    double vlb = L.transpose() * M * L;
    Theta = (1/vlb)*(L*L.transpose());
    Theta = Theta*M;
    a1 = 0; 
    a2 = 0;
    for(int i = 0; i < Rmod; i++){
      for(int j = i; j < Rmod; j++){
        mult = i==j ? 1 : 2;
        a1 += mult*M_theta(i,j)*(Theta*Pmat(i)*M).trace()*(Theta*Pmat(j)*M).trace();
        a2 += mult*M_theta(i,j)*(Theta*Pmat(i)*M*Theta*Pmat(j)*M).trace();
      }
    }
    B = (a1 + 6*a2)*0.5;
    g = (2*a1 - 5*a2)/(3*a2);
    c1 = g/(3+2*(1-g));
    c2 = (1-g)/(3+2*(1-g));
    c3 = (3-g)/(3+2*(1-g));
    v0 = abs(1 + c1*B) < 1e-10 ? 0 : 1 + c1*B;
    v1 = 1 - c2*B;
    v2 = 1/(1 - c3*B);
    rhotop = abs(1-a2) < 1e-10 && abs(v1) < 1e-10 ? 1.0 : (1-a2)/v1;
    rho = rhotop*rhotop*v0*v2;
    out.dof(p) = 4 + 3/(rho-1);
    out.lambda(p) = (1-a2)*out.dof(p)/(out.dof(p) - 2);
  }
  
  return out;
}

template<typename modeltype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::linpred()
{
  return (re.zu_.colwise()+(model.linear_predictor.xb()+model.data.offset));
}

template<typename modeltype>
inline VectorMatrix glmmr::ModelMatrix<modeltype>::b_score()
{
  MatrixXd zuOffset = re.Zu();
  zuOffset.colwise() += model.data.offset;
  MatrixMatrix hess = model.calc.jacobian_and_hessian(zuOffset);
  VectorMatrix out(hess.mat1.rows());
  out.mat = hess.mat1;
  out.mat *= -1.0;
  out.vec = hess.mat2.rowwise().sum();
  return out;
}

// type = 0 - raw
// type = 1 - pearson
// type = 2 - standardised
template<typename modeltype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::residuals(const int type, bool conditional)
{
  int ncol_r = conditional ? re.u_.cols() : 1;
  MatrixXd resids(model.n(), ncol_r);
  VectorXd mu = glmmr::maths::mod_inv_func(model.linear_predictor.xb(), model.family.link);
  VectorXd yxb = model.data.y - mu;
  if(conditional){
    MatrixXd zuOffset = re.Zu();
    zuOffset.colwise() += model.data.offset;
    for(int i = 0; i < ncol_r; i++) resids.col(i) = yxb - zuOffset.col(i);
  } else {
    resids.col(0) = yxb;
  }
  if(type == 1){
    VectorXd var = glmmr::maths::marginal_var(mu, model.family.family, model.data.var_par);
    for(int i = 0; i < ncol_r; i++) resids.col(i).array() *= 1.0/sqrt(var(i));
  } else if(type == 2){
    VectorXd colmeans = resids.colwise().mean().transpose();
    double std_dev;
    for(int i = 0; i < ncol_r; i++){
      std_dev = sqrt((resids.col(i).array() - colmeans(i)).square().sum()/(resids.rows() - 1));
      resids.col(i).array() *= 1.0/std_dev; 
    }
  }
  return resids;
}


template<typename modeltype>
inline VectorMatrix glmmr::ModelMatrix<modeltype>::re_score(){
  VectorXd xbOffset = model.linear_predictor.xb() + model.data.offset;
  model.vcalc.parameters = dblvec(re.u(false).col(0).data(),re.u(false).col(0).data()+re.u(false).rows());
  model.vcalc.data = model.covariance.ZL();
  MatrixMatrix hess = model.vcalc.jacobian_and_hessian(Map<MatrixXd>(xbOffset.data(),xbOffset.size(),1));
  VectorMatrix out(Q());
  hess.mat1 *= -1.0;
  out.mat = hess.mat1 + MatrixXd::Identity(Q(),Q());
  out.vec = hess.mat2.rowwise().sum();
  out.vec -= re.u(false).col(0);
  return out;
}

template<typename modeltype>
inline BoxResults glmmr::ModelMatrix<modeltype>::box(){
  int r = P(); // to follow notation in Skene and Kenward
  BoxResults results(r);
  MatrixXd S = Sigma();
  MatrixXd X = model.linear_predictor.X();
  MatrixXd XtX = X.transpose()*X;
  XtX = XtX.llt().solve(MatrixXd::Identity(XtX.rows(),XtX.cols()));
  MatrixXd Px = X*XtX*X.transpose();
  MatrixXd A = MatrixXd::Identity(Px.rows(),Px.cols()) - Px;
  MatrixXd ASig = A*S;
  double asigtrace = ASig.trace();
  double trasig = ((ASig*ASig.transpose()).trace())/(asigtrace*asigtrace);
  MatrixXd Xr(model.n(),r-1);
  MatrixXd B(Px.rows(),Px.cols());
  MatrixXd XrtXr(r-1,r-1);
  for(int i = 0; i < r; i++){
    if(i == 0){
      Xr = X.rightCols(r-1);
    } else if(i == r-1){
      Xr = X.leftCols(r-1);
    } else {
      Xr.leftCols(i) = X.leftCols(i);
      Xr.rightCols(r-i-1) = Xr.rightCols(r-i-1);
    }
    XrtXr = Xr.transpose()*Xr;
    XrtXr = XrtXr.llt().solve(MatrixXd::Identity(r-1,r-1));
    B = Px - Xr*XrtXr*Xr.transpose();
    MatrixXd BSig = B*S;
    double bsigtrace = BSig.trace();
    double trbsig = ((BSig*BSig.transpose()).trace())/(bsigtrace*bsigtrace);
    double V = trbsig + trasig;
    results.dof[i] = ((4*V+1)-2)/(V-1);
    results.scale[i] = (X.rows() - r)*(results.dof[i] - 2)*bsigtrace/(results.dof[i] * asigtrace);
    results.test_stat[i] = (X.rows() - r)*((model.data.y.transpose()*B*model.data.y)(0))/((model.data.y.transpose()*A*model.data.y)(0));
    boost::math::fisher_f fdist(1.0,results.dof[i]);
    results.p_value[i] = 1 - boost::math::cdf(fdist,results.test_stat[i]/results.scale[i]);
  }
  return results;
}

template<typename modeltype>
inline void glmmr::ModelMatrix<modeltype>::gradient_eta(const VectorXd &v,
                                                        ArrayXd& size_n_array){
  
  if(size_n_array.size() != model.n())throw std::runtime_error("Size n array != n");
  size_n_array = model.xb();
  sparse ZL = model.covariance.ZL_sparse();
  size_n_array += (SparseOperators::operator*(ZL,v)).array();
  
  switch(model.family.family){
  case Fam::poisson:
  {
    switch(model.family.link){
  case Link::identity:
  {
    size_n_array = size_n_array.inverse();
    size_n_array = model.data.y.array()*size_n_array;
    size_n_array -= ArrayXd::Ones(model.n());
    break;
  }
  default:
  {
    size_n_array = size_n_array.exp();
    size_n_array = model.data.y.array() - size_n_array;
    break;
  }
  }
    break;
  }
  case Fam::bernoulli: case Fam::binomial:
  {
    switch(model.family.link){
  case Link::loglink:
  {
    ArrayXd logitxb = 1.0 - size_n_array.exp();
    logitxb = logitxb.inverse();
    logitxb *= size_n_array.exp();
    size_n_array = (model.data.y.array() - model.data.variance)*logitxb;
    size_n_array += model.data.y.array();
    break;
  }
  case Link::identity:
  {
    ArrayXd n_array2 = 1.0 - size_n_array;
    n_array2 = n_array2.inverse();
    n_array2 *= (model.data.variance - model.data.y.array());
    size_n_array = size_n_array.inverse();
    size_n_array *= model.data.y.array();
    size_n_array -= n_array2;
    break;
  }
  case Link::probit:
  {
    ArrayXd n_array2(model.n());
#pragma omp parallel for    
    for (int i = 0; i < model.n(); i++) {
      size_n_array(i) = glmmr::maths::gaussian_pdf(size_n_array(i)) / glmmr::maths::gaussian_cdf(size_n_array(i));
      n_array2(i) = -1.0 * glmmr::maths::gaussian_pdf(size_n_array(i)) / (1 - glmmr::maths::gaussian_cdf(size_n_array(i)));
    }
    size_n_array = model.data.y.array() * size_n_array + (model.data.variance - model.data.y.array()) * n_array2;
    break;
  }
  default:
    //logit
  {
    ArrayXd logitxb = size_n_array.exp();
    logitxb += 1.0;
    logitxb = logitxb.inverse();
    logitxb *= size_n_array.exp();
    size_n_array = model.data.y.array()*(ArrayXd::Constant(model.n(),1) - logitxb) - (model.data.variance - model.data.y.array())*logitxb;
    break;
  }
  }
    break;
  }
  case Fam::gaussian:
  {
    switch(model.family.link){
  case Link::loglink:
  {
    size_n_array = (model.data.y.array() - size_n_array)*model.data.weights*size_n_array.exp();
    break;
  }
  default:
  {
    size_n_array = model.data.y.array() - size_n_array;
    size_n_array *= model.data.weights/model.data.var_par;
    break;
  }
  }
    break;
  }
  case Fam::gamma:
  {
    switch(model.family.link){
  case Link::inverse:
  {
    size_n_array = size_n_array.inverse();
    size_n_array -= model.data.y.array();
    break;
  }
  case Link::identity:
  {
    size_n_array = size_n_array.inverse();
    size_n_array *= (model.data.y.array()*size_n_array - ArrayXd::Ones(model.n()));
    break;
  }
  default:
    //log
  {
    size_n_array *= -1.0;
    size_n_array = size_n_array.exp();
    size_n_array *= model.data.y.array();
    break;
  }
  }
    break;
  }
  case Fam::beta:
  {
#pragma omp parallel for 
    for(int i = 0; i < model.n(); i++){
      size_n_array(i) = exp(size_n_array(i))/(exp(size_n_array(i))+1);
      size_n_array(i) = (size_n_array(i)/(1+exp(size_n_array(i)))) * model.data.var_par * (log(model.data.y(i)) - log(1- model.data.y(i)) - boost::math::digamma(size_n_array(i)*model.data.var_par) + boost::math::digamma((1-size_n_array(i))*model.data.var_par));
    }
    break;
  }
  case Fam::quantile: case Fam::quantile_scaled: 
    {
    switch(model.family.link){
      case Link::identity:
        // size_n_array = (model.data.y.array() - size_n_array);
        if(model.family.family == Fam::quantile_scaled) size_n_array *= 1.0/model.data.var_par;
        for(int i = 0; i < model.n(); i++){
          if(size_n_array(i) <= 0){
            size_n_array(i) *= model.family.quantile - 1;
          } else {
            size_n_array(i) *= model.family.quantile;
          }
        }
        break;
    case Link::loglink:
      {
        // ArrayXd resid = (model.data.y.array() - size_n_array.exp());
        size_n_array = size_n_array.exp();
        if(model.family.family == Fam::quantile_scaled) size_n_array *= 1.0/model.data.var_par;
        for(int i = 0; i < model.n(); i++){
          if(size_n_array(i) <= 0){
            size_n_array(i) *= model.family.quantile - 1;
          } else {
            size_n_array(i) *= model.family.quantile;
          }
        }
        break;
      }
    case Link::logit:
    {
      ArrayXd logitxb = size_n_array.exp();
      logitxb += 1.0;
      logitxb = logitxb.inverse();
      logitxb *= size_n_array.exp();
      //ArrayXd resid = (model.data.y.array() - logitxb);
      logitxb *= (1+size_n_array.exp()).inverse();
      if(model.family.family == Fam::quantile_scaled) size_n_array *= 1.0/model.data.var_par;
      for(int i = 0; i < model.n(); i++){
        if(size_n_array(i) <= 0){
          size_n_array(i) = model.family.quantile - 1;
        } else {
          size_n_array(i) = model.family.quantile;
        }
        size_n_array(i) *= logitxb(i);
      }
      
      break;
    }
    case Link::probit:
      {
      ArrayXd n_array2(model.n());
      if(model.family.family == Fam::quantile_scaled) size_n_array *= 1.0/model.data.var_par;
      for (int i = 0; i < model.n(); i++) {
        n_array2(i) = glmmr::maths::gaussian_pdf(size_n_array(i)) / (glmmr::maths::gaussian_cdf(size_n_array(i)));
        if(size_n_array(i) <= 0){
          size_n_array(i) = model.family.quantile - 1;
        } else {
          size_n_array(i) = model.family.quantile;
        }
        size_n_array(i) *= n_array2(i);
      }
      break;
      }
    case Link::inverse:
      {
        ArrayXd logitxb = size_n_array.inverse();
        logitxb *= size_n_array.inverse();
        if(model.family.family == Fam::quantile_scaled) size_n_array *= 1.0/model.data.var_par;
        for(int i = 0; i < model.n(); i++){
          if(size_n_array(i) <= 0){
            size_n_array(i) = model.family.quantile - 1;
          } else {
            size_n_array(i) = model.family.quantile;
          }
          size_n_array(i) *= logitxb(i);
        }
        
        break;
      }
    }
    break;
    }
  }
}

template<typename modeltype>
inline VectorXd glmmr::ModelMatrix<modeltype>::log_gradient(const VectorXd &v,
                                                            bool betapars){
  ArrayXd size_n_array(model.n());
  gradient_eta(v,size_n_array);
  ArrayXd size_q_array = ArrayXd::Zero(Q());
  ArrayXd size_p_array = ArrayXd::Zero(P());
  sparse ZLt = model.covariance.ZL_sparse();
  ZLt.transpose();
  
  switch(model.family.family){
  case Fam::poisson: case Fam::bernoulli: case Fam::binomial: case Fam::beta: case Fam::quantile: case Fam::quantile_scaled:
  {
    if(betapars){
    size_p_array =  (model.linear_predictor.X().transpose()*size_n_array.matrix()).array();
  } else {
    size_q_array =  SparseOperators::operator*(ZLt, size_n_array.matrix()).array()-v.array();
  }
  break;
  }
  case Fam::gaussian:
  {
    if(betapars){
    size_p_array += ((1.0/(model.data.var_par))*(model.linear_predictor.X().transpose()*size_n_array.matrix())).array();
  } else {
    size_q_array = SparseOperators::operator*(ZLt, size_n_array.matrix()).array();
    size_q_array *= 1.0/(model.data.var_par);
    size_q_array -= v.array();
  }
  break;
  }
  case Fam::gamma:
  {
    switch(model.family.link){
  case Link::inverse:
  {
    if(betapars){
    size_p_array += (model.linear_predictor.X().transpose()*(size_n_array.matrix()-model.data.y)*model.data.var_par).array();
  } else {
    size_q_array = SparseOperators::operator*(ZLt, size_n_array.matrix()).array();
    size_q_array *= model.data.var_par;
    size_q_array -= v.array();
  }
  break;
  }
  case Link::identity:
  {
    if(betapars){
    size_p_array += (model.linear_predictor.X().transpose()*((model.data.y.array()*size_n_array*size_n_array).matrix() - size_n_array.matrix())*model.data.var_par).array();
  } else {
    size_q_array = SparseOperators::operator*(ZLt, size_n_array.matrix()).array();
    size_q_array *= model.data.var_par;
    size_q_array -= v.array();
  }
  break;
  }
  default:
    //log
  {
    if(betapars){
    size_p_array += (model.linear_predictor.X().transpose()*(model.data.y.array()*size_n_array-1).matrix()*model.data.var_par).array();
  } else {
    size_q_array = SparseOperators::operator*(ZLt, size_n_array.matrix()).array();
    size_q_array *= model.data.var_par;
    size_q_array -= v.array();
  }
  break;
  }
  }
    break;
  }
  }
  return betapars ? size_p_array.matrix() : size_q_array.matrix();
}

// when there are non-linear functions of parameters, we need a correction to the Hessian matrix
template<typename modeltype>
inline MatrixXd glmmr::ModelMatrix<modeltype>::hessian_nonlinear_correction(){
  int iter = re.zu_.cols();
  int n2d = model.linear_predictor.calc.parameter_count*(model.linear_predictor.calc.parameter_count + 1)/2;
  int n = model.n();
  MatrixXd H = MatrixXd::Zero(n2d,n);
  ArrayXXd linpred = re.zu_.array();
  ArrayXd xb = model.xb();
  linpred.colwise() += xb + model.data.offset.array();
#pragma omp parallel for 
  for(int i = 0; i<n ; i++){
    double dfdmu;
    ArrayXd xb_ind = linpred.row(i).transpose();
    switch(model.family.family){
    case Fam::poisson:
    {
      switch(model.family.link){
    case Link::identity:
    {
      xb_ind = xb_ind.inverse();
      xb_ind = model.data.y(i)*xb_ind;
      xb_ind -= 1.0;
      break;
    }
    default:
    {
      xb_ind = xb_ind.exp();
      xb_ind = model.data.y(i) - xb_ind;
      break;
    }
    }
      break;
    }
    case Fam::bernoulli: case Fam::binomial:
    {
      switch(model.family.link){
    case Link::loglink:
    {
      ArrayXd logitxb = 1.0 - xb_ind.exp();
      logitxb = logitxb.inverse();
      logitxb *= xb_ind.exp();
      xb_ind = (model.data.y(i) - model.data.variance(i))*logitxb;
      xb_ind += model.data.y(i);
      break;
    }
    case Link::identity:
    {
      ArrayXd n_array2 = 1.0 - xb_ind;
      n_array2 = n_array2.inverse();
      n_array2 *= (model.data.variance(i) - model.data.y(i));
      xb_ind = xb_ind.inverse();
      xb_ind *= model.data.y(i);
      xb_ind -= n_array2;
      break;
    }
    case Link::probit:
    {
      ArrayXd n_array2(xb_ind.size());
      for (int i = 0; i < xb_ind.size(); i++) {
        xb_ind(i) = glmmr::maths::gaussian_pdf(xb_ind(i)) / (glmmr::maths::gaussian_cdf(xb_ind(i)));
        n_array2(i) = -1.0 * glmmr::maths::gaussian_pdf(xb_ind(i)) / (1 - glmmr::maths::gaussian_cdf(xb_ind(i)));
      }
      xb_ind = model.data.y(i) * xb_ind + (model.data.variance(i) - model.data.y(i)) * n_array2;
      break;
    }
    default:
      //logit
    {
      ArrayXd logitxb = xb_ind.exp();
      logitxb += 1.0;
      logitxb = logitxb.inverse();
      logitxb *= xb_ind.exp();
      xb_ind = model.data.y(i)*(ArrayXd::Constant(model.n(),1) - logitxb) - (model.data.variance(i) - model.data.y(i))*logitxb;
      break;
    }
    }
      break;
    }
    case Fam::gaussian:
    {
      switch(model.family.link){
    case Link::loglink:
    {
      xb_ind = (model.data.y(i) - xb_ind)*model.data.weights(i)*xb_ind.exp();
      xb_ind *= 1.0/(model.data.var_par);
      break;
    }
    default:
    {
      xb_ind = model.data.y(i) - xb_ind;
      xb_ind *= model.data.weights(i)/(model.data.var_par);
      break;
    }
    }
      break;
    }
    case Fam::gamma:
    {
      switch(model.family.link){
    case Link::inverse:
    {
      xb_ind = xb_ind.inverse();
      xb_ind -= model.data.y(i);
      xb_ind = xb_ind - model.data.y(i);
      xb_ind *= 1.0/(model.data.var_par);
      xb_ind -= model.data.y(i);
      xb_ind *= model.data.var_par;
      break;
    }
    case Link::identity:
    {
      xb_ind = xb_ind.inverse();
      xb_ind *= (model.data.y.array()*xb_ind - ArrayXd::Ones(model.n()));
      xb_ind = xb_ind*xb_ind*model.data.y(i) - xb_ind;
      xb_ind *= model.data.var_par;
      break;
    }
    default:
      //log
    {
      xb_ind *= -1.0;
      xb_ind = xb_ind.exp();
      xb_ind *= model.data.y(i)*model.data.y(i);
      xb_ind -= 1.0;
      xb_ind *= model.data.var_par;
      break;
    }
    }
      break;
    }
    case Fam::beta:
    {
      for(int i = 0; i < xb_ind.size(); i++){
      xb_ind(i) = exp(xb_ind(i))/(exp(xb_ind(i))+1);
      xb_ind(i) = (xb_ind(i)/(1+exp(xb_ind(i)))) * model.data.var_par * (log(model.data.y(i)) - log(1- model.data.y(i)) - boost::math::digamma(xb_ind(i)*model.data.var_par) + boost::math::digamma((1-xb_ind(i))*model.data.var_par));
    }
      break;
    }
    case Fam::quantile: case Fam::quantile_scaled:
    {
          switch(model.family.link){
        case Link::identity:
          xb_ind = (model.data.y.array() - xb_ind);
          if(model.family.family == Fam::quantile_scaled) xb_ind *= model.data.var_par;
          for(int i = 0; i < model.n(); i++){
            xb_ind(i) = 0.5*xb_ind(i)/abs(xb_ind(i)) - 0.5*(2*model.family.quantile - 1);
          }
          break;
        case Link::loglink:
        {
          ArrayXd resid = (model.data.y.array() - xb_ind.exp());
          if(model.family.family == Fam::quantile_scaled) resid *= model.data.var_par;
          for(int i = 0; i < model.n(); i++){
            xb_ind(i) = 0.5*resid(i)*exp(xb_ind(i))/abs(resid(i)) - 0.5*(2*model.family.quantile - 1)*exp(xb_ind(i));
          }
          break;
        }
        case Link::logit:
        {
          ArrayXd logitxb = xb_ind.exp();
          logitxb += 1.0;
          logitxb = logitxb.inverse();
          logitxb *= xb_ind.exp();
          ArrayXd resid = (model.data.y.array() - logitxb);
          logitxb *= (1+xb_ind.exp()).inverse();
          if(model.family.family == Fam::quantile_scaled) resid *= model.data.var_par;
          for(int i = 0; i < model.n(); i++){
            xb_ind(i) = 0.5*logitxb(i)*(resid(i)/abs(resid(i)) - (2*model.family.quantile - 1));
          }
          break;
        }
        case Link::probit:
        {
          ArrayXd n_array2(model.n());
          for (int i = 0; i < model.n(); i++) {
            n_array2(i) = glmmr::maths::gaussian_pdf(xb_ind(i)) / (glmmr::maths::gaussian_cdf(xb_ind(i)));
            xb_ind(i) = model.data.y(i) - (glmmr::maths::gaussian_cdf(xb_ind(i)));
            xb_ind(i) = 0.5*n_array2(i)*(xb_ind(i)/abs(xb_ind(i)) - (2*model.family.quantile - 1));
          }
          break;
        }
        case Link::inverse:
        {
          ArrayXd logitxb = xb_ind.inverse();
          ArrayXd resid = (model.data.y.array() - logitxb);
          logitxb *= xb_ind.inverse();
          if(model.family.family == Fam::quantile_scaled) resid *= model.data.var_par;
          for(int i = 0; i < model.n(); i++){
            xb_ind(i) = 0.5*logitxb(i)*(resid(i)/abs(resid(i)) - (2*model.family.quantile - 1));
          }
          break;
        }
        }
      break;
    }
    }
    
    dfdmu = xb_ind.mean();
    
    for(int k = 0; k < iter; k++){
      dblvec out = model.linear_predictor.calc.template calculate<CalcDyDx::BetaSecond>(i,0,0,re.zu_(i,k));
      for(int j = 0; j < n2d; j++){
        H(j,i) += dfdmu*out[model.linear_predictor.calc.parameter_count + 1 + j]/(double)iter;
      }
    }
  }
  
  VectorXd Hmean = H.rowwise().sum();
  MatrixXd H0 = MatrixXd::Zero(model.linear_predictor.calc.parameter_count, model.linear_predictor.calc.parameter_count);
  int index_count = 0;
  for(int j = 0; j < model.linear_predictor.calc.parameter_count; j++){
    for(int k = j; k < model.linear_predictor.calc.parameter_count; k++){
      H0(k,j) = Hmean[index_count];
      if(j != k) H0(j,k) = H0(k,j);
      index_count++;
    }
  }
  
  return H0;
}

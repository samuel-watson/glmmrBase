#ifndef MODELMATRIX_HPP
#define MODELMATRIX_HPP

#include "general.h"
#include "modelbits.hpp"
#include "matrixw.hpp"
#include "randomeffects.hpp"
#include "openmpheader.h"
#include "maths.h"

namespace glmmr {

using namespace Eigen;

class ModelMatrix{
  public:
    glmmr::ModelBits& model;
    glmmr::MatrixW W;
    glmmr::RandomEffects& re;
    ModelMatrix(glmmr::ModelBits& model_, glmmr::RandomEffects& re_): model(model_), W(model_), re(re_) { gen_sigma_blocks();};
    MatrixXd information_matrix();
    MatrixXd Sigma(bool inverse = false);
    MatrixXd observed_information_matrix();
    MatrixXd sandwich_matrix();
    std::vector<MatrixXd> sigma_derivatives();
    MatrixXd information_matrix_theta();
    matrix_matrix kenward_roger();
    MatrixXd linpred();
    vector_matrix b_score();
    vector_matrix re_score();
    matrix_matrix hess_and_grad();
    VectorXd log_gradient(const VectorXd &v,bool beta = false);
    std::vector<glmmr::SigmaBlock> get_sigma_blocks();
    
  private:
    std::vector<glmmr::SigmaBlock> sigma_blocks;
    void gen_sigma_blocks();
    MatrixXd sigma_block(int b, bool inverse = false);
    MatrixXd sigma_builder(int b, bool inverse = false);
    MatrixXd information_matrix_by_block(int b);
};

}

inline std::vector<glmmr::SigmaBlock> glmmr::ModelMatrix::get_sigma_blocks(){
  return sigma_blocks;
}

inline MatrixXd glmmr::ModelMatrix::information_matrix_by_block(int b){
  ArrayXi rows = Map<ArrayXi,Unaligned>(sigma_blocks[b].RowIndexes.data(),sigma_blocks[b].RowIndexes.size());
  MatrixXd X = glmmr::Eigen_ext::submat(model.linear_predictor.X(),rows,ArrayXi::LinSpaced(model.linear_predictor.P(),0,model.linear_predictor.P()-1));
  MatrixXd S = sigma_block(b,true);
  MatrixXd M = X.transpose()*S*X;
  return M;
}

inline MatrixXd glmmr::ModelMatrix::information_matrix(){
  W.update();
  MatrixXd M = MatrixXd::Zero(model.linear_predictor.P(),model.linear_predictor.P());
  for(int i = 0; i< sigma_blocks.size(); i++){
    M += information_matrix_by_block(i);
  }
  return M;
}

inline void glmmr::ModelMatrix::gen_sigma_blocks(){
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
  block_counter = 0;
  intvec idx_matches;
  int n_matches;
  for(i = 0; i < model.n(); i++){
    if(block_counter == 0){
      glmmr::SigmaBlock newblock(block_ids[i]);
      newblock.add_row(0);
      sigma_blocks.push_back(newblock);
    } else {
      for(j = 0; j < block_counter; j++){
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
          sigma_blocks.erase(sigma_blocks.begin()+idx_matches[k]);
        }
      }
    }
    idx_matches.clear();
    block_counter = sigma_blocks.size();
  }
}

inline MatrixXd glmmr::ModelMatrix::Sigma(bool inverse){
  W.update();
  MatrixXd S = sigma_builder(0,inverse);
  return S;
}

inline MatrixXd glmmr::ModelMatrix::sigma_block(int b,
                                                bool inverse){
  if(b >= sigma_blocks.size())Rcpp::stop("Index out of range");
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

inline MatrixXd glmmr::ModelMatrix::sigma_builder(int b,
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

inline MatrixXd glmmr::ModelMatrix::observed_information_matrix(){
  // this works but its too slow doing all the cross partial derivatives
  //MatrixXd XZ(n_,P_+Q_);
  //int iter = zu_.cols();
  //XZ.leftCols(P_) = linpred_.X();
  //XZ.rightCols(Q_) = sparse_to_dense(ZL_,false);
  //MatrixXd result = MatrixXd::Zero(P_+Q_,P_+Q_);
  //MatrixXd I = MatrixXd::Identity(P_+Q_,P_+Q_);
  //dblvec params(P_+Q_);
  //std::copy_n(linpred_.parameters_.begin(),P_,params.begin());
  //for(int i = 0; i < iter; i++){
  //  for(int j = 0; j < Q_; j++){
  //    params[P_+j] = u_(j,i);
  //  }
  //  matrix_matrix hess = vcalc_.jacobian_and_hessian(params,XZ,Map<MatrixXd>(offset_.data(),offset_.size(),1));
  //  result += hess.mat1;
  //}
  //result *= (1.0/iter);
  //return result;
  W.update();
  MatrixXd XtXW = (model.linear_predictor.X()).transpose() * W.W().asDiagonal() * model.linear_predictor.X();
  MatrixXd ZL = sparse_to_dense(re.ZL,false);
  MatrixXd XtWZL = (model.linear_predictor.X()).transpose() * W.W().asDiagonal() * ZL;
  MatrixXd ZLWLZ = ZL.transpose() * W.W().asDiagonal() * ZL;
  ZLWLZ += MatrixXd::Identity(model.covariance.Q(),model.covariance.Q());
  MatrixXd infomat(model.linear_predictor.P()+model.covariance.Q(),model.linear_predictor.P()+model.covariance.Q());
  infomat.topLeftCorner(model.linear_predictor.P(),model.linear_predictor.P()) = XtXW;
  infomat.topRightCorner(model.linear_predictor.P(),model.covariance.Q()) = XtWZL;
  infomat.bottomLeftCorner(model.covariance.Q(),model.linear_predictor.P()) = XtWZL.transpose();
  infomat.bottomRightCorner(model.covariance.Q(),model.covariance.Q()) = ZLWLZ;
  return infomat;
}

inline MatrixXd glmmr::ModelMatrix::sandwich_matrix(){
  MatrixXd infomat = observed_information_matrix();
  infomat = infomat.llt().solve(MatrixXd::Identity(model.linear_predictor.P()+model.covariance.Q(),model.linear_predictor.P()+model.covariance.Q()));
  infomat.conservativeResize(model.linear_predictor.P(),model.linear_predictor.P());
  MatrixXd zuOffset = re.Zu();
  zuOffset.colwise() += model.data.offset;
  MatrixXd J = model.calc.jacobian(model.linear_predictor.parameters,model.linear_predictor.Xdata,zuOffset);
  MatrixXd sandwich = infomat * (J * J.transpose()) * infomat;
  return sandwich;
}

inline std::vector<MatrixXd> glmmr::ModelMatrix::sigma_derivatives(){
  std::vector<MatrixXd> derivs;
  model.covariance.derivatives(derivs,2);
  return derivs;
}

inline MatrixXd glmmr::ModelMatrix::information_matrix_theta(){
  if(model.family.family=="gamma" || model.family.family=="beta")Rcpp::stop("Not currently supported for gamma or beta families");
  int R = model.covariance.npar();
  int Rmod = model.family.family=="gaussian" ? R+1 : R;
  MatrixXd M_theta = MatrixXd::Zero(Rmod,Rmod);
  std::vector<MatrixXd> A_matrix;
  MatrixXd SigmaInv = Sigma(true);
  MatrixXd Z = model.covariance.Z();
  std::vector<MatrixXd> derivs;
  model.covariance.derivatives(derivs,1);
  for(int i = 0; i < R; i++){
    A_matrix.push_back(SigmaInv*Z*derivs[1+i]*Z.transpose());
  }
  if(model.family.family=="gaussian"){
    A_matrix.push_back(2*model.data.variance.matrix().asDiagonal()*SigmaInv);
  }
  for(int i = 0; i < Rmod; i++){
    for(int j = i; j < Rmod; j++){
      M_theta(i,j) = 0.5 * (A_matrix[i]*A_matrix[j]).trace();
      if(i!=j)M_theta(j,i)=M_theta(i,j);
    }
  }
  return M_theta;
}

inline matrix_matrix glmmr::ModelMatrix::kenward_roger(){
  if(model.family.family=="gamma" || model.family.family=="beta")Rcpp::stop("Not currently supported for gamma or beta families");
  int R = model.covariance.npar();
  int Rmod = model.family.family=="gaussian" ? R+1 : R;
  MatrixXd M_theta = MatrixXd::Zero(Rmod,Rmod);
  std::vector<MatrixXd> A_matrix;
  MatrixXd SigmaInv = Sigma(true);
  MatrixXd Z = model.covariance.Z();
  MatrixXd M = information_matrix();
  M = M.llt().solve(MatrixXd::Identity(model.linear_predictor.P(),model.linear_predictor.P()));
  MatrixXd X = model.linear_predictor.X();
  MatrixXd SigX = SigmaInv*X;
  MatrixXd middle = MatrixXd::Identity(model.n(),model.n()) - X*M*SigX.transpose();
  
  std::vector<MatrixXd> derivs;
  model.covariance.derivatives(derivs,2);
  for(int i = 0; i < R; i++){
    A_matrix.push_back(Z*derivs[1+i]*Z.transpose());
  }
  if(model.family.family=="gaussian"){
    A_matrix.push_back(2*model.data.variance.matrix().asDiagonal()*MatrixXd::Identity(model.n(),model.n()));
  }
  //possible parallelisation?
  for(int i = 0; i < Rmod; i++){
    for(int j = i; j < Rmod; j++){
      M_theta(i,j) = (SigmaInv*A_matrix[i]*SigmaInv*A_matrix[j]).trace();
      M_theta(i,j) -= (M*SigX.transpose()*A_matrix[i]*SigmaInv*(middle+MatrixXd::Identity(model.n(),model.n()))*A_matrix[j]*SigX).trace();
      M_theta(i,j) *= 0.5;
      if(i!=j)M_theta(j,i)=M_theta(i,j);
    }
  }
  
  M_theta = M_theta.llt().solve(MatrixXd::Identity(Rmod,Rmod));
  MatrixXd meat = MatrixXd::Zero(model.linear_predictor.P(),model.linear_predictor.P());
  for(int i = 0; i < Rmod; i++){
    for(int j = 0; j < Rmod; j++){
      int scnd_idx = i <= j ? i + j*(R-1) - j*(j-1)/2 : j + i*(R-1) - i*(i-1)/2;
      meat += M_theta(i,j)*SigX.transpose()*A_matrix[i]*SigmaInv*middle*A_matrix[j]*SigX;
      if(i < R && j < R){
        meat -= M_theta(i,j)*0.25*SigX.transpose()*Z*derivs[R+1+scnd_idx]*Z.transpose()*SigX;
      }
      if(i==R && j==R){
        meat -= M_theta(i,j)*0.5*SigX.transpose()*SigX;
      }
    }
  }
  
  M += 2*M*meat*M;
  matrix_matrix out(model.linear_predictor.P(),model.linear_predictor.P(),Rmod,Rmod);
  out.mat1 = M;
  out.mat2 = M_theta;
  return out;
}

inline MatrixXd glmmr::ModelMatrix::linpred(){
  return (re.zu_.colwise()+(model.linear_predictor.xb()+model.data.offset));
}

inline vector_matrix glmmr::ModelMatrix::b_score(){
  MatrixXd zuOffset = re.Zu();
  zuOffset.colwise() += model.data.offset;
  matrix_matrix hess = model.calc.jacobian_and_hessian(model.linear_predictor.parameters,model.linear_predictor.Xdata,zuOffset);
  vector_matrix out(hess.mat1.rows());
  out.mat = hess.mat1;
  out.mat *= -1.0;
  out.vec = hess.mat2.rowwise().sum();
  return out;
}

inline matrix_matrix glmmr::ModelMatrix::hess_and_grad(){
  MatrixXd zuOffset = re.Zu();
  zuOffset.colwise() += model.data.offset;
  matrix_matrix hess = model.calc.jacobian_and_hessian(model.linear_predictor.parameters,model.linear_predictor.Xdata,zuOffset);
  return hess;
}

inline vector_matrix glmmr::ModelMatrix::re_score(){
  VectorXd xbOffset = model.linear_predictor.xb() + model.data.offset;
  matrix_matrix hess = model.vcalc.jacobian_and_hessian(dblvec(re.u(false).col(0).data(),re.u(false).col(0).data()+re.u(false).rows()),sparse_to_dense(re.ZL,false),Map<MatrixXd>(xbOffset.data(),xbOffset.size(),1));
  vector_matrix out(model.covariance.Q());
  hess.mat1 *= -1.0;
  out.mat = hess.mat1 + MatrixXd::Identity(model.covariance.Q(),model.covariance.Q());
  out.vec = hess.mat2.rowwise().sum();
  out.vec -= re.u(false).col(0);
  return out;
}

inline VectorXd glmmr::ModelMatrix::log_gradient(const VectorXd &v,
                                                bool beta){
  ArrayXd size_n_array = model.xb();
  ArrayXd size_q_array = ArrayXd::Zero(model.covariance.Q());
  ArrayXd size_p_array = ArrayXd::Zero(model.linear_predictor.P());
  sparse ZLt = re.ZL;
  ZLt.transpose();
  size_n_array += (re.ZL*v).array();
  
  if(beta){
    VectorXd zuOffset = re.ZL*v;
    zuOffset += model.data.offset;
    MatrixXd J = model.calc.jacobian(model.linear_predictor.parameters,model.linear_predictor.Xdata,zuOffset);
    size_p_array = J.transpose().rowwise().sum().array();
  } else {
    switch (model.family.flink){
    case 1:
  {
    size_n_array = size_n_array.exp();
    if(!beta){
      size_n_array = model.data.y.array() - size_n_array;
      size_q_array = ZLt*size_n_array -v.array() ;
    } else {
      size_p_array += (model.linear_predictor.X().transpose()*(model.data.y-size_n_array.matrix())).array();
    }
    break;
  }
    case 2:
  {
    size_n_array = size_n_array.inverse();
    size_n_array = model.data.y.array()*size_n_array;
    size_n_array -= ArrayXd::Ones(model.n());
    if(beta){
      size_p_array +=  (model.linear_predictor.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 3: case 13:
  {
    ArrayXd logitxb = model.xb().array().exp();
    logitxb += 1;
    logitxb = logitxb.inverse();
    logitxb *= model.xb().array().exp();
    size_n_array = model.data.y.array()*(ArrayXd::Constant(model.n(),1) - logitxb) - (model.data.variance - model.data.y.array())*logitxb;
    if(beta){
      size_p_array +=  (model.linear_predictor.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 4: case 14:
  {
    ArrayXd logitxb = model.xb().array().exp();
    logitxb += 1;
    logitxb = logitxb.inverse();
    logitxb *= model.xb().array().exp();
    size_n_array = (model.data.variance - model.data.y.array())*logitxb;
    size_n_array += model.data.y.array();
    if(beta){
      size_p_array +=  (model.linear_predictor.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 5: case 15:
  {
    size_n_array = size_n_array.inverse();
    size_n_array *= model.data.y.array();
    ArrayXd n_array2 = ArrayXd::Constant(model.n(),1.0) - model.xb().array();
    n_array2 =n_array2.inverse();
    n_array2 *= (model.data.variance - model.data.y.array());
    size_n_array -= n_array2;
    if(beta){
      size_p_array +=  (model.linear_predictor.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 6: case 16:
  {
    ArrayXd n_array2(model.n());
#pragma omp parallel for 
    for(int i = 0; i < model.n(); i++){
      size_n_array(i) = (double)R::dnorm(size_n_array(i),0,1,false)/((double)R::pnorm(size_n_array(i),0,1,true,false));
      n_array2(i) = -1.0*(double)R::dnorm(size_n_array(i),0,1,false)/(1-(double)R::pnorm(size_n_array(i),0,1,true,false));
    }
    size_n_array = model.data.y.array()*size_n_array + (model.data.variance - model.data.y.array())*n_array2;
    if(beta){
      size_p_array +=  (model.linear_predictor.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array =  ZLt*size_n_array-v.array();
    }
    break;
  }
    case 7:
  {
    if(beta){
    size_n_array -= model.data.y.array();
    size_n_array *= -1;
    size_n_array *= model.data.weights;
    size_p_array += ((1.0/(model.data.var_par))*(model.linear_predictor.X().transpose()*size_n_array.matrix())).array();
  } else {
    size_n_array = model.data.y.array() - size_n_array;
    size_n_array *= model.data.weights;
    size_q_array = (ZLt*size_n_array)-v.array();
    size_q_array *= 1.0/(model.data.var_par);
  }
  break;
  }
    case 8:
  {
    if(beta){
    size_n_array -= model.data.y.array();
    size_n_array *= -1;
    size_n_array *= model.data.weights;
    size_p_array += ((1.0/(model.data.var_par))*(model.linear_predictor.X().transpose()*(model.data.y - size_n_array.matrix()))).array();
  } else {
    size_n_array = model.data.y.array() - size_n_array;
    size_n_array *= model.data.weights;
    size_q_array = ZLt*size_n_array-v.array();
    size_q_array *= 1.0/(model.data.var_par);
  }
  break;
  }
    case 9:
  {
    size_n_array *= -1.0;
    size_n_array = size_n_array.exp();
    if(beta){
      size_p_array += (model.linear_predictor.X().transpose()*(model.data.y.array()*size_n_array-1).matrix()*model.data.var_par).array();
    } else {
      size_n_array *= model.data.y.array();
      size_q_array = ZLt*size_n_array-v.array();
      size_q_array *= model.data.var_par;
    }
    break;
  }
    case 10:
  {
    size_n_array = size_n_array.inverse();
    if(beta){
      size_p_array += (model.linear_predictor.X().transpose()*(size_n_array.matrix()-model.data.y)*model.data.var_par).array();
    } else {
      size_n_array -= model.data.y.array();
      size_q_array = ZLt*size_n_array-v.array();
      size_q_array *= model.data.var_par;
    }
    break;
  }
    case 11:
  {
    size_n_array = size_n_array.inverse();
    if(beta){
      size_p_array += (model.linear_predictor.X().transpose()*((model.data.y.array()*size_n_array*size_n_array).matrix() - size_n_array.matrix())*model.data.var_par).array();
    } else {
      size_n_array *= (model.data.y.array()*size_n_array - ArrayXd::Ones(model.n()));
      size_q_array = ZLt*size_n_array-v.array();
      size_q_array *= model.data.var_par;
    }
    break;
  }
    case 12:
  {
#pragma omp parallel for 
    for(int i = 0; i < model.n(); i++){
      size_n_array(i) = exp(size_n_array(i))/(exp(size_n_array(i))+1);
      size_n_array(i) = (size_n_array(i)/(1+exp(size_n_array(i)))) * model.data.var_par * (log(model.data.y(i)) - log(1- model.data.y(i)) - boost::math::digamma(size_n_array(i)*model.data.var_par) + boost::math::digamma((1-size_n_array(i))*model.data.var_par));
    }
    if(beta){
      size_p_array += (model.linear_predictor.X().transpose()*size_n_array.matrix()).array();
    } else {
      size_q_array = ZLt*size_n_array-v.array();
    }
    break;
  }
      
    }
  }
  // we can use autodiff here, but the above method is faster
  // else {
  //   VectorXd xbOffset_ = linpred_.xb() + offset_;
  //   MatrixXd J = vcalc_.jacobian(dblvec(v.data(),v.data()+v.size()),
  //                                               sparse_to_dense(ZL_,false),
  //                                               xbOffset_);
  //   size_q_array = (J.transpose().rowwise().sum() - v).array();
  // }
  return beta ? size_p_array.matrix() : size_q_array.matrix();
}

#endif
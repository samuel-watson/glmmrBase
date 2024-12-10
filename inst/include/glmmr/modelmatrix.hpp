#pragma once

#include "general.h"
#include "modelbits.hpp"
#include "matrixw.hpp"
#include "randomeffects.hpp"
#include "openmpheader.h"
#include "maths.h"
#include "matrixfield.h"

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


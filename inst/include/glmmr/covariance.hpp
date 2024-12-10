#pragma once

#define _USE_MATH_DEFINES

#include "openmpheader.h"
#include "general.h"
#include "maths.h"
#include "algo.h"
#include "interpreter.h"
#include "formula.hpp"
#include "sparse.h"
#include "calculator.hpp"
#include "linearpredictor.hpp"

using namespace Eigen;

namespace glmmr {

class Covariance {
public:
  // objects
  glmmr::Formula  form_;
  const ArrayXXd  data_;
  const strvec    colnames_;
  dblvec          parameters_;
  dblvec          other_pars_;

  // constructors
  Covariance(const str& formula,const ArrayXXd &data,const strvec& colnames);
  Covariance(const glmmr::Formula& form,const ArrayXXd &data,const strvec& colnames);
  Covariance(const str& formula,const ArrayXXd &data,const strvec& colnames,const dblvec& parameters);
  Covariance(const glmmr::Formula& form,const ArrayXXd &data,const strvec& colnames,const dblvec& parameters);
  Covariance(const str& formula,const ArrayXXd &data,const strvec& colnames,const ArrayXd& parameters);
  Covariance(const glmmr::Formula& form,const ArrayXXd &data,const strvec& colnames,const ArrayXd& parameters);
  Covariance(const glmmr::Covariance& cov);

  // functions
  virtual void      update_parameters(const dblvec& parameters);
  virtual void      update_parameters_extern(const dblvec& parameters);
  virtual void      update_parameters(const ArrayXd& parameters);
  virtual int       parse();
  double            get_val(int b, int i, int j) const;
  virtual MatrixXd  Z();
  virtual MatrixXd  D(bool chol = false, bool upper = false);
  virtual VectorXd  sim_re();
  virtual double    log_likelihood(const VectorXd &u);
  virtual double    log_determinant();
  virtual int       npar() const;
  virtual int       B() const;
  virtual int       Q() const;
  virtual int       max_block_dim() const;
  virtual int       block_dim(int b) const;
  virtual void      make_sparse();
  virtual MatrixXd  ZL();
  virtual MatrixXd  LZWZL(const VectorXd& w);
  virtual MatrixXd  ZLu(const MatrixXd& u);
  virtual MatrixXd  Lu(const MatrixXd& u);
  virtual void      set_sparse(bool sparse, bool amd = true);
  bool              any_group_re() const;
  intvec            parameter_fn_index() const;
  virtual intvec    re_count() const;
  virtual sparse    ZL_sparse();
  virtual sparse    Z_sparse();
  strvec            parameter_names();
  virtual void      derivatives(std::vector<MatrixXd>& derivs,int order = 1);
  virtual VectorXd  log_gradient(const MatrixXd &umat, double& logl);
  void              linear_predictor_ptr(glmmr::LinearPredictor* ptr);
 
protected:
  // data
  std::vector<glmmr::calculator>      calc_;
  std::vector<std::vector<CovFunc> >  fn_;
  glmmr::LinearPredictor*             linpred_ptr = nullptr;
  intvec                              re_fn_par_link_;
  intvec                              re_count_;
  intvec                              re_order_;
  intvec                              block_size;
  intvec                              block_nvar;
  intvec3d                            re_cols_data_;
  dblvec3d                            re_temp_data_;
  intvec                              z_;
  int                                 Q_;
  sparse                              matZ;
  int                                 n_;
  int                                 B_;
  int                                 npars_;
  MatrixXd                            dmat_matrix;
  VectorXd                            zquad;
  bool                                isSparse = true;
  sparse                              matL;
  SparseChol                          spchol;
  
  // functions
  void                            update_parameters_in_calculators();
  MatrixXd                        get_block(int b);
  MatrixXd                        get_chol_block(int b,bool upper = false);
  MatrixXd                        D_builder(int b,bool chol = false,bool upper = false);
  void                            update_ax();
  void                            L_constructor();
  void                            Z_constructor();
  void                            Z_updater();
  MatrixXd                        D_sparse_builder(bool chol = false, bool upper = false);
  // logical flags
  bool                            sparse_initialised = false;
  bool                            use_amd_permute = true;
public:
  bool                            z_requires_update = false;
protected:
  std::vector<ZNonZero>           z_nonzero;
};

}


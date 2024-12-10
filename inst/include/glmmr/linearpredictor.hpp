#pragma once

#include "general.h"
#include "interpreter.h"
#include "calculator.hpp"
#include "formula.hpp"

namespace glmmr {

class LinearPredictor {
public:
  // data
  dblvec              parameters;
  glmmr::calculator   calc;
  glmmr::Formula&     form;
  // constructors
  LinearPredictor(glmmr::Formula& form_,const Eigen::ArrayXXd &data_,const strvec& colnames_);
  LinearPredictor(glmmr::Formula& form_,const Eigen::ArrayXXd &data_,const strvec& colnames_,const dblvec& parameters_);
  LinearPredictor(glmmr::Formula& form_,const Eigen::ArrayXXd &data_,const strvec& colnames_,const Eigen::ArrayXd& parameters_);
  LinearPredictor(const glmmr::LinearPredictor& linpred);
  // functions
  virtual void      update_parameters(const dblvec& parameters_);
  virtual void      update_parameters(const Eigen::ArrayXd& parameters_);
  int               P() const;
  int               n() const;
  strvec            colnames() const;
  virtual VectorXd  xb();
  virtual MatrixXd  X();
  virtual double    X(const int i, const int j) const;
  strvec            parameter_names() const;
  VectorXd          parameter_vector();
  bool              any_nonlinear() const;
  virtual VectorXd  predict_xb(const ArrayXXd& newdata_,const ArrayXd& newoffset_);
protected:
  // data
  strvec    colnames_vec;
  int       P_;
  int       n_;
  intvec    x_cols;
  MatrixXd  X_;
  bool      x_set = false;
};
}


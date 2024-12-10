#pragma once

#include "general.h"
#include "modelbits.hpp"
#include "randomeffects.hpp"
#include "modelmatrix.hpp"
#include "modelmcmc.hpp"
#include "modeloptim.hpp"

namespace glmmr {

using namespace Eigen;

template<class>
struct check_type : std::false_type {};

template<>
struct check_type<glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor> > : std::true_type {};

template<typename modeltype>
class Model {
public:
  // model objects
  modeltype                       model;
  glmmr::RandomEffects<modeltype> re;
  glmmr::ModelMatrix<modeltype>   matrix;
  glmmr::ModelOptim<modeltype>    optim;
  glmmr::ModelMCMC<modeltype>     mcmc;
  // constructor
  Model(const std::string& formula_,const ArrayXXd& data_,const strvec& colnames_,std::string family_,std::string link_);
  //functions
  virtual void    set_offset(const VectorXd& offset_);
  virtual void    set_weights(const ArrayXd& weights_);
  virtual void    set_y(const VectorXd& y_);
  virtual void    update_beta(const dblvec &beta_);
  virtual void    update_theta(const dblvec &theta_);
  virtual void    update_u(const MatrixXd &u_, bool append = false);
  virtual void    reset_u(); // just resets the random effects samples to zero
  virtual void    set_trace(int trace_);
  virtual dblpair marginal(const MarginType type,
                             const std::string& x,
                             const strvec& at,
                             const strvec& atmeans,
                             const strvec& average,
                             const RandomEffectMargin re_type,
                             const SE se_type,
                             const IM im_type,
                             const dblpair& xvals,
                             const dblvec& atvals,
                             const dblvec& atrevals);
                                             
};

}



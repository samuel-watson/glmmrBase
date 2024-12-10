#pragma once

#include "general.h"
#include "modelbits.hpp"
#include "modelmatrix.hpp"
#include "randomeffects.hpp"
#include "openmpheader.h"
#include "maths.h"

namespace glmmr {

using namespace Eigen;

template<typename modeltype>
class ModelMCMC{
public:
  modeltype&                        model;
  glmmr::ModelMatrix<modeltype>&    matrix;
  glmmr::RandomEffects<modeltype>&  re;
  bool                              verbose = true;
  int                               trace = 1;
  
  ModelMCMC(modeltype& model_,glmmr::ModelMatrix<modeltype>& matrix_,glmmr::RandomEffects<modeltype>& re_);
  
  double    log_prob(const VectorXd &v);
  void      mcmc_sample(int warmup_,int samples_,int adapt_ = 100);
  void      mcmc_set_lambda(double lambda);
  void      mcmc_set_max_steps(int max_steps);
  void      mcmc_set_refresh(int refresh);
  void      mcmc_set_target_accept(double target);
  
protected:
  VectorXd      u0;
  VectorXd      up;
  VectorXd      r;
  VectorXd      grad;
  int           refresh=500;
  double        lambda=0.01;
  int           max_steps = 100;
  int           accept;
  double        e = 0.001;
  double        ebar = 1.0;
  double        H= 0;
  int           steps;
  double        target_accept = 0.9;
  VectorXd      new_proposal(const VectorXd& u0_, bool adapt,  int iter, double rand);
  void          sample(int warmup_,int nsamp_,int adapt_ = 100);
  
};

}




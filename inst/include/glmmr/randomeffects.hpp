#pragma once

#include "general.h"
#include "covariance.hpp"
#include "modelbits.hpp"
#include "maths.h"
#include "sparse.h"
#include "calculator.hpp"

namespace glmmr {

using namespace Eigen;

enum class RandomEffectMargin {
  AtEstimated = 0,
    At = 1,
    AtZero = 2,
    Average = 3
};


template<typename modeltype>
class RandomEffects{
public:
  MatrixXd    u_;
  MatrixXd    scaled_u_;
  MatrixXd    zu_;
  modeltype&  model;
  int         mcmc_block_size = 1; // for saem
  
  RandomEffects(modeltype& model_) : 
    u_(MatrixXd::Zero(model_.covariance.Q(),1)),
    scaled_u_(MatrixXd::Zero(model_.covariance.Q(),1)),
    zu_(model_.n(),1), model(model_) {};
  
  RandomEffects(modeltype& model_, int n, int Q) : 
    u_(MatrixXd::Zero(Q,1)),
    scaled_u_(MatrixXd::Zero(Q,1)),
    zu_(n,1), model(model_) {};
  
  RandomEffects(const glmmr::RandomEffects<modeltype>& re) : u_(re.u_), scaled_u_(re.scaled_u_), zu_(re.zu_), model(re.model) {};
  
  MatrixXd      Zu(){return zu_;};
  MatrixXd      u(bool scaled = true);
  VectorMatrix  predict_re(const ArrayXXd& newdata_,const ArrayXd& newoffset_);
  
};

}



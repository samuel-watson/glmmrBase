#pragma once

#include "general.h"
#include "modelbits.hpp"
#include "openmpheader.h"
#include "maths.h"

namespace glmmr {

using namespace Eigen;

template<typename modeltype>
class MatrixW{
public:
  bool        attenuated = false;
  VectorXd    W_ = VectorXd::Constant(1,1.0);
  modeltype&  model;
  MatrixW(modeltype& model_): model(model_) { update(); };
  VectorXd    W() const;
  void        update();
};
}



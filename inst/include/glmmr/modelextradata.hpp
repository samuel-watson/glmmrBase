#ifndef MODELEXTRADATA_HPP
#define MODELEXTRADATA_HPP

#include "general.h"

namespace glmmr {

using namespace Eigen;

class ModelExtraData{
public:
  VectorXd offset = VectorXd::Zero(1);
  ArrayXd weights = ArrayXd::Constant(1,1.0);
  ArrayXd variance = ArrayXd::Constant(1,1.0);
  double var_par = 1;
  VectorXd y = VectorXd::Constant(1,1.0);
  ModelExtraData(){};
  ModelExtraData(int n){
    offset.conservativeResize(n);
    offset.setConstant(0.0);
    weights.conservativeResize(n);
    weights.setConstant(1.0);
    variance.conservativeResize(n);
    variance.setConstant(1.0);
  }
  void set_offset(const VectorXd& offset_){
    offset.conservativeResize(offset_.size());
    offset = offset_;
  }
  void set_weights(const ArrayXd& weights_){
    weights.conservativeResize(weights_.size());
    weights = weights_;
  }
  void set_weights(int n){
    weights.conservativeResize(n);
    weights.setConstant(1.0);
  }
  void set_variance(const ArrayXd& variance_){
    variance.conservativeResize(variance_.size());
    variance = variance_;
  }
  void set_var_par(double var_par_){
    var_par = var_par_;
  }
  void update_y(const VectorXd& y_){
    y.conservativeResize(y_.size());
    y = y_;
  }
};


}

#endif

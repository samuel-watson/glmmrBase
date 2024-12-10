#pragma once

#include "covariance.hpp"
#include "hsgpcovariance.hpp"
#include "nngpcovariance.hpp"
#include "family.hpp"
#include "modelextradata.hpp"

namespace glmmr {

using namespace Eigen;

template<typename cov, typename linpred>
class ModelBits{
public:
  glmmr::Formula        formula;
  linpred               linear_predictor;
  cov                   covariance;
  glmmr::ModelExtraData data;
  glmmr::Family         family;
  bool                  weighted = false;
  int                   trace = 0;
    
  ModelBits(const std::string& formula_,const ArrayXXd& data_,const strvec& colnames_,std::string family_,std::string link_);
  //functions
  virtual int       n() const {return linear_predictor.n();};
  virtual ArrayXd   xb() {return linear_predictor.xb().array() + data.offset.array();};
  virtual void      make_covariance_sparse(bool amd = true);
  virtual void      make_covariance_dense();
};

}



typedef glmmr::Covariance covariance;
typedef glmmr::nngpCovariance nngp;
typedef glmmr::hsgpCovariance hsgp;
typedef glmmr::LinearPredictor xb;
typedef glmmr::ModelBits<covariance, xb> bits;
typedef glmmr::ModelBits<nngp, xb> bits_nngp;
typedef glmmr::ModelBits<hsgp, xb> bits_hsgp;
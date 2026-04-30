#pragma once

#include "covariance.hpp"
#include "hsgpcovariance.hpp"
#include "nngpcovariance.hpp"
#include "ar1covariance.hpp"
#include "spdecovariance.hpp"
#include "linearpredictor.hpp"
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
    
    ModelBits(const ModelBits&) = default;
    ModelBits(ModelBits&&) = default;
    ModelBits& operator=(const ModelBits&) = default;
    ModelBits& operator=(ModelBits&&) = default;
    
    // Single constructor for all non-ar1 types
    template <typename C = cov, typename = std::enable_if_t<!std::is_same_v<C, glmmr::ar1Covariance>>>
    ModelBits(const std::string& formula_, const ArrayXXd& data_, const strvec& colnames_,
              std::string family_, std::string link_) 
      : formula(formula_), 
        linear_predictor(formula, data_, colnames_),
        covariance(formula, data_, colnames_),
        data(data_.rows()),
        family(family_, link_) 
    {
      if constexpr (std::is_same_v<cov, glmmr::Covariance>) {
        covariance.linear_predictor_ptr(&linear_predictor);
      }
    }
    
    // ar1Covariance constructor (different signature, so no conflict)
    template <typename C = cov, typename = std::enable_if_t<std::is_same_v<C, glmmr::ar1Covariance>>>
    ModelBits(const std::string& formula_, const ArrayXXd& data_, const ArrayXXd& cov_data_,
              const strvec& colnames_, const strvec& colnames_cov_, std::string family_, 
              std::string link_, const int T_)
      : formula(formula_), 
        linear_predictor(formula, data_, colnames_),
        covariance(formula_, cov_data_, colnames_cov_, T_),
        data(data_.rows()),
        family(family_, link_) 
    {}
    
  //functions
  virtual int       n() const {return linear_predictor.n();};
  virtual ArrayXd   xb() {return linear_predictor.xb().array() + data.offset.array();};
  virtual void      make_covariance_sparse();
  virtual void      make_covariance_dense();
};

}

template<typename cov, typename linpred>
inline void glmmr::ModelBits<cov, linpred>::make_covariance_sparse(){
  covariance.set_sparse(true);
}

template<typename cov, typename linpred>
inline void glmmr::ModelBits<cov, linpred>::make_covariance_dense(){
  covariance.set_sparse(false);
}

typedef glmmr::Covariance covariance;
typedef glmmr::ar1Covariance ar1covariance;
typedef glmmr::nngpCovariance nngp;
typedef glmmr::hsgpCovariance hsgp;
typedef glmmr::spdeCovariance spde;
typedef glmmr::LinearPredictor xb;
typedef glmmr::ModelBits<covariance, xb> bits;
typedef glmmr::ModelBits<ar1covariance, xb> bits_ar1;
typedef glmmr::ModelBits<nngp, xb> bits_nngp;
typedef glmmr::ModelBits<spde, xb> bits_spde;
typedef glmmr::ModelBits<hsgp, xb> bits_hsgp;
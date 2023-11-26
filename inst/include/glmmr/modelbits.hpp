#pragma once

#include "general.h"
#include "covariance.hpp"
#include "hsgpcovariance.hpp"
#include "nngpcovariance.hpp"
#include "linearpredictor.hpp"
#include "family.hpp"
#include "modelextradata.hpp"
#include "calculator.hpp"
#include "formula.hpp"

namespace glmmr {

using namespace Eigen;

template<typename linpred>
class GLM {
public:
  glmmr::Formula formula;
  linpred linear_predictor;
  glmmr::ModelExtraData data;
  glmmr::Family family;
  glmmr::calculator calc;
  bool weighted = false;
  GLM(const std::string& formula_,const ArrayXXd& data_,const strvec& colnames_,std::string family_,std::string link_);
  virtual int n() const {return linear_predictor.n();};
  virtual ArrayXd xb() {return linear_predictor.xb().array() + data.offset.array();};
protected :
  void setup_calculator();
};

template<typename cov, typename linpred>
class GLMMixed : public GLM<linpred> {
public:
  cov covariance;
  glmmr::calculator vcalc;
  GLMMixed(const std::string& formula_,const ArrayXXd& data_,const strvec& colnames_,std::string family_,std::string link_);
  void make_covariance_sparse();
  void make_covariance_dense();
private:
  void setup_calculator_v() ovveride;
};

}

template<typename linpred>
inline glmmr::GLM<linpred>::GLM(const std::string& formula_,
                                const ArrayXXd& data_,
                                const strvec& colnames_,
                                std::string family_, 
                                std::string link_) : 
  formula(formula_), 
  linear_predictor(formula,data_,colnames_),
  data(data_.rows()),
  family(family_,link_) { setup_calculator(); };

template<typename cov, typename linpred>
inline glmmr::GLMMixed<cov, linpred>::GLMMixed(const std::string& formula_,
                                                 const ArrayXXd& data_,
                                                 const strvec& colnames_,
                                                 std::string family_, 
                                                 std::string link_) : 
  GLM(formula_, data_, colnames_, family_, link_), covariance(formula,data_,colnames_) { setup_calculator_v(); };

// template<typename cov, typename linpred>
// inline glmmr::ModelBits<cov, linpred>::ModelBits(const std::string& formula_,
//           const ArrayXXd& data_,
//           const strvec& colnames_,
//           std::string family_, 
//           std::string link_) : 
//   formula(formula_), 
//   covariance(formula,data_,colnames_),
//   linear_predictor(formula,data_,colnames_),
//   data(data_.rows()),
//   family(family_,link_) { setup_calculator(); };

template<typename linpred>
inline void glmmr::GLM<linpred>::setup_calculator(){
  dblvec yvec(n(),0.0);
  calc = linear_predictor.calc;
  glmmr::linear_predictor_to_link(calc,family.link);
  glmmr::link_to_likelihood(calc,family.family);
  calc.y = yvec;
  calc.variance.conservativeResize(yvec.size());
  calc.variance = data.variance;
}

template<typename cov, typename linpred>
inline void glmmr::GLMMixed<cov, linpred>::setup_calculator_v(){
  dblvec yvec(n(),0.0);
  vcalc = linear_predictor.calc;
  glmmr::re_linear_predictor(vcalc,covariance.Q());
  glmmr::linear_predictor_to_link(vcalc,family.link);
  glmmr::link_to_likelihood(vcalc,family.family);
  vcalc.y = yvec;
  vcalc.variance.conservativeResize(yvec.size());
  vcalc.variance = data.variance;
}

// template<typename cov, typename linpred>
// inline void glmmr::ModelBits<cov, linpred>::setup_calculator(){
//   dblvec yvec(n(),0.0);
//   calc = linear_predictor.calc;
//   glmmr::linear_predictor_to_link(calc,family.link);
//   glmmr::link_to_likelihood(calc,family.family);
//   calc.y = yvec;
//   calc.variance.conservativeResize(yvec.size());
//   calc.variance = data.variance;
//   vcalc = linear_predictor.calc;
//   glmmr::re_linear_predictor(vcalc,covariance.Q());
//   glmmr::linear_predictor_to_link(vcalc,family.link);
//   glmmr::link_to_likelihood(vcalc,family.family);
//   vcalc.y = yvec;
//   vcalc.variance.conservativeResize(yvec.size());
//   vcalc.variance = data.variance;
// }

template<>
inline void glmmr::GLMMixed<glmmr::hsgpCovariance, glmmr::LinearPredictor>::setup_calculator(){
  int i = 0;
  (void)i;
}

template<typename cov, typename linpred>
void glmmr::GLMMixed<cov, linpred>::make_covariance_sparse(){
  covariance.set_sparse(true);
}

template<typename cov, typename linpred>
void glmmr::GLMMixed<cov, linpred>::make_covariance_dense(){
  covariance.set_sparse(false);
}

typedef glmmr::Covariance covariance;
typedef glmmr::nngpCovariance nngp;
typedef glmmr::hsgpCovariance hsgp;
typedef glmmr::LinearPredictor xb;
typedef glmmr::GLM<xb> glm;
typedef glmmr::GLMMixed<covariance, xb> Mixed;
typedef glmmr::GLMMixed<nngp, xb> MixedNNGP;
typedef glmmr::GLMMixed<hsgp, xb> MixedHSGP;

template<class>
struct check_mixed_type : std::false_type {};
                  
template<>
struct check_mixed_type<Mixed> : std::true_type {};

template<class>
struct check_non_mixed_type : std::false_type {};
                        
template<>
struct check_non_mixed_type<glmd> : std::true_type {};

template<class>
struct check_any_mixed_type : std::false_type {};
                        
template<>
struct check_any_mixed_type<Mixed> : std::true_type {};

template<>
struct check_any_mixed_type<MixedNNGP> : std::true_type {};

template<>
struct check_any_mixed_type<MixedHSGP> : std::true_type {};

template<class>
struct check_any_valid_type : std::false_type {};
                            
template<>
struct check_any_valid_type<Mixed> : std::true_type {};
            
template<>
struct check_any_valid_type<MixedNNGP> : std::true_type {};
                            
template<>
struct check_any_valid_type<MixedHSGP> : std::true_type {}; 

template<>
struct check_any_valid_type<glm> : std::true_type {}; 


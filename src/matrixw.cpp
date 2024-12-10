#include <glmmr/matrixw.hpp>


template<typename modeltype>
VectorXd glmmr::MatrixW<modeltype>::W() const{
  return W_;
}

template<typename modeltype>
void glmmr::MatrixW<modeltype>::update(){
  if(W_.size() != model.n())W_.conservativeResize(model.n());
  ArrayXd nvar_par(model.n());
  ArrayXd xb(model.n());
  switch(model.family.family){
  case Fam::gaussian: 
    nvar_par = model.data.variance;
    break;
  case Fam::gamma: 
    nvar_par = model.data.variance.inverse();
    break;
  case Fam::beta:
    nvar_par = (1+model.data.variance);
    break;
  case Fam::binomial:
    nvar_par = model.data.variance.inverse();
    break;
  case Fam::quantile: case Fam::quantile_scaled:
  {
    double qvar = (1 - 2*model.family.quantile + 2*model.family.quantile*model.family.quantile)/(model.family.quantile*model.family.quantile*(1-model.family.quantile)*(1-model.family.quantile));
    if(model.family.family == Fam::quantile_scaled) qvar *= model.data.var_par*model.data.var_par;
    nvar_par.setConstant(qvar);
    break;
  }
  default:
    nvar_par.setConstant(1.0);
    break;
  }
  
  if(attenuated){
    xb = glmmr::maths::attenuted_xb(model.xb(),model.covariance.Z(),model.covariance.D(),model.family.link);
  } else {
    xb = model.xb();
  }
  W_ = glmmr::maths::dhdmu(xb,model.family);
  W_ = (W_.array()*nvar_par).matrix();
  W_ = ((W_.array().inverse()) * model.data.weights).matrix();
}


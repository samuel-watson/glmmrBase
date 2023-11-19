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

enum class MarginType {
  DyDx = 0,
  Diff = 1,
  Ratio = 2
};

enum class SE {
  GLS = 0,
  KR = 1,
  Robust = 2,
  BW = 3
};

template<typename modeltype>
class Model {
public:
  modeltype model;
  glmmr::RandomEffects<modeltype> re;
  glmmr::ModelMatrix<modeltype> matrix;
  glmmr::ModelOptim<modeltype> optim;
  glmmr::ModelMCMC<modeltype> mcmc;
  Model(const std::string& formula_,const ArrayXXd& data_,const strvec& colnames_,std::string family_,std::string link_);
  virtual void set_offset(const VectorXd& offset_);
  virtual void set_weights(const ArrayXd& weights_);
  virtual void set_y(const VectorXd& y_);
  virtual void update_beta(const dblvec &beta_);
  virtual void update_theta(const dblvec &theta_);
  virtual void update_u(const MatrixXd &u_);
  virtual void set_trace(int trace_);
  virtual dblpair marginal(const MarginType type,
                             const std::string& x,
                             const strvec& at,
                             const strvec& atmeans,
                             const strvec& average,
                             const RandomEffectMargin re_type,
                             const SE se_type,
                             const dblpair& xvals,
                             const dblvec& atvals,
                             const dblvec& atrevals);
                                             
};

}

template<typename modeltype>
inline glmmr::Model<modeltype>::Model(const std::string& formula_,
      const ArrayXXd& data_,
      const strvec& colnames_,
      std::string family_, 
      std::string link_) : model(formula_,data_,colnames_,family_,link_), 
      re(model), 
      matrix(model,re,check_type<modeltype>::value,check_type<modeltype>::value),  
      optim(model,matrix,re), mcmc(model,matrix,re) {};

template<typename modeltype>
inline void glmmr::Model<modeltype>::set_offset(const VectorXd& offset_){
  model.data.set_offset(offset_);
}

template<typename modeltype>
inline void glmmr::Model<modeltype>::set_weights(const ArrayXd& weights_){
  model.data.set_weights(weights_);
  if((weights_ != 1.0).any()){
    model.weighted = true;
  }
}

template<typename modeltype>
inline void glmmr::Model<modeltype>::set_y(const VectorXd& y_){
  model.data.update_y(y_);
}

template<typename modeltype>
inline void glmmr::Model<modeltype>::update_beta(const dblvec &beta_){
  model.linear_predictor.update_parameters(beta_);
}

template<typename modeltype>
inline void glmmr::Model<modeltype>::update_theta(const dblvec &theta_){
  model.covariance.update_parameters(theta_);
  re.zu_ = model.covariance.ZLu(re.u_);
}

template<typename modeltype>
inline void glmmr::Model<modeltype>::update_u(const MatrixXd &u_){
  if(u_.cols()!=re.u(false).cols()){
    re.u_.conservativeResize(model.covariance.Q(),u_.cols());
    re.zu_.conservativeResize(model.covariance.Q(),u_.cols());
  }
  re.u_ = u_;
  re.zu_ = model.covariance.ZLu(re.u_);
}

template<typename modeltype>
inline void glmmr::Model<modeltype>::set_trace(int trace_){
  optim.trace = trace_;
  mcmc.trace = trace_;
  if(trace_ > 0){
    mcmc.verbose = true;
  } else {
    mcmc.verbose = false;
  }
}

// marginal effects:
// type is margin type 
// x is name of variable to calculate margin
// at is fixed effects at a set value specified in atvals
// atmeans specifies the fixed effects to set at their mean value
// average specifies the fixed effects to average over
// re_type is random effects margin type
// se_type is the standard error type
// xvals values of the x variable at which to evaluate marginal effect
// atvals is the values for at argument
// atrevals is the random effects values if re_type is At
template<typename modeltype>
inline dblpair glmmr::Model<modeltype>::marginal(const MarginType type,
                                                 const std::string& x,
                                                 const strvec& at,
                                                 const strvec& atmeans,
                                                 const strvec& average,
                                                 const RandomEffectMargin re_type,
                                                 const SE se_type,
                                                 const dblpair& xvals,
                                                 const dblvec& atvals,
                                                 const dblvec& atrevals){
#ifdef R_BUILD
  // some checks
  int total_p = at.size() + atmeans.size() + average.size() + 1;
  if(total_p != model.linear_predictor.P())Rcpp::warning("Unnamed variables will be averaged");
  if(at.size() != atvals.size())Rcpp::stop("Not enough values specified for at");
  if(re_type == RandomEffectMargin::Average && re.zu_.cols()<=1)Rcpp::warning("No MCMC samples of random effects. Random effects will be set at estimated values.");
#endif
    
  using enum Instruction;
  bool single_row = true;
  MatrixXd newXdata(1,model.linear_predictor.Xdata.cols());
  int P = model.linear_predictor.P();
  int N = 1;
  if(average.size() > 0){
    single_row = false;
    N = model.n();
    newXdata.conservativeResize(model.n(),NoChange);
    
#ifdef R_BUILD
   if(re_type == RandomEffectMargin::At && atrevals.size() != model.covariance.Q())Rcpp::stop("Need to provide values for u vector");
#endif
  } else {
#ifdef R_BUILD
    if(re_type == RandomEffectMargin::At && atrevals.size() != 1)Rcpp::stop("Need to provide single value for Zu");
    if(re_type == RandomEffectMargin::AtEstimated)Rcpp::stop("All covariates are at fixed values, cannot used estimated random effects.");
#endif
  }
  
  auto xidx = std::find(model.linear_predictor.calc.data_names.begin(),model.linear_predictor.calc.data_names.end(),x);
  int xcol = xidx - model.linear_predictor.calc.data_names.begin();
  
  if(at.size() > 0){
    for(int p = 0; p < at.size(); p++){
      auto colidx = std::find(model.linear_predictor.calc.data_names.begin(),model.linear_predictor.calc.data_names.end(),at[p]);
      if(colidx != model.linear_predictor.calc.data_names.end()){
        int pcol = colidx - model.linear_predictor.calc.data_names.begin();
        for(int i = 0; i < newXdata.rows(); i++){
          newXdata(i,pcol) = atvals[p];
        }
      } else {
#ifdef R_BUILD
        Rcpp::stop("Variable "+at[p]+" not in data names");  
#endif
      }
    }
  }
  if(atmeans.size() > 0){
    for(int p = 0; p < atmeans.size(); p++){
      auto colidx = std::find(model.linear_predictor.calc.data_names.begin(),model.linear_predictor.calc.data_names.end(),atmeans[p]);
      if(colidx != model.linear_predictor.calc.data_names.end()){
        int pcol = colidx - model.linear_predictor.calc.data_names.begin();
        double xmean = 0;
        for(int i = 0; i < model.n(); i++) xmean += model.linear_predictor.Xdata(i,pcol);
        xmean *= (1.0/model.n());
        for(int i = 0; i < newXdata.rows(); i++){
          newXdata(i,pcol) = xmean;
        }
      } else {
#ifdef R_BUILD
        Rcpp::stop("Variable "+atmeans[p]+" not in data names");  
#endif
      }
    }
  }
  if(average.size() > 0){
    for(int p = 0; p < average.size(); p++){
      auto colidx = std::find(model.linear_predictor.calc.data_names.begin(),model.linear_predictor.calc.data_names.end(),average[p]);
      if(colidx != model.linear_predictor.calc.data_names.end()){
        int pcol = colidx - model.linear_predictor.calc.data_names.begin();
        for(int i = 0; i < newXdata.rows(); i++){
          newXdata(i,pcol) = model.linear_predictor.Xdata(i,pcol);
        }
      } else {
#ifdef R_BUILD
        Rcpp::stop("Variable "+average[p]+" not in data names");  
#endif
      }
    }
  }
  // now create the new calculator object
  glmmr::calculator mcalc = model.linear_predictor.calc;
  mcalc.instructions.push_back(Instruction::PushExtraData);
  mcalc.instructions.push_back(Instruction::Add);
  glmmr::linear_predictor_to_link(mcalc,model.family.link);
  
  dblpair result;
  VectorXd delta = VectorXd::Zero(P);
  MatrixXd M(P,P);
  
  switch(se_type){
    case SE::KR:
      {
      kenward_data kdata = matrix.kenward_roger();
      M = kdata.vcov_beta;
      break;
      }
    case SE::Robust:
      M = matrix.sandwich_matrix();
      break;
    default:
      M = matrix.information_matrix();
      break;
  }
  
  switch(re_type){
  case RandomEffectMargin::At: case RandomEffectMargin::AtEstimated: case RandomEffectMargin::AtZero:
  {
    VectorXd zu(N);
    if(re_type == RandomEffectMargin::At){
      if(single_row){
        zu(0) = atrevals[0];
      } else {
        VectorXd u(model.covariance.Q());
        for(int i = 0; i < model.covariance.Q(); i++)u(i) = atrevals[i];
        zu = model.covariance.Z()*u;
      }
    } else if(re_type == RandomEffectMargin::AtEstimated) {
      zu = re.zu_.rowwise().mean();
    } else if(re_type == RandomEffectMargin::AtZero){
      zu.setZero();
    }
    
    switch(type){
    case MarginType::DyDx:
    {
      double d_result = 0;
      for(int i = 0; i < N; i++){
        dblvec m_result = mcalc.calculate<CalcDyDx::XBeta>(i,model.linear_predictor.parameters,
                                          newXdata,0,xcol,zu(i));
        d_result += m_result[0];
        for(int p = 0; p < P; p++){
          delta(p) += m_result[p+1+P];
        }
      }
      result.first = d_result/N;
      delta.array() *= (1.0/N);
      result.second = sqrt((delta.transpose()*M*delta)(0));
      break;
    }
    case MarginType::Diff:
      {
      double d_result = 0;
      for(int i = 0; i < N; i++){
        newXdata(i,xcol) = xvals.first;
        dblvec m_result = mcalc.calculate<CalcDyDx::BetaFirst>(i,model.linear_predictor.parameters,
                                                           newXdata,0,0,zu(i));
        d_result += m_result[0];
        for(int p = 0; p < P; p++){
          delta(p) += m_result[p+1];
        }
        newXdata(i,xcol) = xvals.second;
        m_result = mcalc.calculate<CalcDyDx::BetaFirst>(i,model.linear_predictor.parameters,
                                                               newXdata,0,0,zu(i));
        d_result -= m_result[0];
        for(int p = 0; p < P; p++){
          delta(p) -= m_result[p+1];
        }
      }
      result.first = d_result/N;
      delta.array() *= (1.0/N);
      result.second = sqrt((delta.transpose()*M*delta)(0));
      break;
      }
    case MarginType::Ratio:
    {
      double d_result0 = 0;
      double d_result1 = 0;
      dblvec delta0(P,0);
      dblvec delta1(P,0);
      for(int i = 0; i < N; i++){
        newXdata(i,xcol) = xvals.first;
        dblvec m_result0 = mcalc.calculate<CalcDyDx::BetaFirst>(i,model.linear_predictor.parameters,
                                                               newXdata,0,0,zu(i));
        newXdata(i,xcol) = xvals.second;
        dblvec m_result1 = mcalc.calculate<CalcDyDx::BetaFirst>(i,model.linear_predictor.parameters,
                                                        newXdata,0,0,zu(i));
        d_result0 += m_result0[0];
        d_result1 += m_result1[0];
        for(int p = 0; p < P; p++){
          delta0[p] += m_result0[p+1];
          delta1[p] += m_result1[p+1];
        }
      }
      result.first = log(d_result0/N) - log(d_result1/N);
      for(int p = 0; p < P; p++){
        delta(p) = delta0[p]*N/d_result0 - delta1[p]*N/d_result1;
      }
      result.second = sqrt((delta.transpose()*M*delta)(0));
      break;
    }
    
    }
  }
    case RandomEffectMargin::Average:
      int iter = re.zu_.cols();
      switch(type){
      case MarginType::DyDx:
      {
        double d_result = 0;
        for(int j = 0; j < iter; j++){
          for(int i = 0; i < N; i++){
            dblvec m_result = mcalc.calculate<CalcDyDx::XBeta>(i,model.linear_predictor.parameters,
                                                               newXdata,0,xcol,re.zu_(i,j));
            d_result += m_result[0];
            for(int p = 0; p < P; p++){
              delta(p) += m_result[p+1+P];
            }
          }
        }
        result.first = d_result/(N*iter);
        delta.array() *= (1.0/(N*iter));
        result.second = sqrt((delta.transpose()*M*delta)(0));
        break;
      }
      case MarginType::Diff:
      {
        double d_result = 0;
        for(int j = 0; j < iter; j++){
          for(int i = 0; i < N; i++){
            newXdata(i,xcol) = xvals.first;
            dblvec m_result = mcalc.calculate<CalcDyDx::BetaFirst>(i,model.linear_predictor.parameters,
                                                                   newXdata,0,0,re.zu_(i,j));
            d_result += m_result[0];
            for(int p = 0; p < P; j++){
              delta(p) += m_result[p+1];
            }
            newXdata(i,xcol) = xvals.second;
            m_result = mcalc.calculate<CalcDyDx::BetaFirst>(i,model.linear_predictor.parameters,
                                                            newXdata,0,0,re.zu_(i,j));
            d_result -= m_result[0];
            for(int p = 0; p < P; p++){
              delta(p) -= m_result[p+1];
            }
          }
        }
        result.first = d_result/(N*iter);
        delta.array() *= (1.0/(N*iter));
        result.second = sqrt((delta.transpose()*M*delta)(0));
        break;
      }
      case MarginType::Ratio:
      {
        double d_result0 = 0;
        double d_result1 = 0;
        dblvec delta0(P,0);
        dblvec delta1(P,0);
        for(int j = 0; j < iter; j++){
          for(int i = 0; i < N; i++){
            newXdata(i,xcol) = xvals.first;
            dblvec m_result0 = mcalc.calculate<CalcDyDx::BetaFirst>(i,model.linear_predictor.parameters,
                                                                    newXdata,0,0,re.zu_(i,j));
            newXdata(i,xcol) = xvals.second;
            dblvec m_result1 = mcalc.calculate<CalcDyDx::BetaFirst>(i,model.linear_predictor.parameters,
                                                                    newXdata,0,0,re.zu_(i,j));
            d_result0 += m_result0[0];
            d_result1 += m_result1[0];
            for(int p = 0; p < P; p++){
              delta0[p] += m_result0[p+1];
              delta1[p] += m_result1[p+1];
            }
          }
        }
        
        result.first = log(d_result0/(N*iter)) - log(d_result1/(N*iter));
        for(int p = 0; p < P; p++){
          delta(p) = delta0[p]*newXdata.size()/d_result0 - delta1[p]*newXdata.size()/d_result1;
        }
        result.second = sqrt((delta.transpose()*M*delta)(0));
        break;
      }
        
      }
  }
  
  return result;
  
}


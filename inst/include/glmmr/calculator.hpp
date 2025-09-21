#pragma once

#include <functional>
#include <stack>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include "maths.h"
#include "instructions.h"



namespace glmmr {

enum class CalcDyDx {
  None,
  BetaFirst,
  BetaSecond,
  XBeta,
  Zu
};

class calcData {
public:
  const int i;
  const int j;
  const int parameterIndex;
  const double extraData;
  int idx_iter = 0;
  std::stack<double> stack;
  std::vector<std::stack<double> > first_dx;
  std::vector<std::stack<double> > second_dx;
  CalcDyDx dydxvar = CalcDyDx::None;
  
  calcData(const int i_, const int j_, const int parameterIndex_, const double extraData_);
  
  void  push_zero();
};

class calculator {
  using func = void(*)(glmmr::calcData&, const glmmr::calculator&);
  
public:
  std::vector<Do>       instructions; // vector of insructions to execute
  intvec                indexes; // indexes of data or parameter vectors
  dblvec                y;  // outcome data
  std::array<double,20> numbers = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  strvec                parameter_names; //vector of parameter names
  strvec                data_names; // vector of data names
  ArrayXd               variance = ArrayXd::Constant(1,1.0); // variance values
  int                   data_count = 0; // number of data items
  int                   parameter_count = 0; // number of parameters
  int                   user_number_count = 0; // number of numbers in the function
  int                   data_size = 0; // number of data items in the calculation
  bool                  any_nonlinear = false; // for linear predictor - any non-linear functions?
  MatrixXd              data = MatrixXd::Zero(1,1); // the data for the calculation
  dblvec                parameters;
  intvec                parameter_indexes;
  calculator() {};
  
  template<CalcDyDx dydx>
  dblvec calculate(const int i, 
                   const int j = 0,
                   const int parameterIndex = 0,
                   const double extraData = 0.0) const;
  
  calculator& operator= (const glmmr::calculator& calc);
  VectorXd      linear_predictor();
  MatrixXd      jacobian();
  MatrixXd      jacobian(const VectorXd& extraData);
  MatrixXd      jacobian(const MatrixXd& extraData);
  MatrixMatrix  jacobian_and_hessian(const MatrixXd& extraData);
  VectorMatrix  jacobian_and_hessian();
  void          update_parameters(const dblvec& parameters_in);
  double        get_covariance_data(const int i, const int j, const int fn);
  void          print_instructions() const;
  void          print_names(bool print_data = true, bool print_parameters = false) const;
  
  template<Do op>
  void          push_back_function();
  
  void          reverse_vectors();
  void          push_user_number();
  
private:
  
  std::vector<func>   fnptr;
  
  // template<Do op>
  // void          operation(calcData& cdata);
  
};

template<Do op>
void operation(calcData& cdata, const calculator& calc);

}

inline glmmr::calcData::calcData(const int i_, const int j_, const int parameterIndex_, 
                                 const double extraData_) : 
  i(i_), j(j_), parameterIndex(parameterIndex_), extraData(extraData_) {};

inline void glmmr::calcData::push_zero(){
  if(dydxvar != CalcDyDx::None)for(auto& fstack: first_dx)fstack.push(0.0);
  if(dydxvar == CalcDyDx::BetaSecond || dydxvar == CalcDyDx::XBeta)for(auto& sstack: second_dx)sstack.push(0.0);
}

template<>
inline void glmmr::operation<Do::PushData>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.data(cdata.i,calc.indexes[cdata.idx_iter]));
  if (cdata.dydxvar== CalcDyDx::BetaFirst || cdata.dydxvar== CalcDyDx::BetaSecond) for(auto& fstack: cdata.first_dx)fstack.push(0.0);
  if (cdata.dydxvar== CalcDyDx::BetaSecond)for(auto& sstack: cdata.second_dx)sstack.push(0.0);
  if (cdata.dydxvar== CalcDyDx::XBeta){
    if(cdata.parameterIndex == calc.indexes[cdata.idx_iter]){
      cdata.first_dx[0].push(1.0);
    } else {
      cdata.first_dx[0].push(0.0);
    }
    for(int ii = 0; ii < calc.parameter_count; ii++)cdata.first_dx[ii+1].push(0.0);
    for(auto& sstack: cdata.second_dx)sstack.push(0.0);
  }
  if (cdata.dydxvar == CalcDyDx::Zu)cdata.first_dx[0].push(0.0);
  cdata.idx_iter++;
}

template<>
inline void glmmr::operation<Do::PushCovData>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  
  if(cdata.i==cdata.j){
    cdata.stack.push(0.0);
  } else {
    int i1 = cdata.i < cdata.j ? (calc.data_size-1)*cdata.i - ((cdata.i-1)*cdata.i/2) + (cdata.j-cdata.i-1) : (calc.data_size-1)*cdata.j - ((cdata.j-1)*cdata.j/2) + (cdata.i-cdata.j-1);
    cdata.stack.push(calc.data(i1,calc.indexes[cdata.idx_iter]));
  }
  if (cdata.dydxvar!= CalcDyDx::None)for(auto& fstack: cdata.first_dx)fstack.push(0.0);
  if (cdata.dydxvar== CalcDyDx::BetaSecond || cdata.dydxvar== CalcDyDx::XBeta)for(auto& sstack: cdata.second_dx)sstack.push(0.0);
  cdata.idx_iter++;
}

template<>
inline void glmmr::operation<Do::PushParameter>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.parameters[calc.indexes[cdata.idx_iter]]);
  if (cdata.dydxvar== CalcDyDx::BetaFirst || cdata.dydxvar== CalcDyDx::BetaSecond){
    for(int idx = 0; idx < calc.parameter_count; idx++){
      if(idx == calc.indexes[cdata.idx_iter]){
        cdata.first_dx[idx].push(1.0);
      } else {
        cdata.first_dx[idx].push(0.0);
      }
    }
  }
  if (cdata.dydxvar== CalcDyDx::BetaSecond)for(auto& sstack: cdata.second_dx)sstack.push(0.0);
  if (cdata.dydxvar== CalcDyDx::XBeta){
    cdata.first_dx[0].push(0.0);
    for(int idx = 0; idx < calc.parameter_count; idx++){
      if(idx == calc.indexes[cdata.idx_iter]){
        cdata.first_dx[idx+1].push(1.0);
      } else {
        cdata.first_dx[idx+1].push(0.0);
      }
    }
    for(auto& sstack: cdata.second_dx)sstack.push(0.0);
  }
  if (cdata.dydxvar== CalcDyDx::Zu)for(auto& fstack: cdata.first_dx)fstack.push(0.0);
  cdata.idx_iter++;
}

template<>
inline void glmmr::operation<Do::Add>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  double b = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(a+b);
  if (cdata.dydxvar!= CalcDyDx::None){
    for(auto& fstack: cdata.first_dx){
      double a = fstack.top();
      fstack.pop();
      double b = fstack.top();
      fstack.pop();
      fstack.push(a+b);
    }
  }
  if (cdata.dydxvar== CalcDyDx::BetaSecond || cdata.dydxvar== CalcDyDx::XBeta){
    for(auto& sstack: cdata.second_dx){
      double a = sstack.top();
      sstack.pop();
      double b = sstack.top();
      sstack.pop();
      sstack.push(a+b);
    }
  }
}

template<>
inline void glmmr::operation<Do::Subtract>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  double b = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(a-b);
  if (cdata.dydxvar!= CalcDyDx::None){
    for(auto& fstack: cdata.first_dx){
      double a = fstack.top();
      fstack.pop();
      double b = fstack.top();
      fstack.pop();
      fstack.push(a-b);
    }
  }
  if (cdata.dydxvar== CalcDyDx::BetaSecond || cdata.dydxvar== CalcDyDx::XBeta){
    for(auto& sstack: cdata.second_dx){
      double a = sstack.top();
      sstack.pop();
      double b = sstack.top();
      sstack.pop();
      sstack.push(a-b);
    }
  }
}

template<>
inline void glmmr::operation<Do::Multiply>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  double b = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(a*b);
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    dblvec b_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      b_top_dx.push_back(fstack.top());
      fstack.pop();
      fstack.push(a*b_top_dx.back() + b*a_top_dx.back());
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double bdx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = a*bdx2 + b*adx2 + a_top_dx[idx]*b_top_dx[jdx] + a_top_dx[jdx]*b_top_dx[idx];
          cdata.second_dx[index_count].push(result);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double bdx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = a*bdx2 + b*adx2 + a_top_dx[0]*b_top_dx[jdx+1] + a_top_dx[jdx+1]*b_top_dx[0];
        cdata.second_dx[jdx].push(result);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::Divide>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  double b = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(a/b);
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    dblvec b_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      b_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = (b*a_top_dx.back() - a*b_top_dx.back())/(b*b);
      fstack.push(result);
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double bdx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = (adx2*b - a_top_dx[idx]*b_top_dx[jdx]- a_top_dx[jdx]*b_top_dx[idx])/(b*b) - (a*bdx2*b - 2*b_top_dx[idx]*b_top_dx[jdx])/(b*b*b);
          cdata.second_dx[index_count].push(result);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double bdx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = (adx2*b - a_top_dx[0]*b_top_dx[jdx+1]- a_top_dx[jdx+1]*b_top_dx[0])/(b*b) - (a*bdx2*b - 2*b_top_dx[0]*b_top_dx[jdx])/(b*b*b);
        cdata.second_dx[jdx].push(result);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::Sqrt>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(sqrt(a));
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = a==0 ? 0 : 0.5*pow(a,-0.5)*a_top_dx.back();
      fstack.push(result);
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = a==0? 0 : 0.5*pow(a,-0.5)*adx2 - 0.25*a_top_dx[idx]*a_top_dx[jdx]*pow(a,-3/2);
          cdata.second_dx[index_count].push(result);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = a==0? 0 : 0.5*pow(a,-0.5)*adx2 - 0.25*a_top_dx[0]*a_top_dx[jdx+1]*pow(a,-3/2);
        cdata.second_dx[jdx].push(result);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::Power>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  double b = cdata.stack.top();
  cdata.stack.pop();
  double out = pow(a,b);
  
#ifdef R_BUILD
  // try and catch these possible failures as it seems to cause crash if nan allowed to propogate
  if(out != out)throw std::runtime_error("Exponent fail: "+std::to_string(a)+"^"+std::to_string(b));
#endif
  
  cdata.stack.push(out);
  
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    dblvec b_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      b_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = 0;
      if(a > 0) result += pow(a,b)*b_top_dx.back()*log(a) + pow(a,b-1)*b*a_top_dx.back(); 
#ifdef R_BUILD
      // this can sometimes result in a crash if the values of the parameters aren't right
      if(result != result)throw std::runtime_error("Exponent dydx fail: "+std::to_string(a)+"^"+std::to_string(b-1));
#endif
      fstack.push(result);
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double bdx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result1 = cdata.first_dx[jdx].top()*b_top_dx[idx]*log(a) + cdata.stack.top()*(bdx2*log(a) + b_top_dx[idx]*(1/a)*a_top_dx[jdx]);
          double result2 = pow(a,b-1)*b_top_dx[jdx]*log(a) + pow(a,b-2)*(b-1)*a_top_dx[jdx];
          double result3 = result2*b*a_top_dx[idx] + pow(a,b-1)*(b*adx2+b_top_dx[jdx]*a_top_dx[idx]);
          double result4 = 0;
          if(a > 0) result4 = result1 + result3;
#ifdef R_BUILD
          // this can sometimes result in a crash if the values of the parameters aren't right
          if(result4 != result4)throw std::runtime_error("Exponent d2ydx2 fail: "+std::to_string(a)+"^"+std::to_string(b-2)+" (i,j) = ("+std::to_string(idx)+","+std::to_string(jdx)+")");
#endif
          cdata.second_dx[index_count].push(result4);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double bdx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result1 = cdata.first_dx[jdx+1].top()*b_top_dx[0]*log(a) + cdata.stack.top()*(bdx2*log(a) + b_top_dx[0]*(1/a)*a_top_dx[jdx+1]);
        double result2 = pow(a,b-1)*b_top_dx[jdx+1]*log(a) + pow(a,b-2)*(b-1)*a_top_dx[jdx+1];
        double result3 = result2*b*a_top_dx[0] + pow(a,b-1)*(b*adx2+b_top_dx[jdx+1]*a_top_dx[0]);
        cdata.second_dx[jdx].push(result1 + result3);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::Exp>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(exp(a));
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = cdata.stack.top()*a_top_dx.back();
      fstack.push(result);
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = cdata.stack.top()*(a_top_dx[idx]*a_top_dx[jdx] + adx2);
          cdata.second_dx[index_count].push(result);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = cdata.stack.top()*(a_top_dx[0]*a_top_dx[jdx+1] + adx2);
        cdata.second_dx[jdx].push(result);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::Negate>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(-1*a);
  if (cdata.dydxvar!= CalcDyDx::None){
    for(auto& fstack: cdata.first_dx){
      double ftop = fstack.top();
      fstack.pop();
      fstack.push(-1.0*ftop);
    }
  }
  if (cdata.dydxvar== CalcDyDx::BetaSecond || cdata.dydxvar== CalcDyDx::XBeta){
    for(auto& sstack: cdata.second_dx){
      double adx2 = sstack.top();
      sstack.pop();
      sstack.push(-1*adx2);
    }
  }
}

template<>
inline void glmmr::operation<Do::Bessel>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  double b = boost::math::cyl_bessel_k(1,a);
  cdata.stack.push(b);
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = -0.5*boost::math::cyl_bessel_k(0,a)-0.5*boost::math::cyl_bessel_k(2,a);
      fstack.push(result*a_top_dx.back());
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = 0.25*boost::math::cyl_bessel_k(-1,a)+0.5*boost::math::cyl_bessel_k(1,a)+0.25*boost::math::cyl_bessel_k(3,a);
          double result1 = -0.5*boost::math::cyl_bessel_k(0,a)-0.5*boost::math::cyl_bessel_k(2,a);
          double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
          cdata.second_dx[index_count].push(result2);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = 0.25*boost::math::cyl_bessel_k(-1,a)+0.5*boost::math::cyl_bessel_k(1,a)+0.25*boost::math::cyl_bessel_k(3,a);
        double result1 = -0.5*boost::math::cyl_bessel_k(0,a)-0.5*boost::math::cyl_bessel_k(2,a);
        double result2 = result*a_top_dx[0]*a_top_dx[jdx+1]+result1*adx2;
        cdata.second_dx[jdx].push(result2);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::Gamma>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(tgamma(a));
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = cdata.stack.top()*boost::math::polygamma(0,a);
      fstack.push(result*a_top_dx.back());
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = cdata.stack.top()*boost::math::polygamma(0,a)*boost::math::polygamma(0,a) + cdata.stack.top()*boost::math::polygamma(1,a);
          double result1 = cdata.stack.top()*boost::math::polygamma(0,a);
          double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
          cdata.second_dx[index_count].push(result2);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = cdata.stack.top()*boost::math::polygamma(0,a)*boost::math::polygamma(0,a) + cdata.stack.top()*boost::math::polygamma(1,a);
        double result1 = cdata.stack.top()*boost::math::polygamma(0,a);
        double result2 = result*a_top_dx[0]*a_top_dx[jdx+1]+result1*adx2;
        cdata.second_dx[jdx].push(result2);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::Sin>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(sin(a));
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = cos(a)*a_top_dx.back();
      fstack.push(result);
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = -1.0*sin(a);
          double result1 = cos(a);
          double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
          cdata.second_dx[index_count].push(result2);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = -1.0*sin(a);
        double result1 = cos(a);
        double result2 = result*a_top_dx[0]*a_top_dx[jdx+1]+result1*adx2;
        cdata.second_dx[jdx].push(result2);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::Cos>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(cos(a));
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = -1.0*sin(a)*a_top_dx.back();
      fstack.push(result);
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = -1.0*cos(a);
          double result1 = -1.0*sin(a);
          double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
          cdata.second_dx[index_count].push(result2);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = -1.0*cos(a);
        double result1 = -1.0*sin(a);
        double result2 = result*a_top_dx[0]*a_top_dx[jdx+1]+result1*adx2;
        cdata.second_dx[jdx].push(result2);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::BesselK>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  double b = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(boost::math::cyl_bessel_k(b,a));
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = -0.5*boost::math::cyl_bessel_k(b-1,a)-0.5*boost::math::cyl_bessel_k(b+1,a);
      fstack.push(result*a_top_dx.back());
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = 0.25*boost::math::cyl_bessel_k(b-2,a)+0.5*boost::math::cyl_bessel_k(b,a)+0.25*boost::math::cyl_bessel_k(b+2,a);
          double result1 = -0.5*boost::math::cyl_bessel_k(b-1,a)-0.5*boost::math::cyl_bessel_k(b+1,a);
          double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
          cdata.second_dx[index_count].push(result2);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = 0.25*boost::math::cyl_bessel_k(b-2,a)+0.5*boost::math::cyl_bessel_k(b,a)+0.25*boost::math::cyl_bessel_k(b+2,a);
        double result1 = -0.5*boost::math::cyl_bessel_k(b-1,a)-0.5*boost::math::cyl_bessel_k(b+1,a);
        double result2 = result*a_top_dx[0]*a_top_dx[jdx+1]+result1*adx2;
        cdata.second_dx[jdx].push(result2);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::ErrorFunc>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(boost::math::erf(a));
  if (cdata.dydxvar!= CalcDyDx::None)
  {
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx)
    {
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = 1.12837916709551257390 * exp(-1.0 * a * a);
      result *= a_top_dx.back();
      fstack.push(result);
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond)
    {
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = -2.0 * a * 1.12837916709551257390 * exp(-1.0 * a * a);
          double result1 = 1.12837916709551257390 * exp(-1.0 * a * a);
          double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
          cdata.second_dx[index_count].push(result2);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = -2.0 * a * 1.12837916709551257390 * exp(-1.0 * a * a);
        double result1 = 1.12837916709551257390 * exp(-1.0 * a * a);
        double result2 = result*a_top_dx[0]*a_top_dx[jdx+1]+result1*adx2;
        cdata.second_dx[jdx].push(result2);
      }
    }
  }       
}

template<>
inline void glmmr::operation<Do::Log>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(log(a));
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = (1/a)*a_top_dx.back();
      fstack.push(result);
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result = -1.0/(a*a);
          double result1 = 1/a;
          double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
          cdata.second_dx[index_count].push(result2);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result = -1.0/(a*a);
        double result1 = 1/a;
        double result2 = result*a_top_dx[0]*a_top_dx[jdx+1]+result1*adx2;
        cdata.second_dx[jdx].push(result2);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::Square>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  cdata.stack.push(a*a);
  if (cdata.dydxvar!= CalcDyDx::None){
    dblvec a_top_dx;
    for(auto& fstack: cdata.first_dx){
      a_top_dx.push_back(fstack.top());
      fstack.pop();
      double result = 2*a*a_top_dx.back();
      fstack.push(result);
    }
    if (cdata.dydxvar== CalcDyDx::BetaSecond){
      int index_count = 0;
      for(int idx = 0; idx < calc.parameter_count; idx++){
        for(int jdx = idx; jdx < calc.parameter_count; jdx++){
          double adx2 = cdata.second_dx[index_count].top();
          cdata.second_dx[index_count].pop();
          double result2 = 2*a_top_dx[idx]*a_top_dx[jdx]+2*a*adx2;
          cdata.second_dx[index_count].push(result2);
          index_count++;
        }
      }
    }
    if (cdata.dydxvar== CalcDyDx::XBeta){
      for(int jdx = 0; jdx < calc.parameter_count; jdx++){
        double adx2 = cdata.second_dx[jdx].top();
        cdata.second_dx[jdx].pop();
        double result2 = 2*a_top_dx[0]*a_top_dx[jdx+1]+2*a*adx2;
        cdata.second_dx[jdx].push(result2);
      }
    }
  }
}

template<>
inline void glmmr::operation<Do::PushExtraData>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(cdata.extraData);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Sign>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = calc.data(cdata.i,calc.indexes[cdata.idx_iter]);
  if(a > 0){
    cdata.stack.push(1.0);
  } else if(a< 0){
    cdata.stack.push(-1.0);
  } else {
    cdata.stack.push(0.0);
  }     
  cdata.push_zero();
  cdata.idx_iter++;
}

template<>
inline void glmmr::operation<Do::SignNoZero>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = calc.data(cdata.i,calc.indexes[cdata.idx_iter]);
  if(a >= 0){
    cdata.stack.push(1.0);
  } else {
    cdata.stack.push(-1.0);
  }    
  cdata.push_zero();
  cdata.idx_iter++;
}

template<>
inline void glmmr::operation<Do::PushY>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.y[cdata.i]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int10>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(10);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int1>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(1);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int2>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(2);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int3>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(3);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int4>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(4);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int5>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(5);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int6>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(6);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int7>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(7);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int8>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(8);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Int9>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(9);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Pi>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(M_PI);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Constant1>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(0.3275911);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Constant2>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(0.254829592);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Constant3>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(-0.284496736);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Constant4>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(1.421413741);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Constant5>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(-1.453152027);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Constant6>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(1.061405429);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::LogFactorialApprox>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  double a = cdata.stack.top();
  cdata.stack.pop();
  // Ramanujan approximation
  if(a == 0){
    cdata.stack.push(0);
  } else {
    double result = a*log(a) - a + log(a*(1+4*a*(1+2*a)))/6 + log(3.141593)/2;
    cdata.stack.push(result);
  }
}

template<>
inline void glmmr::operation<Do::PushVariance>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.variance(cdata.i));
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber0>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  //Rcpp::Rcout << "\nPush " << calc.numbers[0];
  cdata.stack.push(calc.numbers[0]);
  //cdata.stack.push(1.0);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber1>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[1]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber2>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[2]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber3>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[3]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber4>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[4]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber5>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[5]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber6>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[6]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber7>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[7]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber8>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[8]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber9>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[9]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber10>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[10]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber11>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[11]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber12>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[12]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber13>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[13]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber14>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[14]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber15>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[15]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber16>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[16]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber17>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[17]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber18>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[18]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::PushUserNumber19>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(calc.numbers[19]);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::SqrtTwo>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(0.7071068);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Half>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(0.5);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::HalfLog2Pi>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(0.9189385);
  cdata.push_zero();
}

template<>
inline void glmmr::operation<Do::Pi2>(glmmr::calcData& cdata, const glmmr::calculator& calc){
  cdata.stack.push(2*M_PI);
  cdata.push_zero();
}

template<Do op>
inline void glmmr::calculator::push_back_function(){
  fnptr.push_back(&operation<op>);
  // fnptr.push_back(
  //   static_cast<func>([this](calcData& cdata){
  //     std::invoke(&operation<op>,cdata,*this);
  //   })
  // );
}

inline void glmmr::calculator::push_user_number(){
  switch(user_number_count){
  case 0:
    push_back_function<Do::PushUserNumber0>();
    break;
  case 1:
    push_back_function<Do::PushUserNumber1>();
    break;
  case 2:
    push_back_function<Do::PushUserNumber2>();
    break;
  case 3:
    push_back_function<Do::PushUserNumber3>();
    break;
  case 4:
    push_back_function<Do::PushUserNumber4>();
    break;
  case 5:
    push_back_function<Do::PushUserNumber5>();
    break;
  case 6:
    push_back_function<Do::PushUserNumber6>();
    break;
  case 7:
    push_back_function<Do::PushUserNumber7>();
    break;
  case 8:
    push_back_function<Do::PushUserNumber8>();
    break;
  case 9:
    push_back_function<Do::PushUserNumber9>();
    break;
  case 10:
    push_back_function<Do::PushUserNumber10>();
    break;
  case 11:
    push_back_function<Do::PushUserNumber11>();
    break;
  case 12:
    push_back_function<Do::PushUserNumber12>();
    break;
  case 13:
    push_back_function<Do::PushUserNumber13>();
    break;
  case 14:
    push_back_function<Do::PushUserNumber14>();
    break;
  case 15:
    push_back_function<Do::PushUserNumber15>();
    break;
  case 16:
    push_back_function<Do::PushUserNumber16>();
    break;
  case 17:
    push_back_function<Do::PushUserNumber17>();
    break;
  case 18:
    push_back_function<Do::PushUserNumber18>();
    break;
  case 19:
    push_back_function<Do::PushUserNumber19>();
    break;
  }
  
}

inline void glmmr::calculator::reverse_vectors(){
  std::reverse(instructions.begin(),instructions.end());
  std::reverse(fnptr.begin(),fnptr.end());
  std::reverse(indexes.begin(),indexes.end());
}

inline void glmmr::calculator::update_parameters(const dblvec& parameters_in){
  if(static_cast<int>(parameters_in.size()) < parameter_count)throw std::runtime_error("Expecting "+std::to_string(parameter_count)+" parameters in calculator but got "+std::to_string(parameters_in.size()));
  for(int i = 0; i < parameter_indexes.size(); i++)parameters[i] = parameters_in[parameter_indexes[i]];
}

inline VectorXd glmmr::calculator::linear_predictor(){
  int n = data.rows();
  VectorXd x(n);
#pragma omp parallel for
  for(int i = 0; i < n; i++){
    x(i) = calculate<CalcDyDx::None>(i)[0];
  }
  return x;
};

inline glmmr::calculator& glmmr::calculator::operator= (const glmmr::calculator& calc){
  instructions = calc.instructions;
  indexes = calc.indexes;
  parameter_names = calc.parameter_names;
  data_names = calc.data_names;
  variance.conservativeResize(calc.variance.size());
  variance = calc.variance;
  data_count = calc.data_count;
  parameter_count = calc.parameter_count;
  any_nonlinear = calc.any_nonlinear;
  data.conservativeResize(calc.data.rows(),calc.data.cols());
  data = calc.data;
  parameters.resize(calc.parameters.size());
  parameters = calc.parameters;
  return *this;
};

inline MatrixXd glmmr::calculator::jacobian(){
  int n = data.rows();
#ifdef ENABLE_DEBUG
  if(n==0)throw std::runtime_error("No data initialised in calculator");
#endif
  MatrixXd J(n,parameter_count);
#pragma omp parallel for
  for(int i = 0; i<n ; i++){
    dblvec out = calculate<CalcDyDx::BetaFirst>(i);
    for(int j = 0; j<parameter_count; j++){
      J(i,j) = out[j+1];
    }
  }
  return J;
};

inline MatrixXd glmmr::calculator::jacobian(const VectorXd& extraData){
  int n = data.rows();
#ifdef ENABLE_DEBUG
  if(n==0)throw std::runtime_error("No data initialised in calculator");
#endif 
  MatrixXd J(n,parameter_count);
//#pragma omp parallel for
  for(int i = 0; i<n ; i++){
    dblvec out = calculate<CalcDyDx::BetaFirst>(i,0,0,extraData(i));
    for(int j = 0; j<parameter_count; j++){
      J(i,j) = out[j+1];
    }
  }
  return J;
};

inline MatrixXd glmmr::calculator::jacobian(const MatrixXd& extraData){
  int n = data.rows();
  
#ifdef ENABLE_DEBUG
  if(n==0)throw std::runtime_error("No data initialised in calculator");
  if(extraData.rows()!=n)throw std::runtime_error("Extra data not of length n");
#endif
  
  int iter = extraData.cols();
  MatrixXd J = MatrixXd::Zero(parameter_count,n);
//#pragma omp parallel for
  for(int i = 0; i<n ; i++){
    dblvec out;
    for(int k = 0; k < iter; k++){
      out = calculate<CalcDyDx::BetaFirst>(i,0,0,extraData(i,k));
      for(int j = 0; j < parameter_count; j++){
        J(j,i) += out[1+j]/iter;
      }
    }
  }
  return J;
};


inline MatrixMatrix glmmr::calculator::jacobian_and_hessian(const MatrixXd& extraData){
  int n = data.rows();
  MatrixMatrix result(parameter_count,parameter_count,parameter_count,n);
  
#ifdef ENABLE_DEBUG
  if(n==0)throw std::runtime_error("No data initialised in calculator");
  if(extraData.rows()!=n)throw std::runtime_error("Extra data not of length n");
#endif
  
  int iter = extraData.cols();
  int n2d = parameter_count*(parameter_count + 1)/2;
  MatrixXd H = MatrixXd::Zero(n2d,n);
  MatrixXd J = MatrixXd::Zero(parameter_count,n);
//#pragma omp parallel for collapse(2)
  for(int i = 0; i<n ; i++){
    for(int k = 0; k < iter; k++){
      dblvec out = calculate<CalcDyDx::BetaSecond>(i,0,0,extraData(i,k));
      for(int j = 0; j < parameter_count; j++){
        J(j,i) += out[1+j]/iter;
      }
      for(int j = 0; j < n2d; j++){
        H(j,i) += out[parameter_count + 1 + j]/iter;
      }
    }
  }
  VectorXd Hmean = H.rowwise().sum();
  MatrixXd H0 = MatrixXd::Zero(parameter_count, parameter_count);
  int index_count = 0;
  for(int j = 0; j < parameter_count; j++){
    for(int k = j; k < parameter_count; k++){
      H0(k,j) = Hmean[index_count];
      if(j != k) H0(j,k) = H0(k,j);
      index_count++;
    }
  }
  result.mat1 = H0;
  result.mat2 = J;
  return result;
};

inline VectorMatrix glmmr::calculator::jacobian_and_hessian(){
  VectorMatrix result(parameter_count);
  int n2d = parameter_count*(parameter_count + 1)/2;
  VectorXd H = VectorXd::Zero(n2d);
  VectorXd J = VectorXd::Zero(parameter_count);
  MatrixXd dat = MatrixXd::Zero(1,1);
  dblvec out = calculate<CalcDyDx::BetaSecond>(0,0,2,0);
  for(int j = 0; j < parameter_count; j++){
    J(j,0) += out[1+j];
  }
  for(int j = 0; j < n2d; j++){
    H(j) += out[parameter_count + 1 + j];
  }
  MatrixXd H0 = MatrixXd::Zero(parameter_count, parameter_count);
  int index_count = 0;
  for(int j = 0; j < parameter_count; j++){
    for(int k = j; k < parameter_count; k++){
      H0(k,j) = H(index_count);
      if(j != k) H0(j,k) = H0(k,j);
      index_count++;
    }
  }
  result.mat = H0;
  result.vec = J;
  return result;
};

inline double glmmr::calculator::get_covariance_data(const int i, const int j, const int fn){
  int i1 = i < j ? (data_size-1)*i - ((i-1)*i/2) + (j-i-1) : (data_size-1)*j - ((j-1)*j/2) + (i-j-1);
#ifdef ENABLE_DEBUG
  if(i1 >= data.rows())throw std::runtime_error("PushCovData: Index out of range: "+std::to_string(i1)+" versus "+std::to_string(data.rows()));
#endif
  return data(i1,fn);
}


template<glmmr::CalcDyDx dydx>
inline dblvec glmmr::calculator::calculate(const int i, 
                                           const int j,
                                           const int parameterIndex,
                                           const double extraData) const {
  
  calcData cdata(i, j, parameterIndex, extraData);
  
  if constexpr(dydx != CalcDyDx::None){
    if constexpr(dydx == CalcDyDx::BetaFirst){
      cdata.first_dx.resize(parameter_count);
      cdata.dydxvar = CalcDyDx::BetaFirst;
    } else if constexpr(dydx == CalcDyDx::BetaSecond){
      cdata.first_dx.resize(parameter_count);
      cdata.dydxvar = CalcDyDx::BetaSecond;
    } else if constexpr(dydx == CalcDyDx::XBeta){
      cdata.dydxvar = CalcDyDx::XBeta;
      cdata.first_dx.resize(1+parameter_count);
    } else if constexpr(dydx == CalcDyDx::Zu){
      cdata.dydxvar = CalcDyDx::Zu;
      cdata.first_dx.resize(1);
    }
  } 
    
  
  if constexpr (dydx == CalcDyDx::XBeta || dydx == CalcDyDx::BetaSecond){
    if constexpr(dydx == CalcDyDx::BetaSecond){
      cdata.second_dx.resize(parameter_count*(parameter_count + 1)/2);
    } else if constexpr(dydx ==  CalcDyDx::XBeta){
      cdata.second_dx.resize(parameter_count);
    }
  }
  
  // Rcpp::Rcout << "\nLength of fnvec: " << fnptr.size();
  // throw std::runtime_error("Stopping here to see if it doesn't crash");
  
  for(auto k: fnptr){
    //std::invoke(k, cdata);
    (*k)(cdata, *this);
  }
  
#ifdef ENABLE_DEBUG
  if(cdata.stack.size()>1)Rcpp::warning("More than one element on the stack at end of calculation");
#endif
  
  dblvec result;
  result.push_back(cdata.stack.top());
  
  if constexpr (dydx != CalcDyDx::None){
    for(const auto& fstack: cdata.first_dx){
#ifdef ENABLE_DEBUG
      if(fstack.size()==0)throw std::runtime_error("Error derivative stack empty");
#endif
      result.push_back(fstack.top());
    }
  }
  if constexpr (dydx == CalcDyDx::BetaSecond || dydx == CalcDyDx::XBeta){
    for(const auto& sstack: cdata.second_dx){
#ifdef ENABLE_DEBUG
      if(sstack.size()==0)throw std::runtime_error("Error second derivative stack empty");
#endif
      result.push_back(sstack.top());
    }
  }
  
  return result;
}

inline void glmmr::calculator::print_instructions() const {
  //currently only setup for R
#ifdef R_BUILD
  int counter = 1;
  int idx_iter = 0;
  Rcpp::Rcout << "\nInstructions:\n";
  for(const auto& i: instructions){
    Rcpp::Rcout << counter << ". " << instruction_str.at(i);
    switch(i){
    case Do::PushUserNumber0:
      Rcpp::Rcout << " = " << numbers[0] << "\n";
      break;
    case Do::PushUserNumber1:
      Rcpp::Rcout << " = " << numbers[1] << "\n";
      break;
    case Do::PushUserNumber2:
      Rcpp::Rcout << " = " << numbers[2] << "\n";
      break;
    case Do::PushUserNumber3:
      Rcpp::Rcout << " = " << numbers[3] << "\n";
      break;
    case Do::PushUserNumber4:
      Rcpp::Rcout << " = " << numbers[4] << "\n";
      break;
    case Do::PushUserNumber5:
      Rcpp::Rcout << " = " << numbers[5] << "\n";
      break;
    case Do::PushUserNumber6:
      Rcpp::Rcout << " = " << numbers[6] << "\n";
      break;
    case Do::PushUserNumber7:
      Rcpp::Rcout << " = " << numbers[7] << "\n";
      break;
    case Do::PushUserNumber8:
      Rcpp::Rcout << " = " << numbers[8] << "\n";
      break;
    case Do::PushUserNumber9:
      Rcpp::Rcout << " = " << numbers[9] << "\n";
      break;
    case Do::PushParameter:
      {
        if(indexes[idx_iter] >= parameter_names.size()){
          Rcpp::Rcout << "\nError in instruction set";
          Rcpp::Rcout << "\nIndex " << indexes[idx_iter] << " requested for parameter size " << parameter_names.size();
          Rcpp::Rcout << "\nIndexes: ";
          glmmr::print_vec_1d(indexes);
          Rcpp::Rcout << "\nParameter names: ";
          glmmr::print_vec_1d(parameter_names);
          throw std::runtime_error("Execution halted");
        }
        Rcpp::Rcout << ": " << parameter_names[indexes[idx_iter]] << "; index " << indexes[idx_iter] <<"\n";
        idx_iter++;
        break;
      }
    case Do::PushData: case Do::Sign: case Do::SignNoZero:
      {
        if(indexes[idx_iter] >= data_names.size()){
          Rcpp::Rcout << "\nError in instruction set";
          Rcpp::Rcout << "\nIndex " << indexes[idx_iter] << " requested for data size " << data_names.size();
          Rcpp::Rcout << "\nIndexes: ";
          glmmr::print_vec_1d(indexes);
          Rcpp::Rcout << "\nData names: ";
          glmmr::print_vec_1d(data_names);
          throw std::runtime_error("Execution halted");
        }
        Rcpp::Rcout << " (column " << data_names[indexes[idx_iter]] << "; index " << indexes[idx_iter] <<")\n";
        idx_iter++;
        break;
      }
      
    case Do::PushCovData:
      Rcpp::Rcout << " (column " << indexes[idx_iter] << ")\n";
      idx_iter++;
      break;
    default:
      Rcpp::Rcout << "\n";
    }
    counter++;
  }
  Rcpp::Rcout << "\n";
#endif
}

inline void glmmr::calculator::print_names(bool print_data, bool print_parameters) const {
#ifdef R_BUILD
  Rcpp::Rcout << "\nParameter count " << parameter_count << " vec size: " << parameters.size();
  Rcpp::Rcout << "\nData count " << data_count << " mat size: " << data.rows() << " x " << data.cols();
  Rcpp::Rcout << "\nIndexes: ";
  glmmr::print_vec_1d(indexes);
  Rcpp::Rcout << "\nAny nonlinear? " << any_nonlinear;
  if(print_data)
  {
    Rcpp::Rcout << "\nData names: ";
    glmmr::print_vec_1d<strvec>(data_names);
  }
  if(print_parameters)
  {
    Rcpp::Rcout << "\nParameter names: ";
    glmmr::print_vec_1d<strvec>(parameter_names);
  }
  VectorXd x(10);
  for(int i = 0; i < 10; i++) x(i) = calculate<CalcDyDx::None>(i)[0];
  Rcpp::Rcout << "\nExample data: " << x.transpose() << "\n";
  
#endif
}
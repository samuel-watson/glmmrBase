#pragma once

#include "general.h"
#include "openmpheader.h"
#include "instructions.h"

namespace glmmr {

class calculator {
  public:
    std::vector<Instruction> instructions;
    intvec indexes;
    dblvec y;
    std::array<double,10> numbers;
    strvec parameter_names;
    ArrayXd variance = ArrayXd::Constant(1,1.0);
    int data_count = 0;
    int parameter_count = 0;
    int user_number_count = 0;
    bool any_nonlinear = false;
    
    calculator() {};
    
    dblvec calculate(const int i, 
                     const dblvec& parameters, 
                     const MatrixXd& data,
                     const int j = 0,
                     int order = 0, 
                     double extraData = 0.0,
                     int n = 0);
    
    calculator& operator= (const glmmr::calculator& calc);
    VectorXd linear_predictor(const dblvec& parameters,const MatrixXd& data);
    MatrixXd jacobian(const dblvec& parameters, const MatrixXd& data);
    MatrixXd jacobian(const dblvec& parameters,const MatrixXd& data,const VectorXd& extraData);
    MatrixXd jacobian(const dblvec& parameters,const MatrixXd& data,const MatrixXd& extraData);
    matrix_matrix jacobian_and_hessian(const dblvec& parameters,const MatrixXd& data,const MatrixXd& extraData);
    vector_matrix jacobian_and_hessian(const dblvec& parameters);
    void print_instructions();
};

}

inline VectorXd glmmr::calculator::linear_predictor(const dblvec& parameters, 
                                                    const MatrixXd& data){
  int n = data.rows();
  VectorXd x(n);
#pragma omp parallel for
  for(int i = 0; i < n; i++){
    x(i) = calculate(i,parameters,data)[0];
  }
  return x;
};

inline glmmr::calculator& glmmr::calculator::operator= (const glmmr::calculator& calc){
  instructions = calc.instructions;
  indexes = calc.indexes;
  parameter_names = calc.parameter_names;
  variance.conservativeResize(calc.variance.size());
  variance = calc.variance;
  data_count = calc.data_count;
  parameter_count = calc.parameter_count;
  any_nonlinear = calc.any_nonlinear;
  return *this;
};

inline MatrixXd glmmr::calculator::jacobian(const dblvec& parameters,
                                            const MatrixXd& data){
  int n = data.rows();
  #if defined(ENABLE_DEBUG) && defined(R_BUILD)
  if(n==0)Rcpp::stop("No data initialised in calculator");
  #endif
  MatrixXd J(n,parameter_count);
#pragma omp parallel for
  for(int i = 0; i<n ; i++){
    dblvec out = calculate(i,parameters,data,0,1,0);
    for(int j = 0; j<parameter_count; j++){
      J(i,j) = out[j+1];
    }
  }
  return J;
};

inline MatrixXd glmmr::calculator::jacobian(const dblvec& parameters,
                                            const MatrixXd& data,
                                            const VectorXd& extraData){
  int n = data.rows();
  #if defined(ENABLE_DEBUG) && defined(R_BUILD)
  if(n==0)Rcpp::stop("No data initialised in calculator");
  #endif 
  MatrixXd J(n,parameter_count);
#pragma omp parallel for
  for(int i = 0; i<n ; i++){
    dblvec out = calculate(i,parameters,data,0,1,extraData(i));
    for(int j = 0; j<parameter_count; j++){
      J(i,j) = out[j+1];
    }
  }
  return J;
};

inline MatrixXd glmmr::calculator::jacobian(const dblvec& parameters,
                                            const MatrixXd& data,
                                            const MatrixXd& extraData){
  int n = data.rows();
  
  #if defined(ENABLE_DEBUG) && defined(R_BUILD)
  if(n==0)Rcpp::stop("No data initialised in calculator");
  if(extraData.rows()!=n)Rcpp::stop("Extra data not of length n");
  #endif
  
  int iter = extraData.cols();
  MatrixXd J = MatrixXd::Zero(parameter_count,n);
#pragma omp parallel for
  for(int i = 0; i<n ; i++){
    dblvec out;
    for(int k = 0; k < iter; k++){
      out = calculate(i,parameters,data,0,1,extraData(i,k));
      for(int j = 0; j < parameter_count; j++){
        J(j,i) += out[1+j]/iter;
      }
    }
  }
  return J;
};


inline matrix_matrix glmmr::calculator::jacobian_and_hessian(const dblvec& parameters,
                                                             const MatrixXd& data,
                                                             const MatrixXd& extraData){
  int n = data.rows();
  matrix_matrix result(parameter_count,parameter_count,parameter_count,n);
  
  #ifdef ENABLE_DEBUG
  if(n==0)Rcpp::stop("No data initialised in calculator");
  if(extraData.rows()!=n)Rcpp::stop("Extra data not of length n");
  #endif
  
  int iter = extraData.cols();
  int n2d = parameter_count*(parameter_count + 1)/2;
  MatrixXd H = MatrixXd::Zero(n2d,n);
  MatrixXd J = MatrixXd::Zero(parameter_count,n);
#pragma omp parallel for collapse(2)
  for(int i = 0; i<n ; i++){
    for(int k = 0; k < iter; k++){
      dblvec out = calculate(i,parameters,data,0,2,extraData(i,k));
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

inline vector_matrix glmmr::calculator::jacobian_and_hessian(const dblvec& parameters){
  vector_matrix result(parameter_count);
  int n2d = parameter_count*(parameter_count + 1)/2;
  VectorXd H = VectorXd::Zero(n2d);
  VectorXd J = VectorXd::Zero(parameter_count);
  MatrixXd dat = MatrixXd::Zero(1,1);
  dblvec out = calculate(0,parameters,dat,0,2,0);
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

inline dblvec glmmr::calculator::calculate(const int i, 
                                           const dblvec& parameters, 
                                           const MatrixXd& data,
                                           const int j,
                                           int order,
                                           double extraData,
                                           int n){
  using enum Instruction;
  int idx_iter = 0;
  double a,b;
  std::stack<double> stack;
  std::vector<std::stack<double> > first_dx;
  std::vector<std::stack<double> > second_dx;
  if(order > 0){
    first_dx.resize(parameter_count);
  }
  if(order == 2){
    second_dx.resize(parameter_count*(parameter_count + 1)/2);
  }
  
  auto addZeroDx = [&] (){
    for(int idx = 0; idx < parameter_count; idx++){
      first_dx[idx].push(0.0);
    }
  };
  
  auto addZeroDx2 = [&] (){
    int index_count = 0;
    for(int idx = 0; idx < parameter_count; idx++){
      for(int jdx = idx; jdx < parameter_count; jdx++){
        second_dx[index_count].push(0.0);
        index_count++;
      }
    }
  };
  
  for(const auto& k: instructions){
    switch(k){
    case PushData:
  {
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
    if(idx_iter >= indexes.size())Rcpp::stop("Index out of range: case 0 idx iter: "+std::to_string(idx_iter)+" versus "+std::to_string(indexes.size()));
    if(indexes[idx_iter] >= data.cols())Rcpp::stop("Index out of range: case 0 indexes: "+std::to_string(indexes[idx_iter])+" versus "+std::to_string(data.cols()));
    if(i >= data.rows())Rcpp::stop("Row index out of range: case 0: "+std::to_string(i)+" versus "+std::to_string(data.rows()));
#endif
    
    stack.push(data(i,indexes[idx_iter]));
    if(order > 0){
      addZeroDx();
    }
    if(order == 2){
      addZeroDx2();
    }
    idx_iter++;
    break;
  }
    case PushCovData:
  {
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
    if(idx_iter >= indexes.size())Rcpp::stop("Index out of range: case 1 idx iter: "+std::to_string(idx_iter)+" versus "+std::to_string(indexes.size()));
    if(indexes[idx_iter] >= data.cols())Rcpp::stop("Index out of range: case 1 indexes: "+std::to_string(indexes[idx_iter])+" versus "+std::to_string(data.cols()));
#endif
    
    if(i==j){
      stack.push(0.0);
    } else {
      int i1 = i < j ? (n-1)*i - ((i-1)*i/2) + (j-i-1) : (n-1)*j - ((j-1)*j/2) + (i-j-1);
      stack.push(data(i1,indexes[idx_iter]));
    }
    if(order > 0){
      addZeroDx();
    }
    if(order == 2){
      addZeroDx2();
    }
    idx_iter++;
    break;
  }
    case PushParameter:
  {
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
    if((unsigned)idx_iter >= indexes.size())Rcpp::stop("Index out of range: case 2 idx iter: "+std::to_string(idx_iter)+" versus "+std::to_string(indexes.size()));
    if(indexes[idx_iter] >= parameter_count)Rcpp::stop("Index out of range: case 2 indexes: "+std::to_string(indexes[idx_iter])+" versus "+std::to_string(parameter_count));
    if(indexes[idx_iter] >= parameters.size())Rcpp::stop("Index out of range (pars): case 2 indexes: "+std::to_string(indexes[idx_iter])+" versus "+std::to_string(parameters.size()));
#endif
    
    stack.push(parameters[indexes[idx_iter]]);
    if(order > 0){
      for(int idx = 0; idx < parameter_count; idx++){
        if(idx == indexes[idx_iter]){
          first_dx[idx].push(1.0);
        } else {
          first_dx[idx].push(0.0);
        }
      }
    }
    if(order == 2){
      addZeroDx2();
    }
    idx_iter++;
    break;
  }
    case Add:
  {
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
    if(stack.size()<2)Rcpp::stop("Stack too small (3)");
#endif
    
    a = stack.top();
    stack.pop();
    b = stack.top();
    stack.pop();
    stack.push(a+b);
    if(order > 0){
      for(int idx = 0; idx < parameter_count; idx++){
        a = first_dx[idx].top();
        first_dx[idx].pop();
        b = first_dx[idx].top();
        first_dx[idx].pop();
        first_dx[idx].push(a+b);
      }
    }
    if(order == 2){
      int index_count = 0;
      for(int idx = 0; idx < parameter_count; idx++){
        for(int jdx = idx; jdx < parameter_count; jdx++){
          a = second_dx[index_count].top();
          second_dx[index_count].pop();
          b = second_dx[index_count].top();
          second_dx[index_count].pop();
          second_dx[index_count].push(a+b);
          index_count++;
        }
      }
    }
    break;
  }
    case Subtract:
  {
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
    if(stack.size()<2)Rcpp::stop("Stack too small (4)");
#endif
    
    a = stack.top();
    stack.pop();
    b = stack.top();
    stack.pop();
    stack.push(a-b);
    if(order > 0){
      for(int idx = 0; idx < parameter_count; idx++){
        a = first_dx[idx].top();
        first_dx[idx].pop();
        b = first_dx[idx].top();
        first_dx[idx].pop();
        first_dx[idx].push(a-b);
      }
    }
    if(order == 2){
      int index_count = 0;
      for(int idx = 0; idx < parameter_count; idx++){
        for(int jdx = idx; jdx < parameter_count; jdx++){
          a = second_dx[index_count].top();
          second_dx[index_count].pop();
          b = second_dx[index_count].top();
          second_dx[index_count].pop();
          second_dx[index_count].push(a-b);
          index_count++;
        }
      }
    }
    break;
  }
    case Multiply:
  {
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
    if(stack.size()<2)Rcpp::stop("Stack too small (5)");
#endif
    
    a = stack.top();
    stack.pop();
    b = stack.top();
    stack.pop();
    stack.push(a*b);
    if(order > 0){
      dblvec a_top_dx;
      dblvec b_top_dx;
      for(int idx = 0; idx < parameter_count; idx++){
        a_top_dx.push_back(first_dx[idx].top());
        first_dx[idx].pop();
        b_top_dx.push_back(first_dx[idx].top());
        first_dx[idx].pop();
        first_dx[idx].push(a*b_top_dx.back() + b*a_top_dx.back());
      }
      if(order == 2){
        int index_count = 0;
        for(int idx = 0; idx < parameter_count; idx++){
          for(int jdx = idx; jdx < parameter_count; jdx++){
            double adx2 = second_dx[index_count].top();
            second_dx[index_count].pop();
            double bdx2 = second_dx[index_count].top();
            second_dx[index_count].pop();
            double result = a*bdx2 + b*adx2 + a_top_dx[idx]*b_top_dx[jdx] + a_top_dx[jdx]*b_top_dx[idx];
            second_dx[index_count].push(result);
            index_count++;
          }
        }
      }
    }
    break;
  }
    case Divide:
  {
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
    if(stack.size()<2)Rcpp::stop("Stack too small (6)");
#endif
    
    a = stack.top();
    stack.pop();
    b = stack.top();
    
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
    if(b == 0)Rcpp::stop("Divide by zero (6)");
#endif
    
    stack.pop();
    stack.push(a/b);
    if(order > 0){
      dblvec a_top_dx;
      dblvec b_top_dx;
      for(int idx = 0; idx < parameter_count; idx++){
        a_top_dx.push_back(first_dx[idx].top());
        first_dx[idx].pop();
        b_top_dx.push_back(first_dx[idx].top());
        first_dx[idx].pop();
        double result = (b*a_top_dx.back() - a*b_top_dx.back())/(b*b);
        first_dx[idx].push(result);
      }
      if(order == 2){
        int index_count = 0;
        for(int idx = 0; idx < parameter_count; idx++){
          for(int jdx = idx; jdx < parameter_count; jdx++){
            double adx2 = second_dx[index_count].top();
            second_dx[index_count].pop();
            double bdx2 = second_dx[index_count].top();
            second_dx[index_count].pop();
            double result = (adx2*b - a_top_dx[idx]*b_top_dx[jdx]- a_top_dx[jdx]*b_top_dx[idx])/(b*b) - (a*bdx2*b - 2*b_top_dx[idx]*b_top_dx[jdx])/(b*b*b);
            second_dx[index_count].push(result);
            index_count++;
          }
        }
      }
    }
    break;
  }
    case Sqrt:
      
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (7)");
#endif
      
      a = stack.top();
      stack.pop();
      stack.push(sqrt(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = a==0 ? 0 : 0.5*pow(a,-0.5)*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = a==0? 0 : 0.5*pow(a,-0.5)*adx2 - 0.25*a_top_dx[idx]*a_top_dx[jdx]*pow(a,-3/2);
              second_dx[index_count].push(result);
              index_count++;
            }
          }
        }
      }
      break;
    case Power:
      {
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
        if(stack.size()<2)Rcpp::stop("Stack too small (8)");
#endif
        
        a = stack.top();
        stack.pop();
        b = stack.top();
        stack.pop();
        double out = pow(a,b);
        
#ifdef R_BUILD
        if(out != out)Rcpp::stop("Exponent fail: "+std::to_string(a)+"^"+std::to_string(b));
#endif
        
        stack.push(out);
        if(order > 0){
          dblvec a_top_dx;
          dblvec b_top_dx;
          for(int idx = 0; idx < parameter_count; idx++){
            a_top_dx.push_back(first_dx[idx].top());
            first_dx[idx].pop();
            b_top_dx.push_back(first_dx[idx].top());
            first_dx[idx].pop();
            double result = pow(a,b)*b_top_dx.back()*log(a) + pow(a,b-1)*b*a_top_dx.back();
            first_dx[idx].push(result);
          }
          if(order == 2){
            int index_count = 0;
            for(int idx = 0; idx < parameter_count; idx++){
              for(int jdx = idx; jdx < parameter_count; jdx++){
                double adx2 = second_dx[index_count].top();
                second_dx[index_count].pop();
                double bdx2 = second_dx[index_count].top();
                second_dx[index_count].pop();
                double result1 = first_dx[jdx].top()*b_top_dx[idx]*log(a) + stack.top()*(bdx2*log(a) + b_top_dx[idx]*(1/a)*a_top_dx[jdx]);
                double result2 = pow(a,b-1)*b_top_dx[jdx]*log(a) + pow(a,b-2)*(b-1)*a_top_dx[jdx];
                double result3 = result2*b*a_top_dx[idx] + pow(a,b-1)*(b*adx2+b_top_dx[jdx]*a_top_dx[idx]);
                second_dx[index_count].push(result1 + result3);
                index_count++;
              }
            }
          }
        }
        
        break;
      }
    case Exp:
      
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (9)");
#endif
      
      a = stack.top();
      stack.pop();
      stack.push(exp(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = stack.top()*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = stack.top()*(a_top_dx[idx]*a_top_dx[jdx] + adx2);
              second_dx[index_count].push(result);
              index_count++;
            }
          }
        }
      }
      break;
    case Negate:
      
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (10)");
#endif
      
      a = stack.top();
      stack.pop();
      stack.push(-1*a);
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = -1.0*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              second_dx[index_count].push(-1*adx2);
              index_count++;
            }
          }
        }
      }
      break;
    case Bessel:
      
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (11)");
#endif
      
      a = stack.top();
      stack.pop();
      b = boost::math::cyl_bessel_k(1,a);
      stack.push(b);
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = -0.5*boost::math::cyl_bessel_k(0,a)-0.5*boost::math::cyl_bessel_k(2,a);
          first_dx[idx].push(result*a_top_dx.back());
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = 0.25*boost::math::cyl_bessel_k(-1,a)+0.5*boost::math::cyl_bessel_k(1,a)+0.25*boost::math::cyl_bessel_k(3,a);
              double result1 = -0.5*boost::math::cyl_bessel_k(0,a)-0.5*boost::math::cyl_bessel_k(2,a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case Gamma:
      
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (12)");
#endif
      
      a = stack.top();
      stack.pop();
      stack.push(tgamma(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = stack.top()*boost::math::polygamma(0,a);
          first_dx[idx].push(result*a_top_dx.back());
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = stack.top()*boost::math::polygamma(0,a)*boost::math::polygamma(0,a) + stack.top()*boost::math::polygamma(1,a);
              double result1 = stack.top()*boost::math::polygamma(0,a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case Sin:
      
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (13)");
#endif
      
      a = stack.top();
      stack.pop();
      stack.push(sin(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = cos(a)*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = -1.0*sin(a);
              double result1 = cos(a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case Cos:
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (14)");
#endif
      
      a = stack.top();
      stack.pop();
      stack.push(cos(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = -1.0*sin(a)*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = -1.0*cos(a);
              double result1 = -1.0*sin(a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case BesselK:
      
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (15)");
#endif
      
      a = stack.top();
      stack.pop();
      b = stack.top();
      stack.pop();
      stack.push(boost::math::cyl_bessel_k(b,a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = -0.5*boost::math::cyl_bessel_k(b-1,a)-0.5*boost::math::cyl_bessel_k(b+1,a);
          first_dx[idx].push(result*a_top_dx.back());
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = 0.25*boost::math::cyl_bessel_k(b-2,a)+0.5*boost::math::cyl_bessel_k(b,a)+0.25*boost::math::cyl_bessel_k(b+2,a);
              double result1 = -0.5*boost::math::cyl_bessel_k(b-1,a)-0.5*boost::math::cyl_bessel_k(b+1,a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case Log:
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (16)");
#endif
      
      a = stack.top();
      stack.pop();
      stack.push(log(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = (1/a)*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = -1.0/(a*a);
              double result1 = 1/a;
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case Square:
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(stack.size()==0)Rcpp::stop("Stack too small (17)");
#endif
      
      a = stack.top();
      stack.pop();
      stack.push(a*a);
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = 2*a*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result2 = 2*a_top_dx[idx]*a_top_dx[jdx]+2*a*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case PushExtraData:
      {
        stack.push(extraData);
        if(order > 0){
          addZeroDx();
        }
        if(order == 2){
          addZeroDx2();
        }
        break;
      }
    case PushY:
      {
        stack.push(y[i]);
        if(order > 0){
          addZeroDx();
        }
        if(order == 2){
          addZeroDx2();
        }
        break;
      }
    case Int10:
      stack.push(10);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Int1:
      stack.push(1);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Int2:
      stack.push(2);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Int3:
      stack.push(3);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Int4:
      stack.push(4);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Int5:
      stack.push(5);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Int6:
      stack.push(6);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Int7:
      stack.push(7);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Int8:
      stack.push(8);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Int9:
      stack.push(9);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Pi:
      stack.push(M_PI);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Constant1:
      stack.push(0.3275911);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Constant2:
      stack.push(0.254829592);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Constant3:
      stack.push(-0.284496736);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Constant4:
      stack.push(1.421413741);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Constant5:
      stack.push(-1.453152027);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case Constant6:
      stack.push(1.061405429);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case LogFactorialApprox:
      {
        //log factorial approximation
        #if defined(ENABLE_DEBUG) && defined(R_BUILD)
        if(stack.size()==0)Rcpp::stop("Stack too small (40)");
        #endif
        
        a = stack.top();
        stack.pop();
        // Ramanujan approximation
        if(a == 0){
          stack.push(0);
        } else {
          double result = a*log(a) - a + log(a*(1+4*a*(1+2*a)))/6 + log(3.141593)/2;
          stack.push(result);
        }
        // NOTE: this function is only ever used in Poisson/binom log likelihood and so the top of the derivative
        // stacks should be 0, so we don't need to do anything. However, this should be updated if ever this
        // function is used more broadly.
        //if(order > 0){
        //  addZeroDx();
        //}
        //if(order == 2){
        //  addZeroDx2();
        //}
        break;
      }
    case PushVariance:
      {
        stack.push(variance(i));
        if(order > 0){
          addZeroDx();
        }
        if(order == 2){
          addZeroDx2();
        }
        break;
      }
    case PushUserNumber0:
      stack.push(numbers[0]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case PushUserNumber1:
      stack.push(numbers[1]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case PushUserNumber2:
      stack.push(numbers[2]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case PushUserNumber3:
      stack.push(numbers[3]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case PushUserNumber4:
      stack.push(numbers[4]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case PushUserNumber5:
      stack.push(numbers[5]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case PushUserNumber6:
      stack.push(numbers[6]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case PushUserNumber7:
      stack.push(numbers[7]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case PushUserNumber8:
      stack.push(numbers[8]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case PushUserNumber9:
      stack.push(numbers[9]);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    }
    
    #if defined(ENABLE_DEBUG) && defined(R_BUILD)
    if(stack.size() == 0)Rcpp::stop("Error stack empty!");
    if(stack.top() != stack.top() || isnan(stack.top()))Rcpp::stop("Calculation evaluates to NaN");
    #endif
  }
  
#if defined(ENABLE_DEBUG) && defined(R_BUILD)
  if(stack.size()>1)Rcpp::warning("More than one element on the stack at end of calculation");
#endif
  
  dblvec result;
  result.push_back(stack.top());
  if(order > 0){
    for(int idx = 0; idx < parameter_count; idx++){
      
      #if defined(ENABLE_DEBUG) && defined(R_BUILD)
      if(first_dx[idx].size()==0)Rcpp::stop("Error derivative stack empty");
      #endif
      
      result.push_back(first_dx[idx].top());
    }
  }
  if(order == 2){
    int index_count = 0;
    for(int idx = 0; idx < parameter_count; idx++){
      for(int jdx = idx; jdx < parameter_count; jdx++){
        
        #if defined(ENABLE_DEBUG) && defined(R_BUILD)
        if(second_dx[index_count].size()==0)Rcpp::stop("Error second derivative stack empty");
        #endif
        
        result.push_back(second_dx[index_count].top());
        index_count++;
      }
    }
  }
  
  return result;
}

inline void glmmr::calculator::print_instructions(){
  //currently only setup for R
  #ifdef R_BUILD
  using enum Instruction;
  int counter = 1;
  int idx_iter = 0;
  Rcpp::Rcout << "\nInstructions:\n";
  for(const auto& i: instructions){
    Rcpp::Rcout << counter << ". " << instruction_str.at(i);
    switch(i){
      case PushUserNumber0:
        Rcpp::Rcout << " = " << numbers[0] << "\n";
        break;
    case PushUserNumber1:
      Rcpp::Rcout << " = " << numbers[1] << "\n";
      break;
    case PushUserNumber2:
      Rcpp::Rcout << " = " << numbers[2] << "\n";
      break;
    case PushUserNumber3:
      Rcpp::Rcout << " = " << numbers[3] << "\n";
      break;
    case PushUserNumber4:
      Rcpp::Rcout << " = " << numbers[4] << "\n";
      break;
    case PushUserNumber5:
      Rcpp::Rcout << " = " << numbers[5] << "\n";
      break;
    case PushUserNumber6:
      Rcpp::Rcout << " = " << numbers[6] << "\n";
      break;
    case PushUserNumber7:
      Rcpp::Rcout << " = " << numbers[7] << "\n";
      break;
    case PushUserNumber8:
      Rcpp::Rcout << " = " << numbers[8] << "\n";
      break;
    case PushUserNumber9:
      Rcpp::Rcout << " = " << numbers[9] << "\n";
      break;
    case PushParameter:
      Rcpp::Rcout << ": " << parameter_names[indexes[idx_iter]] << "\n";
      idx_iter++;
      break;
    case PushData:
      Rcpp::Rcout << "(column " << indexes[idx_iter] << ")\n";
      idx_iter++;
      break;
    case PushCovData:
      Rcpp::Rcout << "(column " << indexes[idx_iter] << ")\n";
      idx_iter++;
      break;
    default:
      Rcpp::Rcout << "\n";
    }
    counter++;
  }
  #endif
}

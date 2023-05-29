#ifndef CALCULATOR_H
#define CALCULATOR_H

#include "general.h"

namespace glmmr {

class calculator {
  public:
    intvec instructions;
    intvec indexes;
    dblvec2d data;
    dblvec parameters;
    strvec parameter_names;
    int data_count = 0;
    int parameter_count = 0;
    bool any_nonlinear;
    
    calculator(){};
    
    void resize(int n){
      data.clear();
      data.resize(n);
    };
    
    double calculate(const int i, const int j = 0);
    
    //VectorXd calculate(int order = 0);
    
    //MatrixXd jacobian();
    
    //MatrixXd hessian();
    
};

}

inline double glmmr::calculator::calculate(const int i, const int j){
  int idx_iter = 0;
  double a,b,var;
  std::stack<double> stack;
  
  for(int k = 0; k < instructions.size(); k++){
    switch(instructions[k]){
    case 0:
      stack.push(data[i][indexes[idx_iter]]);
      idx_iter++;
      break;
    case 1:
      stack.push(data[j][indexes[idx_iter]]);
      idx_iter++;
      break;
    case 2:
      stack.push(parameters[indexes[idx_iter]]);
      idx_iter++;
      break;
    case 3:
      a = stack.top();
      stack.pop();
      b = stack.top();
      stack.pop();
      stack.push(a+b);
      break;
    case 4:
      a = stack.top();
      stack.pop();
      b = stack.top();
      stack.pop();
      stack.push(a-b);
      break;
    case 5:
      a = stack.top();
      stack.pop();
      b = stack.top();
      stack.pop();
      stack.push(a*b);
      break;
    case 6:
      a = stack.top();
      stack.pop();
      b = stack.top();
      stack.pop();
      stack.push(a/b);
      break;
    case 7:
      a = stack.top();
      stack.pop();
      stack.push(sqrt(a));
      break;
    case 8:
      {
        a = stack.top();
        stack.pop();
        b = stack.top();
        stack.pop();
        double out = pow(a,b);
        stack.push(out);
        break;
      }
    case 9:
      a = stack.top();
      stack.pop();
      stack.push(exp(a));
      break;
    case 10:
      a = stack.top();
      stack.pop();
      stack.push(-1*a);
      break;
    case 11:
      a = stack.top();
      stack.pop();
      b = R::bessel_k(a, 1, 1);
      stack.push(b);
      break;
    case 12:
      a = stack.top();
      stack.pop();
      stack.push(tgamma(a));
      break;
    case 13:
      a = stack.top();
      stack.pop();
      stack.push(sin(a));
      break;
    case 14:
      a = stack.top();
      stack.pop();
      stack.push(cos(a));
      break;
    case 15:
      a = stack.top();
      stack.pop();
      b = stack.top();
      stack.pop();
      stack.push(R::bessel_k(a, b, 1));
      break;
    case 16:
      a = stack.top();
      stack.pop();
      stack.push(log(a));
      break;
    case 20:
      stack.push(10);
      break;
    case 21:
      stack.push(1);
      break;
    case 22:
      stack.push(2);
      break;
    case 23:
      stack.push(3);
      break;
    case 24:
      stack.push(4);
      break;
    case 25:
      stack.push(5);
      break;
    case 26:
      stack.push(6);
      break;
    case 27:
      stack.push(7);
      break;
    case 28:
      stack.push(8);
      break;
    case 29:
      stack.push(9);
      break;
    case 30:
      stack.push(M_PI);
      break;
    }
    if(stack.size() == 0)Rcpp::stop("Error stack empty!");
    //var = stack.top();
    //Rcpp::Rcout << " | Top: " << var;
  }
  return stack.top();
}

#endif
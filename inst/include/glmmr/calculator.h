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

inline double glmmr::calculator::calculate(const int i, 
                                           const int j,
                                           int order = 0){
  if(order > 2)Rcpp::stop("Only up to second order derivatives allowed.")
  int idx_iter = 0;
  double a,b,var;
  std::stack<double> stack;
  // for higher order derivatives
  std::vector<std::stack<double> > first_dx;
  std::vector<std::stack<double> > second_dx;
  if(order > 0){
    first_dx.resize(parameter_count);
  }
  if(order == 2){
    second_dx.resize(parameter_count*(parameter_count + 1)/2);
  }
  
  for(int k = 0; k < instructions.size(); k++){
    switch(instructions[k]){
    case 0:
      {
        //push data (i)
        stack.push(data[i][indexes[idx_iter]]);
        if(order > 0){
          for(int i = 0; i < parameter_count; i++){
            first_dx[i].push(0.0);
          }
        }
        if(order == 2){
          int index_count = 0;
          for(int i = 0; i < parameter_count; i++){
            for(int j = i; j < parameter_count; j++){
              second_dx[index_count].push(0.0);
              index_count++;
            }
          }
        }
        idx_iter++;
        break;
      }
    case 1:
      {
        // push data (j)
        stack.push(data[j][indexes[idx_iter]]);
        if(order > 0){
          for(int i = 0; i < parameter_count; i++){
            first_dx[i].push(0.0);
          }
        }
        if(order == 2){
          int index_count = 0;
          for(int i = 0; i < parameter_count; i++){
            for(int j = i; j < parameter_count; j++){
              second_dx[index_count].push(0.0);
              index_count++;
            }
          }
        }
        idx_iter++;
        break;
      }
    case 2:
      {
        // push parameter
        stack.push(parameters[indexes[idx_iter]]);
        if(order > 0){
          for(int i = 0; i < parameter_count; i++){
            if(i == indexes[idx_iter]){
              first_dx[i].push(1.0);
            } else {
              first_dx[i].push(0.0);
            }
          }
        }
        if(order == 2){
          int index_count = 0;
          for(int i = 0; i < parameter_count; i++){
            for(int j = i; j < parameter_count; j++){
              second_dx[index_count].push(0.0);
              index_count++;
            }
          }
        }
        idx_iter++;
        break;
      }
    case 3:
      {
        // add
        a = stack.top();
        stack.pop();
        b = stack.top();
        stack.pop();
        stack.push(a+b);
        if(order > 0){
          for(int i = 0; i < parameter_count; i++){
            a = first_dx[i].top();
            first_dx[i].pop();
            b = first_dx[i].top();
            first_dx[i].pop();
            first_dx[i].push(a+b);
          }
        }
        if(order == 2){
          int index_count = 0;
          for(int i = 0; i < parameter_count; i++){
            for(int j = i; j < parameter_count; j++){
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
    case 4:
      {
        // subtract
        a = stack.top();
        stack.pop();
        b = stack.top();
        stack.pop();
        stack.push(a-b);
        if(order > 0){
          for(int i = 0; i < parameter_count; i++){
            a = first_dx[i].top();
            first_dx[i].pop();
            b = first_dx[i].top();
            first_dx[i].pop();
            first_dx[i].push(a-b);
          }
        }
        if(order == 2){
          int index_count = 0;
          for(int i = 0; i < parameter_count; i++){
            for(int j = i; j < parameter_count; j++){
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
    case 5:
      {
        // multiply
        a = stack.top();
        stack.pop();
        b = stack.top();
        stack.pop();
        stack.push(a*b);
        if(order > 0){
          dblvec a_top_dx;
          dblvec b_top_dx;
          for(int i = 0; i < parameter_count; i++){
            a_top_dx.push_back(first_dx[i].top());
            first_dx[i].pop();
            b_top_dx.push_back(first_dx[i].top());
            first_dx[i].pop();
            first_dx[i].push(a*b_top_dx.back() + b*a_top_dx.back());
          }
          if(order == 2){
            int index_count = 0;
            for(int i = 0; i < parameter_count; i++){
              for(int j = i; j < parameter_count; j++){
                double adx2 = second_dx[index_count].top();
                second_dx[index_count].pop();
                double bdx2 = second_dx[index_count].top();
                second_dx[index_count].pop();
                double result = a*bdx2 + b*adx2 + a_top_dx[i]*b_top_dx[j] + a_top_dx[j]*b_top_dx[i];
                second_dx[index_count].push(result);
                index_count++;
              }
            }
          }
        }
        break;
      }
    case 6:
      {
        //division
        a = stack.top();
        stack.pop();
        b = stack.top();
        stack.pop();
        stack.push(a/b);
        if(order > 0){
          dblvec a_top_dx;
          dblvec b_top_dx;
          for(int i = 0; i < parameter_count; i++){
            a_top_dx.push_back(first_dx[i].top());
            first_dx[i].pop();
            b_top_dx.push_back(first_dx[i].top());
            first_dx[i].pop();
            double result = (b*a_top_dx.back() - a*b_top_dx.back())/(b*b);
            first_dx[i].push(result);
          }
          if(order == 2){
            int index_count = 0;
            for(int i = 0; i < parameter_count; i++){
              for(int j = i; j < parameter_count; j++){
                double adx2 = second_dx[index_count].top();
                second_dx[index_count].pop();
                double bdx2 = second_dx[index_count].top();
                second_dx[index_count].pop();
                //a*bdx2 + b*adx2 + a_top_dx[i]*b_top_dx[j] + a_top_dx[j]*b_top_dx[i];
                double result = (adx2*b - a_top_dx[i]*b_top_dx[j]- a_top_dx[j]*b_top_dx[i])/(b*b) - (a*bdx2*b - 2*b_top_dx[i]*b_top_dx[j])/(b*b*b);
                second_dx[index_count].push(result);
                index_count++;
              }
            }
          }
        }
        break;
      }
    case 7:
      a = stack.top();
      stack.pop();
      stack.push(sqrt(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int i = 0; i < parameter_count; i++){
          a_top_dx.push_back(first_dx[i].top());
          first_dx[i].pop();
          double result = 0.5*pow(a,-0.5)*a_top_dx.back();
          first_dx[i].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int i = 0; i < parameter_count; i++){
            for(int j = i; j < parameter_count; j++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = 0.5*pow(a,-0.5)*adx2 - 0.25*a_top_dx[i]*a_top_dx[j]*pow(a,-3/2);
              second_dx[index_count].push(result);
              index_count++;
            }
          }
        }
      }
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
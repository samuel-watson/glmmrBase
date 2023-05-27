#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "general.h"

namespace glmmr {

inline intvec interpret_re(const std::string& fn,
                                  const intvec& A){
  intvec B;
  switch(string_to_case.at(fn)){
  case 1:
    B = {2,2,5};
    break;
  case 2:
    B.insert(B.end(), A.begin(), A.end());
    B.push_back(2);
    B.push_back(8);
    break;
  case 3:
     {
      const intvec C = {6,10,9};
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C.begin(), C.end());
      break;
     }
  case 4:
    {
       const intvec C = {6,10,9,2,2,5,5};
       B.push_back(2);
       B.insert(B.end(), A.begin(), A.end());
       B.insert(B.end(), C.begin(), C.end());
       break;
    }
  case 5:
    {
      const intvec C1 = {2,2,5};
      const intvec C2 = {5,6,10,9};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      break;
    }
  case 6:
    {
      const intvec C1 = {2,2,5};
      const intvec C2 = {5,6,10,9,2,2,5,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      break;
    }
  case 7:
    {
      const intvec C = {6,11};
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C.begin(), C.end());
      break;
    }
  case 8:
    {
      const intvec C1 = {2,12,2,21,4,22,8,6,22,2,5,7,2};
      const intvec C2 = {6,5,2,8,5,2,22,2,5,7,2};
      const intvec C3 = {6,5,15,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      break;
    }
  case 9:
    {
      const intvec C = {21,4,8,5};
      B.push_back(2);
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C.begin(), C.end());
      break;
    }
  case 10:
    {
      const intvec C1 = {2,2,21,3};
      const intvec C2 = {5,21,3,5,2,21,3};
      const intvec C3 = {21,4,8,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      break;
    }
  case 11:
    {
      const intvec C1 = {2,21};
      const intvec C2 = {22,2,3,5,3,23,21,4,21,2,22,3,2,22,3,5,4,5};
      const intvec C3 = {5,5,3,5,2,22,3};
      const intvec C4 = {21,4,8,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C4.begin(), C4.end());
      break;
    }
  case 12:
    {
      const intvec C1 = {2,2,12,2,21,4,22,8,6,5,2};
      const intvec C2 = {8,5,2};
      const intvec C3 = {15,5};
      const intvec C4 = {5,20,20,5,20,27,3,3,4,5,22,20,21,3,6};
      const intvec C5 = {5,3,21,3,5};
      const intvec C6 = {21,4,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C4.begin(), C4.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C5.begin(), C5.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C6.begin(), C6.end());
      break;
    }
  case 13:
    {
      const intvec C1 = {8,21,4,23,10,8,2,5};
      const intvec C2 = {30,5,14};
      const intvec C3 = {21,4,5};
      const intvec C4 = {30,13,30,21,6,5,3,5};
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      B.insert(B.end(), A.begin(), A.end());
      break;
    }
  case 14:
    {
      const intvec C1 = {8,10,9,2,22,30};
      const intvec C2 = {5,5,22,30};
      const intvec C3 = {5,5,13,6};
      const intvec C4 = {21,4,5,22,30};
      const intvec C5 = {5,5,22,30};
      const intvec C6 = {5,5,14,21,4,6,30,21,6,5,3,5};
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C4.begin(), C4.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C5.begin(), C5.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C6.begin(), C6.end());
      break;
    }
  }
  return B;
}

//add in the indexes for each function
inline intvec interpret_re_par(const std::string& fn,
                               const intvec& col_idx,
                               const intvec& par_idx){
  intvec B;
  
  
  auto addA = [&] (){
    for(int i = 0; i<col_idx.size();i++){
      B.push_back(col_idx[i]);
      B.push_back(col_idx[i]);
      B.push_back(col_idx[i]);
      B.push_back(col_idx[i]);
    }
  };
  
  auto addPar2 = [&] (int i){
    B.push_back(par_idx[i]);
    B.push_back(par_idx[i]);
  };
  
  
  switch(string_to_case.at(fn)){
  case 1:
    addPar2(0);
    break;
  case 2: case 3: case 7:
    B.push_back(par_idx[0]);
    addA();
    break;
  case 4:
    B.push_back(par_idx[1]);
    addA();
    addPar2(0);
    break;
  case 5:
    addPar2(0);
    addA();
    addA();
    break;
  case 6:
    addPar2(1);
    addA();
    addA();
    addPar2(0);
    break;
  case 8:
    addPar2(0);
    addPar2(0);
    B.push_back(par_idx[1]);
    addA();
    addPar2(0);
    B.push_back(par_idx[0]);
    B.push_back(par_idx[1]);
    addA();
    break;
  case 9:
    B.push_back(par_idx[0]);
    B.push_back(par_idx[1]);
    addA();
    break;
  case 10:
    addPar2(0);
    B.push_back(par_idx[1]);
    addA();
    break;
  case 11:
    B.push_back(par_idx[0]);
    addA();
    addPar2(1);
    addA();
    addA();
    B.push_back(par_idx[1]);
    addA();
    break;
  case 12:
    B.push_back(par_idx[0]);
    addPar2(1);
    addPar2(0);
    addA();
    addA();
    addA();
    addA();
    addA();
    addA();
    break;
  case 13:
    B.push_back(par_idx[1]);
    addA();
    B.push_back(par_idx[0]);
    addA();
    addA();
    addA();
    break;
  case 14:
    B.push_back(par_idx[1]);
    addA();
    B.push_back(par_idx[0]);
    addA();
    addA();
    addA();
    addA();
    addA();
    break;
  }
  return B;
}

inline double calculate(const intvec& instructions,
                        const intvec& indexes,
                        const dblvec& parameters,
                        const dblvec2d& data,
                        const int& i,
                        const int& j){
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

}

#endif
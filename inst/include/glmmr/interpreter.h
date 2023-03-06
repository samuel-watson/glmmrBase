#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "general.h"

namespace glmmr {

inline intvec interpret_re(const std::string& fn,
                                  const intvec& A){
  intvec B;
  const static std::unordered_map<std::string,int> string_to_case{
    {"gr",1},
    {"ar1",2}
  };
  switch(string_to_case.at(fn)){
  case 1:
    B = {2,2,5};
    break;
  case 2:
    B.push_back(2);
    B.insert(B.end(), A.begin(), A.end());
    B.push_back(8);
    break;
  }
  return B;
}

inline intvec interpret_re_par(const std::string& fn,
                               const intvec& col_idx,
                               const intvec& par_idx){
  intvec B;
  const static std::unordered_map<std::string,int> string_to_case{
    {"gr",1},
    {"ar1",2}
  };
  switch(string_to_case.at(fn)){
  case 1:
    B = {par_idx[0],par_idx[0]};
    break;
  case 2:
    B.push_back(par_idx[0]);
    for(int i = 0; i<col_idx.size();i++){
      B.push_back(col_idx[i]);
      B.push_back(col_idx[i]);
      B.push_back(col_idx[i]);
      B.push_back(col_idx[i]);
    }
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
  dblvec stack;
  for(int k = 0; k < instructions.size(); k++){

    switch(instructions[k]){
    case 0:
      stack.insert(stack.begin(),data[i][indexes[idx_iter]]);
      idx_iter++;
      break;
    case 1:
      stack.insert(stack.begin(),data[j][indexes[idx_iter]]);
      idx_iter++;
      break;
    case 2:
      stack.insert(stack.begin(),parameters[indexes[idx_iter]]);
      idx_iter++;
      break;
    case 3:
      stack[0] += stack[1];
      stack.erase(std::next(stack.begin()));
      break;
    case 4:
      stack[0] -= stack[1];
      stack.erase(std::next(stack.begin()));
      break;
    case 5:
      stack[0] *= stack[1];
      stack.erase(std::next(stack.begin()));
      break;
    case 6:
      stack[0] *= 1/stack[1];
      stack.erase(std::next(stack.begin()));
      break;
    case 7:
      stack[0] = sqrt(stack[0]);
      break;
    case 8:
      stack[0] = pow(stack[1],stack[0]);
      stack.erase(std::next(stack.begin()));
      break;
    case 9:
      stack[0] = exp(stack[0]);
      break;
    }
  }
  return stack[0];
}

template<typename T>
inline void print_vec_1d(const T& vec){
  Rcpp::Rcout << "\n[1]: ";
  for(auto j: vec) Rcpp::Rcout << j << " ";
}

template<typename T>
inline void print_vec_2d(const T& vec){
  for(int i = 0; i < vec.size(); i++){
    Rcpp::Rcout << "\n[" << i << "]: ";
    for(auto j: vec[i]) Rcpp::Rcout << j << " ";
  }
}

template<typename T>
inline void print_vec_3d(const T& vec){
  for(int i = 0; i < vec.size(); i++){
    Rcpp::Rcout << "\n[" << i << "]:";
    for(int k = 0; k < vec[i].size(); k++){
      Rcpp::Rcout << "\n   [" << i <<","<< k << "]: ";
      for(auto j: vec[i][k]) Rcpp::Rcout << j << " ";
    }
  }
}

}

#endif
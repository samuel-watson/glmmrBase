#ifndef GENERAL_H
#define GENERAL_H

#include <RcppEigen.h>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cmath>
#include <stack>
#include "algo.h"

typedef std::string str;
typedef std::vector<str> strvec;
typedef std::vector<int> intvec;
typedef std::vector<double> dblvec;
typedef std::vector<strvec> strvec2d;
typedef std::vector<dblvec> dblvec2d;
typedef std::vector<intvec> intvec2d;
typedef std::vector<dblvec2d> dblvec3d;
typedef std::vector<intvec2d> intvec3d;


namespace glmmr {
const static std::unordered_map<str, double> nvars = {  
  {"gr", 1},
  {"ar1", 1},
  {"fexp0", 1},
  {"fexp", 2},
  {"sqexp0",1},
  {"sqexp",2},
  {"bessel",1},
  {"matern",2},
  {"wend0",2},
  {"wend1",2},
  {"wend2",2},
  {"prodwm",2},
  {"prodcb",2},
  {"prodek",2}
};

const static std::unordered_map<str,int> string_to_case{
  {"gr",1},
  {"ar1",2},
  {"fexp0", 3},
  {"fexp", 4},
  {"sqexp0",5},
  {"sqexp",6},
  {"bessel",7},
  {"matern",8},
  {"wend0",9},
  {"wend1",10},
  {"wend2",11},
  {"prodwm",12},
  {"prodcb",13},
  {"prodek",14}
};

const static intvec xvar_rpn = {0,1,4,0,1,4,5};

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
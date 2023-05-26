#ifndef FORMULA_HPP
#define FORMULA_HPP

#include <RcppEigen.h>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cmath>
#include "general.h"
#include "interpreter.h"

namespace glmmr{

class Formula {
public:
  str formula_;
  strvec fe_;
  strvec re_;
  strvec z_;
  intvec re_order_;
  bool RM_INT;

  Formula(const str& formula) : 
    formula_(formula) {
    tokenise();
  };
  
  Formula(const glmmr::Formula& formula) : formula_(formula.formula_){
    tokenise();
  };
  
  Formula& operator= (const glmmr::Formula& formula){
    formula_ = formula.formula_;
    tokenise();
    return *this;
  };

  void tokenise();
  
  void formula_validate();
   
  strvec re(){
    return re_;
  }

  strvec fe(){
    return fe_;
  }

  strvec z(){
    return z_;
  }

  strvec re_terms(){
    return re_terms_;
  }
  
private:
  //strvec tokens_;
  strvec re_terms_;
};

}

#include "formula.ipp"

#endif
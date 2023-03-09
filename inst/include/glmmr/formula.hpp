#ifndef FORMULA_HPP
#define FORMULA_HPP

#include "general.h"
#include "interpreter.h"

namespace glmmr{

class Formula {
public:
  std::string formula_;
  strvec fe_;
  strvec re_;
  strvec z_;
  intvec re_order_;
  bool RM_INT;

  Formula(const std::string& formula) : 
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
  strvec tokens_;
  strvec re_terms_;
};

}

#include "formula.ipp"

#endif
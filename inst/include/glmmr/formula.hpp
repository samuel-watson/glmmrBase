#ifndef FORMULA_HPP
#define FORMULA_HPP

#include "general.h"
#include "interpreter.h"
#include "calculator.hpp"
#include "formulaparse.h"

namespace glmmr{

class Formula {
public:
  str formula_;
  std::vector<char> linear_predictor_;
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
  
  void calculate_linear_predictor(glmmr::calculator& calculator,
                                  const ArrayXXd& data,
                                  const strvec& colnames,
                                  MatrixXd& Xdata){
    //calculator.resize(data.rows());
    bool outparse = glmmr::parse_formula(linear_predictor_,
                                          calculator,
                                          data,
                                          colnames,
                                          Xdata);
    std::reverse(calculator.instructions.begin(),calculator.instructions.end());
    std::reverse(calculator.indexes.begin(),calculator.indexes.end());
  }
   
  strvec re(){
    return re_;
  }

  // strvec fe(){
  //   return fe_;
  // }

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
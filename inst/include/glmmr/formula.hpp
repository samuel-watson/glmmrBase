#pragma once

#include "general.h"
#include "interpreter.h"
#include "calculator.hpp"

namespace glmmr{

bool check_data(str& formula,
                       calculator& calc,
                       const ArrayXXd& data,
                       const strvec& colnames,
                       MatrixXd& Xdata,
                       bool push = true,
                       bool add_data = true);

bool check_parameter(str& token_as_str,
                            calculator& calc,
                            bool bracket_flag = false);

void add_factor(std::vector<char>& s2,
                       calculator& calc,
                       const ArrayXXd& data,
                       const strvec& colnames,
                       MatrixXd& Xdata,
                       bool add_data = true);

void sign_fn(std::vector<char>& formula,
                    calculator& calc,
                    const ArrayXXd& data,
                    const strvec& colnames,
                    MatrixXd& Xdata,
                    int type,
                    bool add_data = true);

void two_way_fn(std::vector<char>& formula,
                       calculator& calc,
                       const ArrayXXd& data,
                       const strvec& colnames,
                       MatrixXd& Xdata,
                       int type,
                       bool add_data = true);

bool parse_formula(std::vector<char>& formula,
                          calculator& calc,
                          const ArrayXXd& data,
                          const strvec& colnames,
                          MatrixXd& Xdata,
                          bool bracket_flag = false,
                          bool add_data = true);

class Formula {
public:
  str                 formula_;
  std::vector<char>   linear_predictor_;
  strvec              re_;
  strvec              z_;
  intvec              re_order_;
  bool                RM_INT;
  strvec              fe_parameter_names_;
  Formula(const str& formula) : formula_(formula) {tokenise();};
  Formula(const Formula& formula) : formula_(formula.formula_), fe_parameter_names_(formula.fe_parameter_names_) {tokenise();};
  Formula& operator= (const Formula& formula){
    formula_ = formula.formula_;
    fe_parameter_names_ = formula.fe_parameter_names_;
    tokenise();
    return *this;
  };
  void    tokenise();
  void    formula_validate();
  void    calculate_linear_predictor(calculator& calculator,const ArrayXXd& data,const strvec& colnames, MatrixXd& Xdata);
  strvec  re() const;
  strvec  z() const;
  strvec  re_terms() const;
private:
  strvec  re_terms_;
};
}



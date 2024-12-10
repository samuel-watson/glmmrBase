#pragma once

#include "general.h"
#include "openmpheader.h"
#include "instructions.h"

enum class CalcDyDx {
  None,
  BetaFirst,
  BetaSecond,
  XBeta,
  Zu
};

namespace glmmr {

class calculator {
public:
  std::vector<Do>       instructions; // vector of insructions to execute
  intvec                indexes; // indexes of data or parameter vectors
  dblvec                y;  // outcome data
  std::array<double,20> numbers = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  strvec                parameter_names; //vector of parameter names
  strvec                data_names; // vector of data names
  ArrayXd               variance = ArrayXd::Constant(1,1.0); // variance values
  int                   data_count = 0; // number of data items
  int                   parameter_count = 0; // number of parameters
  int                   user_number_count = 0; // number of numbers in the function
  int                   data_size = 0; // number of data items in the calculation
  bool                  any_nonlinear = false; // for linear predictor - any non-linear functions?
  MatrixXd              data = MatrixXd::Zero(1,1); // the data for the calculation
  dblvec                parameters;
  intvec                parameter_indexes;
  calculator() {};
  
  template<CalcDyDx dydx>
  dblvec calculate(const int i, 
                   const int j = 0,
                   const int parameterIndex = 0,
                   const double extraData = 0.0) const;
  
  calculator& operator= (const glmmr::calculator& calc);
  VectorXd      linear_predictor();
  MatrixXd      jacobian();
  MatrixXd      jacobian(const VectorXd& extraData);
  MatrixXd      jacobian(const MatrixXd& extraData);
  MatrixMatrix  jacobian_and_hessian(const MatrixXd& extraData);
  VectorMatrix  jacobian_and_hessian();
  void          update_parameters(const dblvec& parameters_in);
  double        get_covariance_data(const int i, const int j, const int fn);
  void          print_instructions() const;
  void          print_names(bool print_data = true, bool print_parameters = false) const;
};

}


#ifndef CALCULATOR_HPP
#define CALCULATOR_HPP

#include "general.h"


namespace glmmr {

class calculator {
  public:
    intvec instructions;
    intvec indexes;
    dblvec y;
    strvec parameter_names;
    double var_par = 0;
    int data_count = 0;
    int parameter_count = 0;
    bool any_nonlinear = false;
    
    calculator() {};
    
    dblvec calculate(const int i, 
                     const dblvec& parameters, 
                     const MatrixXd& data,
                     const int j = 0,
                     int order = 0, 
                     double extraData = 0.0);
    
    calculator& operator= (const glmmr::calculator& calc);
    VectorXd linear_predictor(const dblvec& parameters,const MatrixXd& data);
    VectorXd first_derivative(int i,const dblvec& parameters,const MatrixXd& data,double extraData = 0);
    MatrixXd second_derivative(int i,const dblvec& parameters,const MatrixXd& data, double extraData = 0);
    MatrixXd jacobian(const dblvec& parameters, const MatrixXd& data);
    MatrixXd jacobian(const dblvec& parameters,const MatrixXd& data,const VectorXd& extraData);
    MatrixXd jacobian(const dblvec& parameters,const MatrixXd& data,const MatrixXd& extraData);
    matrix_matrix jacobian_and_hessian(const dblvec& parameters,const MatrixXd& data,const MatrixXd& extraData);
    vector_matrix jacobian_and_hessian(const dblvec& parameters);

};

}


#include "calculator.ipp"

#endif
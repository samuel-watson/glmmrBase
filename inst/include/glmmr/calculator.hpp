#ifndef CALCULATOR_HPP
#define CALCULATOR_HPP

#include "general.h"

namespace glmmr {

class calculator {
  public:
    intvec instructions;
    intvec indexes;
    dblvec2d data;
    dblvec parameters;
    strvec parameter_names;
    double var_par = 0;
    int data_count = 0;
    int parameter_count = 0;
    bool any_nonlinear = false;
    
    calculator(){};
    
    void resize(int n){
      data.clear();
      data.resize(n);
    };
    
    dblvec calculate(const int i, const int j = 0,int order = 0, double extraData = 0.0);
    
    calculator& operator= (const glmmr::calculator& calc){
      instructions = calc.instructions;
      indexes = calc.indexes;
      data = calc.data;
      parameters = calc.parameters;
      parameter_names = calc.parameter_names;
      var_par = calc.var_par;
      data_count = calc.data_count;
      parameter_count = calc.parameter_count;
      any_nonlinear = calc.any_nonlinear;
      return *this;
    };
    
    VectorXd linear_predictor(){
      int n = data.size();
      if(n==0)Rcpp::stop("No data");
      VectorXd x(n);
#pragma omp parallel for
      for(int i = 0; i < n; i++){
        x(i) = calculate(i)[0];
      }
      return x;
    };
    
    VectorXd first_derivative(int i, double extraData = 0){
      dblvec out = calculate(i,0,1,extraData);
      VectorXd d = Map<VectorXd, Unaligned>(out.data()+1, out.size()-1);
      return d;
    };
    
    MatrixXd second_derivative(int i, double extraData = 0){
      dblvec out = calculate(i,0,2, extraData);
      MatrixXd h(parameter_count, parameter_count);
      int index_count = parameter_count+1;
      for(int j = 0; j < parameter_count; j++){
        for(int k = j; k < parameter_count; k++){
          h(k,j) = out[index_count];
          if(j != k) h(j,k) = h(k,j);
          index_count++;
        }
      }
      return h;
    };
    
    MatrixXd jacobian(){
      int n = data.size();
      if(n==0)Rcpp::stop("No data initialised in calculator");
      MatrixXd J(n,parameter_count);
      J.setZero();
#pragma omp parallel for
      for(int i = 0; i<n ; i++){
        J.row(i) = (first_derivative(i)).transpose();
      }
      return J;
    };
    
    MatrixXd hessian(){
      int n = data.size();
      if(n==0)Rcpp::stop("No data initialised in calculator");
      int n2d = parameter_count*(parameter_count + 1)/2;
      MatrixXd H(n2d,n);
#pragma omp parallel for
      for(int i = 0; i<n ; i++){
        dblvec out = calculate(i,0,2);
        for(int j = 0; j < n2d; j++){
          H(j,i) = out[parameter_count + 1 + j];
        }
      }
      VectorXd Hmean = H.rowwise().mean();
      MatrixXd H0 = MatrixXd::Zero(parameter_count, parameter_count);
      int index_count = 0;
      for(int j = 0; j < parameter_count; j++){
        for(int k = j; k < parameter_count; k++){
          H0(k,j) = Hmean[index_count];
          if(j != k) H0(j,k) = H0(k,j);
          index_count++;
        }
      }
      return H0;
    };
    
    MatrixXd hessian(const VectorXd& extraData){
      int n = data.size();
      if(n==0)Rcpp::stop("No data initialised in calculator");
      if(extraData.size()!=n)Rcpp::stop("Extra data not of length n");
      int n2d = parameter_count*(parameter_count + 1)/2;
      MatrixXd H(n2d,n);
#pragma omp parallel for
      for(int i = 0; i<n ; i++){
        dblvec out = calculate(i,0,2,extraData(i));
        for(int j = 0; j < n2d; j++){
          H(j,i) = out[parameter_count + 1 + j];
        }
      }
      VectorXd Hmean = H.rowwise().mean();
      MatrixXd H0 = MatrixXd::Zero(parameter_count, parameter_count);
      int index_count = 0;
      for(int j = 0; j < parameter_count; j++){
        for(int k = j; k < parameter_count; k++){
          H0(k,j) = Hmean[index_count];
          if(j != k) H0(j,k) = H0(k,j);
          index_count++;
        }
      }
      return H0;
    };
    
    MatrixXd hessian(const MatrixXd& extraData){
      int n = data.size();
      if(n==0)Rcpp::stop("No data initialised in calculator");
      if(extraData.rows()!=n)Rcpp::stop("Extra data not of length n");
      int iter = extraData.cols();
      int n2d = parameter_count*(parameter_count + 1)/2;
      MatrixXd H = MatrixXd::Zero(n2d,n);
#pragma omp parallel for
      for(int i = 0; i<n ; i++){
        for(int k = 0; k < iter; k++){
          dblvec out = calculate(i,0,2,extraData(i,k));
          for(int j = 0; j < n2d; j++){
            H(j,i) += out[parameter_count + 1 + j]/iter;
          } 
        }
      }
      VectorXd Hmean = H.rowwise().sum();
      MatrixXd H0 = MatrixXd::Zero(parameter_count, parameter_count);
      int index_count = 0;
      for(int j = 0; j < parameter_count; j++){
        for(int k = j; k < parameter_count; k++){
          H0(k,j) = Hmean(index_count);
          if(j != k) H0(j,k) = H0(k,j);
          index_count++;
        }
      }
      return H0;
    };
    
    matrix_matrix jacobian_and_hessian(const MatrixXd& extraData){
      matrix_matrix result(parameter_count,data.size(),parameter_count,parameter_count);
      int n = data.size();
      if(n==0)Rcpp::stop("No data initialised in calculator");
      if(extraData.rows()!=n)Rcpp::stop("Extra data not of length n");
      int iter = extraData.cols();
      int n2d = parameter_count*(parameter_count + 1)/2;
      MatrixXd H = MatrixXd::Zero(n2d,n);
      MatrixXd J = MatrixXd::Zero(parameter_count,n);
#pragma omp parallel for
      for(int i = 0; i<n ; i++){
        dblvec out;
        for(int k = 0; k < iter; k++){
          out = calculate(i,0,2,extraData(i,k));
          for(int j = 0; j < parameter_count; j++){
            J(j,i) += out[1+j]/iter;
          }
          for(int j = 0; j < n2d; j++){
            H(j,i) += out[parameter_count + 1 + j]/iter;
          }
        }
      }
      VectorXd Hmean = H.rowwise().sum();
      //VectorXd Jmean = J.rowwise().mean();
      MatrixXd H0 = MatrixXd::Zero(parameter_count, parameter_count);
      int index_count = 0;
      for(int j = 0; j < parameter_count; j++){
        for(int k = j; k < parameter_count; k++){
          H0(k,j) = Hmean[index_count];
          if(j != k) H0(j,k) = H0(k,j);
          index_count++;
        }
      }
      result.mat1 = H0;
      result.mat2 = J;
      return result;
    };
    
};

}

#include "calculator.ipp"

#endif
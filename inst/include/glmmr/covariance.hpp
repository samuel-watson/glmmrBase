#ifndef COVARIANCE_HPP
#define COVARIANCE_HPP

#include "general.h"
#include "interpreter.h"
#include "formula.hpp"

namespace glmmr {

class Covariance {
public:
  const Eigen::ArrayXXd data_;
  const strvec colnames_;
  dblvec parameters_;
  dblvec other_pars_;
  
  Covariance(const glmmr::Formula& form,
                                const Eigen::ArrayXXd &data,
                                const strvec& colnames) : 
    data_(data), colnames_(colnames), form_(form), Q_(0) {
    parse();
  };
  
  Covariance(const glmmr::Formula& form,
                                const Eigen::ArrayXXd &data,
                                const strvec& colnames,
                                const dblvec& parameters) : 
    data_(data), colnames_(colnames), parameters_(parameters),form_(form),  Q_(0) {
    parse();
  };
  
  Covariance(const glmmr::Formula& form,
                                const Eigen::ArrayXXd &data,
                                const strvec& colnames,
                                const Eigen::ArrayXd& parameters) : 
    data_(data), colnames_(colnames),form_(form),  Q_(0) {
    update_parameters(parameters);
    parse();
  };
  
  void update_parameters(const dblvec& parameters){
    parameters_ = parameters;
  };
  
  void update_parameters(const Eigen::ArrayXd& parameters){
    if(parameters_.size()==0){
      for(int i = 0; i < parameters.size(); i++){
        parameters_.push_back(parameters(i));
      }
    } else if(parameters_.size() == parameters.size()){
      for(int i = 0; i < parameters.size(); i++){
        parameters_[i] = parameters(i);
      }
    } else {
      Rcpp::stop("Wrong number of parameters");
    }
  };
  
  void parse();
  
  double get_val(int b, int i, int j);
  
  Eigen::MatrixXd get_block(int b);
  
  Eigen::MatrixXd get_chol_block(int b,bool upper = false);
  
  Eigen::MatrixXd Z();
  
  Eigen::MatrixXd D(bool chol = false,
                    bool upper = false){
    return D_builder(0,chol,upper);
  };
  
  int npar(){
    return npars_;
  };
  
  int B(){
    return B_;
  }
  
  Eigen::VectorXd sim_re();
  
  int Q(){
    if(Q_==0)Rcpp::stop("Random effects not initialised");
    return Q_;
  }
  
private:
  const glmmr::Formula& form_;
  intvec z_;
  dblvec3d re_data_;
  intvec3d re_cols_;
  dblvec3d z_data_;
  strvec2d fn_;
  intvec re_order_;
  intvec3d re_pars_;
  intvec2d re_rpn_;
  intvec2d re_index_;
  int Q_;
  int n_;
  int B_;
  int npars_;
  
  Eigen::MatrixXd D_builder(int b,
                            bool chol = false,
                            bool upper = false){
    if (b == B_ - 1) {
      return chol ? get_chol_block(b,upper) : get_block(b);
    }
    else {
      Eigen::MatrixXd mat1 = chol ? get_chol_block(b,upper) : get_block(b);
      Eigen::MatrixXd mat2;
      if (b == B_ - 2) {
        mat2 = chol ? get_chol_block(b+1,upper) : get_block(b+1);
      }
      else {
        mat2 = D_builder(b + 1, chol, upper);
      }
      int n1 = mat1.rows();
      int n2 = mat2.rows();
      Eigen::MatrixXd dmat = Eigen::MatrixXd::Zero(n1+n2, n1+n2);
      dmat.block(0,0,n1,n1) = mat1;
      dmat.block(n1, n1, n2, n2) = mat2;
      return dmat;
    }
  }
};

}

#include "covariance.ipp"

#endif
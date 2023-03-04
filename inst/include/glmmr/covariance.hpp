#ifndef COVARIANCE_HPP
#define COVARIANCE_HPP

#define _USE_MATH_DEFINES

#include "general.h"
#include "interpreter.h"
#include "formula.hpp"

namespace glmmr {

class Covariance {
public:
  glmmr::Formula form_;
  const Eigen::ArrayXXd data_;
  const strvec colnames_;
  dblvec parameters_;
  dblvec other_pars_;
  
  Covariance(const std::string& formula,
             const Eigen::ArrayXXd &data,
             const strvec& colnames) : 
    form_(formula), data_(data), colnames_(colnames), Q_(0),
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()) {};
  
  Covariance(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames) : 
    form_(form), data_(data), colnames_(colnames), Q_(0),
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()) {};
  
  Covariance(const std::string& formula,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) : 
    form_(formula), data_(data), colnames_(colnames), parameters_(parameters), 
    Q_(0),size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()) {};
  
  Covariance(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) : 
    form_(form), data_(data), colnames_(colnames), parameters_(parameters), 
    Q_(0),size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()) {};
  
  Covariance(const std::string& formula,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const Eigen::ArrayXd& parameters) : 
    form_(formula), data_(data), colnames_(colnames),Q_(0),
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()) {
    update_parameters(parameters);
  };
  
  Covariance(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
              const strvec& colnames,
              const Eigen::ArrayXd& parameters) : 
    form_(form), data_(data), colnames_(colnames),Q_(0),
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()) {
    update_parameters(parameters);
  };
  
  void update_parameters(const dblvec& parameters){
    parameters_ = parameters;
  };
  
  //external version for Rcpp module
  void update_parameters_extern(const dblvec& parameters){
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
    Eigen::MatrixXd D = D_builder(0,chol,upper);
    return D;
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
  
  int max_block_dim(){
    int max = 0;
    for(int i = 0; i<re_data_.size(); i++){
      if(re_data_[i].size() > max)max = re_data_[i].size();
    }
    return max;
  }
  
  double log_likelihood(const Eigen::VectorXd &u);
  
  double log_determinant();
  
private:
  intvec z_;
  dblvec3d re_data_;
  intvec3d re_cols_;
  dblvec3d z_data_;
  strvec2d fn_;
  intvec re_order_;
  intvec3d re_pars_;
  intvec2d re_rpn_;
  intvec2d re_index_;
  intvec2d re_obs_index_;
  int Q_;
  int n_;
  int B_;
  int npars_;
  Eigen::ArrayXd size_B_array;
  Eigen::MatrixXd dmat_matrix;
  Eigen::VectorXd zquad;
  
  Eigen::MatrixXd D_builder(int b,
                            bool chol = false,
                            bool upper = false);
  
 
};

}

#include "covariance.ipp"

#endif
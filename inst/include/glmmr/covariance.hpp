#ifndef COVARIANCE_HPP
#define COVARIANCE_HPP

#define _USE_MATH_DEFINES

#include <SparseChol.h>
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
    zquad(max_block_dim()), mat(intvec({0,1})) {};

  Covariance(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames) :
    form_(form), data_(data), colnames_(colnames), Q_(0),
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()), mat(intvec({0,1})) {};

  Covariance(const std::string& formula,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) :
    form_(formula), data_(data), colnames_(colnames), parameters_(parameters),
    Q_(0),size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()), mat(intvec({0,1})) {};

  Covariance(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) :
    form_(form), data_(data), colnames_(colnames), parameters_(parameters),
    Q_(0),size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()), mat(intvec({0,1})) {};

  Covariance(const std::string& formula,
             const Eigen::ArrayXXd &data,
             const strvec& colnames,
             const Eigen::ArrayXd& parameters) :
    form_(formula), data_(data), colnames_(colnames),Q_(0),
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()), mat(intvec({0,1})) {
    update_parameters(parameters);
  };

  Covariance(const glmmr::Formula& form,
             const Eigen::ArrayXXd &data,
              const strvec& colnames,
              const Eigen::ArrayXd& parameters) :
    form_(form), data_(data), colnames_(colnames),Q_(0),
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()), mat(intvec({0,1})) {
    update_parameters(parameters);
  };

  void update_parameters(const dblvec& parameters){
    parameters_ = parameters;
    if(isSparse)update_ax();
  };

  void update_parameters_extern(const dblvec& parameters){
    parameters_ = parameters;
    if(isSparse)update_ax();
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
      if(isSparse)update_ax();
    } else {
      Rcpp::stop("Wrong number of parameters");
    }
  };

  void parse();

  double get_val(int b, int i, int j);

  Eigen::MatrixXd Z();

  Eigen::MatrixXd D(bool chol = false,
                    bool upper = false){
    Eigen::MatrixXd D(Q_,Q_);
    if(isSparse){
      D = D_sparse_builder(chol,upper);
    } else {
      D = D_builder(0,chol,upper);
    }
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
    for(int i = 0; i<B_; i++){
      if(block_dim(i) > max)max = block_dim(i);
    }
    return max;
  }
  
  double log_likelihood(const Eigen::VectorXd &u);

  double log_determinant();

  int block_dim(int b){
    return re_data_[b].size();
  };

  void make_sparse();
  
  void make_dense();
  
  bool any_group_re();
  
  intvec parameter_fn_index(){
    return re_fn_par_link_;
  }
  
  intvec re_count(){
    return re_count_;
  }

private:
  intvec z_;
  dblvec3d re_data_;
  intvec3d re_cols_;
  intvec3d re_cols_data_;
  //dblvec3d z_data_;
  strvec2d fn_;
  intvec re_order_;
  intvec3d re_pars_;
  intvec2d re_rpn_;
  intvec2d re_index_;
  intvec2d re_obs_index_;
  intvec re_fn_par_link_;
  intvec re_count_;
  int Q_;
  int n_;
  int B_;
  int npars_;
  Eigen::ArrayXd size_B_array;
  Eigen::MatrixXd dmat_matrix;
  Eigen::VectorXd zquad;
  bool isSparse = false;
  sparse mat;
  
  Eigen::MatrixXd get_block(int b);
  
  Eigen::MatrixXd get_chol_block(int b,bool upper = false);

  Eigen::MatrixXd D_builder(int b,
                            bool chol = false,
                            bool upper = false);

  void update_ax();

  Eigen::MatrixXd D_sparse_builder(bool chol = false,
                                   bool upper = false);
};

}

#include "covariance.ipp"

#endif
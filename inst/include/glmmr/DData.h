#ifndef DDATA_H
#define DDATA_H

//#include <Eigen/Dense>
#include <RcppEigen.h>
//#include <Rcpp.h>

// class to hold data that describes how to construct the covariance matrix
// option to add a constructor to generate these data from a formula and data but requires 
// parsing character string which I can't be bothered to do in c++!

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {
  class DData {
    public: 
      Eigen::ArrayXXi cov_;
      Eigen::ArrayXd data_;
      Eigen::ArrayXd eff_range_;
      Eigen::ArrayXXi subcov_;
      Eigen::ArrayXd subdata_;
      Eigen::ArrayXd subeff_range_;
      int b_;
      int B_;
      int parstart_;
      int parsize_;
      
      DData(Eigen::ArrayXXi cov,
            Eigen::ArrayXd data,
            Eigen::ArrayXd eff_range) :
        cov_(cov), data_(data), eff_range_(eff_range) {
        b_ = 0;
        B_ = cov_.col(0).maxCoeff() + 1;
        subdata(0);
      }
      
      void subdata(int b){
        // find block numbers
        int start = 0;
        int end = 0;
        for(int i = 0; i < cov_.rows(); i++){
          if(cov_(i,0) != b && start == end){
            start++;
            end++;
          } else if(cov_(i,0) == b) {
            end++;
          } else if(cov_(i,0) != b && start < end){
            break;
          }
        }
        subcov_ = cov_.block(start,0,end-start,cov_.cols());
        subeff_range_ = eff_range_.segment(start,end-start);
        int dstart = 0;
        int dend = 0;
        
        for(int i = 0; i < end; i++){
          if(i < start ){
            dstart += cov_(i,1)*cov_(i,3);
            dend += cov_(i,1)*cov_(i,3);
          } else {
            dend += cov_(i,1)*cov_(i,3);
          }
        }
        subdata_ = data_.segment(dstart,dend-dstart);
        b_ = b;
        
        parstart_ = 0;
        parsize_ = 0; 
        for(int i = 0; i < end; i++){
          if(i < start){
            parstart_ += cov_(i,4);
          } else {
           parsize_ += cov_(i,4); 
          }
        }
      }
      
      int n_dim(){
        return subcov_(0,1);
      }
      
      bool check_all_func_def(){
        return (subcov_.col(2) != 1).any();
      }
      
      int n_func(){
        return subcov_.rows();
      }
      
      int n_var_func(int k){
        return subcov_(k,3);
      }
      
      int func_def(int k){
        return subcov_(k,2);
      }
      
      // k th function
      // p th variable
      double value(int k, int p, int i){
        int n = n_dim();
        int svar = 0;
        for(int j = 0; j < k; j++){
          svar += n_var_func(j);
        }
        //Rcpp::Rcout << "\nData idx: " << svar << " n:" << n << " i:" << i << " all: " << svar*n + p*n + i << " val:" << subdata_(svar*n + p*n + i) << "\n";
        return subdata_(svar*n + p*n + i);
      }
      
      int par_index(int k){
        return subcov_(k,4);
      }
  };
}
// 
// namespace glmmr {
// class DData {
// public:
//   int B_;
//   Eigen::VectorXi N_dim_;
//   Eigen::VectorXi N_func_;
//   Eigen::MatrixXi func_def_;
//   Eigen::MatrixXi N_var_func_;
//   Eigen::MatrixXd eff_range_;
//   std::vector<Eigen::MatrixXi> col_id_;
//   Eigen::MatrixXi N_par_;
//   std::vector<Eigen::MatrixXd> cov_data_;
//   Eigen::VectorXd gamma_;
//   
//   DData(int B,
//         Eigen::VectorXi N_dim,
//         Eigen::VectorXi N_func,
//         Eigen::MatrixXi func_def,
//         Eigen::MatrixXi N_var_func,
//         Eigen::MatrixXd eff_range,
//         std::vector<Eigen::MatrixXi> col_id,
//         Eigen::MatrixXi N_par,
//         std::vector<Eigen::MatrixXd> cov_data) :
//     B_(B), N_dim_(N_dim), N_func_(N_func),
//     func_def_(func_def), N_var_func_(N_var_func),
//     eff_range_(eff_range), col_id_(col_id),
//     N_par_(N_par), cov_data_(cov_data) {}
//   
//   DData(int B,
//         Eigen::VectorXi N_dim,
//         Eigen::VectorXi N_func,
//         Eigen::MatrixXi func_def,
//         Eigen::MatrixXi N_var_func,
//         Eigen::MatrixXd eff_range,
//         Eigen::MatrixXi col_id,
//         Eigen::MatrixXi N_par,
//         Eigen::MatrixXd cov_data) : B_(B), N_dim_(N_dim), N_func_(N_func),
//         func_def_(func_def), N_var_func_(N_var_func),
//         eff_range_(eff_range), N_par_(N_par) {
//     int max_N_func = N_func_.maxCoeff();
//     int max_N_dim = N_dim_.maxCoeff();
//     int iter1 = 0;
//     int iter2 = 0;
//     for (int i = 0; i < B_; i++) {
//       col_id_.push_back(col_id.block(iter1,0,max_N_func,col_id.cols()));
//       cov_data_.push_back(cov_data.block(iter2,0,max_N_dim,cov_data.cols()));
//       iter1 += max_N_func;
//       iter2 += max_N_dim;                   
//     }
//   }
//   
//   DData(Rcpp::List data) {
//     B_ = (int)(data["B"]);
//     col_id_.reserve(B_);
//     cov_data_.reserve(B_);
//     N_dim_ = Rcpp::as<Eigen::Map<Eigen::VectorXi> >(data["N_dim"]);
//     N_func_ = Rcpp::as<Eigen::Map<Eigen::VectorXi> >(data["N_func"]);
//     func_def_ = Rcpp::as<Eigen::Map<Eigen::MatrixXi> >(data["func_def"]);
//     N_var_func_ = Rcpp::as<Eigen::Map<Eigen::MatrixXi> >(data["N_var_func"]);
//     eff_range_ = Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(data["eff_range"]);
//     N_par_ = Rcpp::as<Eigen::Map<Eigen::MatrixXi> >(data["N_par"]);
//     int max_N_func = N_func_.maxCoeff();
//     int max_N_dim = N_dim_.maxCoeff();
//     int iter1 = 0;
//     int iter2 = 0;
//     Eigen::MatrixXi col_id_full = Rcpp::as<Eigen::Map<Eigen::MatrixXi> >(data["col_id"]);
//     Eigen::MatrixXd cov_data_full = Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(data["cov_data"]);
//     for (int i = 0; i < B_; i++) {
//       col_id_.push_back(col_id_full.block(iter1,0,max_N_func,col_id_full.cols()));
//       cov_data_.push_back(cov_data_full.block(iter2,0,max_N_dim,cov_data_full.cols()));
//       iter1 += max_N_func;
//       iter2 += max_N_dim;                   
//     }
//     }
//   
//   };
// }



#endif
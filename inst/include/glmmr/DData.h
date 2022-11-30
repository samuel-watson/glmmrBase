#ifndef DDATA_H
#define DDATA_H

#include <RcppEigen.h>
//#include <Rcpp.h>

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
      int matstart_;
      
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
        
        matstart_ = 0;
        bool isb = b==0;
        int iter = 0;
        int bnow = B_+1;
        while(!isb){
          isb = cov_(iter,0) == b;
          if(bnow != cov_(iter,0) && !isb )matstart_ += cov_(iter,1);
          bnow = cov_(iter,0);
          iter++;
        }
        
      }
      
      int N(){
        int N_ = 0;
        int btr = B_;
        for(int i = 0; i < cov_.rows(); i++){
          if(btr != cov_(i,0)){
            N_ += cov_(i,1);
            btr = cov_(i,0);
          } 
        }
        return N_;
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
      
      int n_cov_pars(){
        Eigen::Index maxIndex;
        int maxVal = cov_.col(4).maxCoeff(&maxIndex);
        int fdef = cov_(maxIndex,2);
        //check if 1 or 2 parameters (perhaps this could be improved!)
        bool onepar = (fdef == 1 || fdef == 2 || fdef == 3 || fdef == 6 || fdef ==14);
        maxVal += onepar ? 1 : 2;
        return maxVal;
      }
  };
}



#endif
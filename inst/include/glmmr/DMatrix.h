#ifndef DMATRIX_H
#define DMATRIX_H

#include <cmath> 
#include <unordered_map>
#include <RcppEigen.h>

#include "DData.h"
#include "maths.h"
#include "algo.h"

// #ifdef _OPENMP
// #include <omp.h>
// #endif

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {

class DSubMatrix {
public:
  int B_;
  int n_;
  DData* data_;
  Eigen::VectorXd gamma_;
  
  DSubMatrix(int B,
             DData* data,
             Eigen::VectorXd &gamma) :
  B_(B), data_(data), gamma_(gamma) {
    data_->subdata(B_);
    n_ = data_->n_dim();
  }
  
  DSubMatrix(int B,
             DData* data,
             Eigen::ArrayXd &gamma) :
  B_(B), data_(data), gamma_(gamma.size()) {
    gamma_ = gamma.matrix();
    data_->subdata(B_);
    n_ = data_->n_dim();
  }
  
  DSubMatrix(int B,
             DData* data,
             const std::vector<double> &gamma) :
    B_(B), data_(data), gamma_(gamma.size()) {
    std::vector<double> par2 = gamma;
    gamma_ = Eigen::Map<Eigen::VectorXd>(par2.data(),par2.size());
    data_->subdata(B_);
    n_ = data_->n_dim();
  }
  
  // generate and return the submatrix D
  Eigen::MatrixXd genSubD() {
    Eigen::MatrixXd D_ = Eigen::MatrixXd::Zero(n_, n_);
    if (data_->check_all_func_def()) {
//#pragma omp parallel for // slower in parallel!
      for (int i = 0; i < (n_ - 1); i++) {
        for (int j = i + 1; j < n_; j++) {
          double val = get_val(i, j);
          D_(i, j) = val;
          D_(j, i) = val;
        }
      }
    }
    
//#pragma omp parallel for
    for (int i = 0; i < n_; i++) {
      double val = get_val(i, i);
      D_(i, i) = val;
    }
    return D_;
  }
  
  // generate the value of an element of the covariance matrix
  double get_val(int i, int j) {
    double val = 1;
    for (int k = 0; k < data_->n_func(); k++) {
      double dist = 0;
      if (i != j) {
        for (int p = 0; p < data_->n_var_func(k); p++) {
          double diff = data_->value(k,p,i) - data_->value(k,p,j);
          dist += diff * diff;
        }
        dist = sqrt(dist);
      }
      int mcase = data_->func_def(k);
      switch (mcase) {
      case 1:
        if (dist == 0) {
          val = val * gamma_(data_->par_index(k)) * gamma_(data_->par_index(k));
        }
        else {
          val = 0;
        }
        break;
      case 2:
        val = val * exp(-1 * dist / gamma_(data_->par_index(k)));
        break;
      case 3:
        val = val * pow(gamma_(data_->par_index(k)), dist);
        break;
      case 4:
        val = val * gamma_(data_->par_index(k)) * exp(-1 * dist * dist / (gamma_(data_->par_index(k) + 1) * gamma_(data_->par_index(k) + 1)));
        break;
      case 5:
        {
          double xr = sqrt(2 * gamma_(data_->par_index(k) + 1)) * dist / gamma_(data_->par_index(k));
          double ans = 1;
          if (xr != 0) {
            if (gamma_(data_->par_index(k) + 1) == 0.5) {
              ans = exp(-xr);
            }
            else {
              double cte = pow(2.0, -1 * (gamma_(data_->par_index(k) + 1) - 1)) / tgamma(gamma_(data_->par_index(k) + 1));
              ans = cte * pow(xr, gamma_(data_->par_index(k) + 1)) * R::bessel_k(xr, gamma_(data_->par_index(k) + 1),1);
            }
          }
          val = val * ans;
          break;
        }
      case 6:
        val = val * R::bessel_k(dist / gamma_(data_->par_index(k)), 1, 1);
        break;
      case 7:
        //wend 0
        {
          double pdist = dist / data_->subeff_range_(k);
          if (pdist >= 1) {
            val = 0;
          }
          else {
            val = val * gamma_(data_->par_index(k)) * pow((1 - pdist), gamma_(data_->par_index(k) + 1));
          }
          break;
        }
      case 8:
        // wend 1
        {
          double pdist = dist / data_->subeff_range_(k);
          if (pdist >= 1) {
            val = 0;
          }
          else {
            val = val * gamma_(data_->par_index(k)) * (1 + (1 + gamma_(data_->par_index(k) + 1)) * pdist) * pow((1 - pdist), gamma_(data_->par_index(k) + 1) + 1.0);
          }
          break;
        }
      case 9:
        // wend 2
        {
          double pdist = dist / data_->subeff_range_(k);
          if (pdist >= 1) {
            val = 0;
          }
          else {
            val = val * gamma_(data_->par_index(k)) * (1 + (gamma_(data_->par_index(k) + 1) + 2) * pdist + 0.333 * ((gamma_(data_->par_index(k) + 1) + 2) * (gamma_(data_->par_index(k) + 1) + 2) - 1) * pdist * pdist) * pow((1 - pdist), gamma_(data_->par_index(k) + 1) + 2.0);
          }
          break;
        }
      case 10:
        //prodwm
        {
          double pdist = dist / data_->subeff_range_(k);
          if (pdist >= 1) {
            val = 0;
          }
          else {
            double wm;
            if (pdist == 0) {
              wm = 1;
            }
            else {
              wm = (pow(2.0, 1 - gamma_(data_->par_index(k) + 1)) / tgamma(gamma_(data_->par_index(k) + 1))) * pow(pdist, gamma_(data_->par_index(k) + 1)) * R::bessel_k(pdist, gamma_(data_->par_index(k) + 1),1);
              
            }
            double poly = (1 + (11 / 2) * pdist + (117 / 12) * pdist * pdist) * pow(1 - pdist, (11 / 2));
            val = val * gamma_(data_->par_index(k)) * wm * poly;
          }
          break;
        }
      case 11:
        //prodcb
        {
          double pdist = dist / data_->subeff_range_(k);
          if (pdist >= 1) {
            val = 0;
          }
          else {
            double cauc = pow((1 + pow(pdist, gamma_(data_->par_index(k) + 1))), -3);
            double boh = (1 - pdist) * cos(M_PI * pdist) * (1 / M_PI) * sin(M_PI * pdist);
            val = val * gamma_(data_->par_index(k)) * cauc * boh;
          }
          break;
        }
      case 12:
        //prodek
        {
          double pdist = dist / data_->subeff_range_(k);
          if (pdist >= 1) {
            val = 0;
          }
          else {
            double pexp = exp(-1.0 * pow(pdist, gamma_(data_->par_index(k) + 1)));
            double kan = (1 - pdist) * sin(2 * M_PI * pdist) / (2 * M_PI * pdist) + (1 / M_PI) * (1 - cos(2 * M_PI * pdist)) / (2 * M_PI * pdist);
            val = val * gamma_(data_->par_index(k)) * pexp * kan;
          }
          break;
        }
      case 13:
        val = val * gamma_(data_->par_index(k)) * exp(-1 * dist / gamma_(data_->par_index(k) + 1));
        break;
      case 14:
        val = val * exp(-1 * dist * dist / (gamma_(data_->par_index(k)) * gamma_(data_->par_index(k))));
      }
    }
    
    return val;
  }
  
  // generate the cholesky decomposition of the submatrix
  Eigen::MatrixXd genCholSubD(bool upper = false) {
    int n = (int)n_;
    std::vector<double> L(n * n, 0.0);
    //double* L = (double*)std::calloc(n * n, sizeof(double));
    
    for (int j = 0; j < n; j++) {
      double s = algo::inner_sum(&L[j * n], &L[j * n], j);
      L[j * n + j] = sqrt(get_val(j, j) - s);
      //#pragma omp parallel for schedule(static, 8) // this does not make it faster!!
      for (int i = j + 1; i < n; i++) {
        double s = algo::inner_sum(&L[j * n], &L[i * n], j);
        L[i * n + j] = (1.0 / L[j * n + j] * (get_val(j, i) - s));
      }
    }
    Eigen::MatrixXd M = Eigen::Map<Eigen::MatrixXd>(L.data(), n, n);
    if (upper) {
      return M;
      } else {
        return M.transpose();
      }
  }
  
};

class DMatrix {
  public:
  
  Eigen::VectorXd gamma_;
  DData* data_;
  
  DMatrix(DData* data,
          const Eigen::VectorXd& gamma) :
  gamma_(gamma),
  data_(data) {}
  
  DMatrix(DData* data,
          const Eigen::ArrayXd& gamma) :
    gamma_(gamma.size()),
    data_(data) 
  {
    gamma_ = gamma.matrix();  
  }
  
  DMatrix(DData* data,
          const std::vector<double>& gamma) :
    gamma_(gamma.size()),
    data_(data) 
  {
    std::vector<double> par2 = gamma;
    gamma_ = Eigen::Map<Eigen::VectorXd>(par2.data(),par2.size());
  }
  
  Eigen::MatrixXd gen_block_mat(int b,
                                bool chol = false,
                                bool upper = false)
  {
    Eigen::MatrixXd bblock;
    DSubMatrix* dblock;
    data_->subdata(b);
    dblock = new DSubMatrix(b, data_, gamma_);
    
    if (!chol)
    {
      bblock = dblock->genSubD();
    }
    else
    {
      bblock = dblock->genCholSubD(upper);
    }
    delete dblock;
    return bblock;
  }
  
  void update_parameters(const Eigen::VectorXd& gamma)
  {
    gamma_ = gamma;
  }
  
  void update_parameters(const Eigen::ArrayXd& gamma)
  {
    gamma_ = gamma.matrix();
  }
  
  void update_parameters(const std::vector<double>& gamma)
  {
    std::vector<double> par2 = gamma;
    gamma_ = Eigen::Map<Eigen::VectorXd>(par2.data(),par2.size()); 
  }
  
  int B()
  {
    return data_->B_;
  }
  
  Eigen::VectorXd sim_re(){
    Eigen::VectorXd samps(data_->N());
    int idx = 0;
    for(int i=0; i< data_->B_; i++){
      data_->subdata(i);
      Eigen::MatrixXd L = gen_block_mat(i,true,false);
      Rcpp::NumericVector z = Rcpp::rnorm(data_->n_dim());
      Eigen::Map<Eigen::VectorXd> Z(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(z));
      samps.segment(idx,data_->n_dim()) = L*Z;
      idx += data_->n_dim();
    }
    return samps;
  }
  
  Eigen::MatrixXd genD(int b,
                       bool chol = false,
                       bool upper = false) {
    if (b == data_->B_ - 1) {
      return gen_block_mat(b, chol, upper);
    }
    else {
      Eigen::MatrixXd mat1 = gen_block_mat(b, chol, upper);
      Eigen::MatrixXd mat2;
      if (b == data_->B_ - 2) {
        mat2 = gen_block_mat(b + 1, chol, upper);
      }
      else {
        mat2 = genD(b + 1, chol, upper);
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


#endif
#ifndef GLMMR_H
#define GLMMR_H

#include <cmath> 
#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

inline double gaussian_cdf(double x){
  return R::pnorm(x, 0, 1, true, false);
}

inline arma::vec gaussian_cdf_vec(const arma::vec& v){
  arma::vec res = arma::zeros<arma::vec>(v.n_elem);
  for (arma::uword i = 0; i < v.n_elem; ++i)
    res[i] = gaussian_cdf(v[i]);
  return res;
}

inline double gaussian_pdf(double x){
  return R::dnorm(x, 0, 1, false);
}

inline arma::vec gaussian_pdf_vec(const arma::vec& v){
  arma::vec res = arma::zeros<arma::vec>(v.n_elem);
  for (arma::uword i = 0; i < v.n_elem; ++i)
    res[i] = gaussian_pdf(v[i]);
  return res;
}

inline double log_mv_gaussian_pdf(const arma::vec& u,
                                  const arma::mat& D,
                                  const double& logdetD){
  arma::uword Q = u.n_elem;
  return (-0.5*Q*log(2*arma::datum::pi)-
          0.5*logdetD - 0.5*arma::as_scalar(u.t()*D*u));
}

inline arma::vec mod_inv_func(arma::vec mu,
                              std::string link){
  //arma::uword n = mu.n_elem;
  if(link=="logit"){
    mu = exp(mu) / (1+exp(mu));
  }
  if(link=="log"){
    mu = exp(mu);
  }
  if(link=="probit"){
    mu = gaussian_cdf_vec(mu);
  }
  
  return mu;
}

inline arma::vec forward_sub(const arma::mat &L,
                             const arma::vec &u){
  arma::vec z(u.size());
  for(arma::uword i=0; i<u.size(); i++){
    double lsum = 0;
    if(i > 0){
      for(arma::uword k=0; k<i; k++){
        lsum += L(i,k)*z(k);
      }
    }
    z(i) = (u(i) - lsum)/L(i,i);
  }
  return z;
}

inline arma::vec forward_back_sub(const arma::mat &L,
                                  const arma::mat &U,
                                  const arma::vec &u){
  arma::uword n = u.size();
  arma::vec z = forward_sub(L,u);
  arma::vec y(n);
  
  y(n-1) = z(n-1)/U(n-1,n-1);
  for(arma::uword i=n-1; i>0; i--){
    double lsum = 0;
    for(arma::uword k=i; k<n; k++){
      lsum += U(i-1,k)*y(k);
    }
    y(i-1) = (z(i-1) - lsum)/U(i-1,i-1);
  }
  return y;
}


class DSubMatrix {
  size_t N_dim_;
  size_t N_func_;
  arma::uvec func_def_;
  arma::uvec N_var_func_;
  arma::umat col_id_;
  arma::uvec N_par_;
  arma::mat cov_data_;
  arma::vec gamma_;
public:
  DSubMatrix(size_t N_dim,
             size_t N_func,
             const arma::uvec &func_def,
             const arma::uvec &N_var_func,
             const arma::umat &col_id,
             const arma::uvec &N_par,
             const arma::mat &cov_data,
             const arma::vec &gamma):
  N_dim_(N_dim), N_func_(N_func), func_def_(func_def),
  N_var_func_(N_var_func), col_id_(col_id), N_par_(N_par),
  cov_data_(cov_data), gamma_(gamma) {}
  
  arma::mat genSubD(){
    arma::mat D_(N_dim_,N_dim_,fill::zeros);
    if(!all(func_def_ == 1)){
#pragma omp parallel for
      for(arma::uword i=0;i<(N_dim_-1);i++){
        for(arma::uword j=i+1;j<N_dim_;j++){
          double val = get_val(i,j);
          D_(i,j) = val;
          D_(j,i) = val;
        }
      }
    }
    
#pragma omp parallel for
    for(arma::uword i=0;i<N_dim_;i++){
      double val = get_val(i,i);
      D_(i,i) = val;
    }
    return D_;
  }
  
  double get_val(arma::uword i, arma::uword j){
    double val = 1;
    if(i==j){
      size_t gamma_idx_ii = 0;
      for(arma::uword k=0;k<N_func_;k++){
        if(func_def_(k)==1){
          val = val*pow(gamma_(gamma_idx_ii),2);
        }
        gamma_idx_ii += N_par_(k); 
      }
    } else {
      size_t gamma_idx = 0;
      for(arma::uword k=0;k<N_func_;k++){
        double dist = 0;
        for(arma::uword p=0; p<N_var_func_(k); p++){
          dist += pow(cov_data_(i,col_id_(k,p)-1) - cov_data_(j,col_id_(k,p)-1),2);
        }
        dist= pow(dist,0.5);
        
        int mcase = (int)func_def_(k);
        switch (mcase){
        case 1:
          if(dist==0){
            val = val*pow(gamma_(gamma_idx),2);
          } else {
            val = 0;
          }
          break;
        case 2:
          val = val*exp(-1*dist/gamma_(gamma_idx));
          break;
        case 3:
          val = val*pow(gamma_(gamma_idx),dist);
          break;
        case 4:
          val = val*gamma_(gamma_idx)*exp(-1*pow(dist,2)/pow(gamma_(gamma_idx+1),2));
          break;
        case 5:
          {
            double xr = pow(2*gamma_(gamma_idx+1),0.5)*dist/gamma_(gamma_idx);
            double ans = 1;
            if(xr!=0){
              if(gamma_(gamma_idx+1) == 0.5){
                ans = exp(-xr);
              } else {
                double cte = pow(2,-1*(gamma_(gamma_idx+1)-1))/R::gammafn(gamma_(gamma_idx+1));
                ans = cte*pow(xr, gamma_(gamma_idx+1))*R::bessel_k(xr,gamma_(gamma_idx+1),1);
              }
            }
            val = val*ans;
            break;
          }
        case 6:
          val = val* R::bessel_k(dist/gamma_(gamma_idx),1,1);
        }
        
        gamma_idx += N_par_(k);      
      }
    }
    
    return val;
  }
  
  arma::mat genCholSubD(){
    arma::mat L(N_dim_,N_dim_,fill::zeros);
    for(arma::uword i=0; i< N_dim_; i++){
      double sum = 0;
      for(arma::uword k=0; k<i; k++){
        sum += pow(L(i,k),2);
      }
      L(i,i) = sqrt(get_val(i,i) - sum);
      
      for(arma::uword j=i+1; j<N_dim_; j++){
        sum=0;
        for(arma::uword k =0; k<i; k++){
          sum += L(i,k) * L(j,k);
        }
        L(j,i) = ((1.0 / L(i,i)) * (get_val(j,i) - sum));
      }
    }
    return L;
  }
};

class DMatrix {
  arma::field<arma::mat> DBlocks_;
public:
  DMatrix(const arma::uword &B,
          const arma::uvec &N_dim,
          const arma::uvec &N_func,
          const arma::umat &func_def,
          const arma::umat &N_var_func,
          const arma::ucube &col_id,
          const arma::umat &N_par,
          const arma::uword &sum_N_par,
          const arma::cube &cov_data,
          const arma::vec &gamma):
  B_(B), N_dim_(N_dim), N_func_(N_func),
  func_def_(func_def), N_var_func_(N_var_func),
  col_id_(col_id), N_par_(N_par), sum_N_par_(sum_N_par),
  cov_data_(cov_data), gamma_(gamma) {
    DBlocks_ = arma::field<arma::mat>(B_);
  }
  
  arma::uword B_;
  arma::uvec N_dim_;
  arma::uvec N_func_;
  arma::umat func_def_;
  arma::umat N_var_func_;
  arma::ucube col_id_;
  arma::umat N_par_;
  arma::uword sum_N_par_;
  arma::cube cov_data_;
  arma::vec gamma_;
  
  arma::mat gen_block_mat(arma::uword b,
                          bool chol = false){
    arma::mat bblock;
    DSubMatrix *dblock;
    arma::uvec N_par_col0 = N_par_.col(0);
    arma::uword glim = (b == B_-1 || max(N_par_.row(b)) >= max(N_par_col0)) ?  gamma_.size() : min(N_par_col0(arma::find(N_par_col0 > max(N_par_.row(b)))));
    dblock = new DSubMatrix(N_dim_(b),
                            N_func_(b),
                            func_def_.row(b).t(),
                            N_var_func_.row(b).t(),
                            col_id_.slice(b),
                            N_par_.row(b).t(),
                            cov_data_.slice(b),
                            gamma_.subvec(min(N_par_.row(b)),glim-1));
    if(!chol){
      bblock = dblock->genSubD();
    } else {
      bblock = dblock->genCholSubD();
    }
    delete dblock;
    return bblock;
  }
  
  void get_block(arma::uword b,
                 bool chol = false){
    DBlocks_[b] = gen_block_mat(b,chol);
  }
  
  arma::field<arma::mat> genD(){
    for(arma::uword b=0;b<B_;b++){
      get_block(b);
    }
    return(DBlocks_);
  }
  
  arma::field<arma::mat> genCholD(){
    for(arma::uword b=0;b<B_;b++){
      get_block(b,true);
    }
    return DBlocks_;
  }
  
  void gen_blocks_byfunc(){
    for(arma::uword b=0;b<B_;b++){
      bool chol = !all(func_def_.row(b)==1);
      get_block(b,chol);
    }
  }
  
  double loglik(const arma::vec &u){
    arma::vec loglV(B_);
    double logdetD;
    //arma::uword ndim_idx = 0;
#pragma omp parallel for
    for(arma::uword b=0;b<B_;b++){
      arma::uword begin = b==0 ? 0 : sum(N_dim_.subvec(0,b-1));
      arma::uword end = sum(N_dim_.subvec(0,b)) - 1;
      if(all(func_def_.row(b)==1)){
        //arma::mat matB = gen_block_mat(b);
        arma::vec loglvec(DBlocks_[b].n_rows);
        for(arma::uword k=0; k<DBlocks_[b].n_rows; k++){
          loglvec(k) = -0.5*log(DBlocks_[b](k,k)) -0.5*log(2*arma::datum::pi) -
            0.5*pow(u(begin+k),2)/DBlocks_[b](k,k);
        }
        loglV(b) = sum(loglvec);
      } else {
        //arma::mat matB = gen_block_mat(b,true);
        logdetD = 2*sum(log(DBlocks_[b].diag()));
        arma::vec zquad(u.n_rows);
        double quadform;
        zquad  = forward_sub(DBlocks_[b],u.subvec(begin,end));
        quadform = arma::dot(zquad,zquad);
        loglV(b) = (-0.5*N_dim_(b) * log(2*arma::datum::pi) - 0.5*logdetD - 0.5*quadform);
    }
      //ndim_idx += N_dim_(b);
  }
    return arma::as_scalar(sum(loglV));
  }
  
  arma::rowvec log_gradient(const arma::vec &u){
    arma::uword n = u.size();
    arma::mat loglM(B_,n,fill::zeros);
#pragma omp parallel for
    for(arma::uword b=0;b<B_;b++){
      arma::uword begin = b==0 ? 0 : sum(N_dim_.subvec(0,b-1));
      arma::uword end = sum(N_dim_.subvec(0,b)) - 1;
      if(all(func_def_.row(b)==1)){
        loglM.row(b).cols(begin,end) = -(0.5/DBlocks_[b](0,0))*arma::trans(u.subvec(begin,end));
      } else {
        arma::vec zquad(n);
        zquad  = forward_back_sub(DBlocks_[b],arma::trans(DBlocks_[b]),u.subvec(begin,end));
        loglM.row(b).cols(begin,end) = -0.5*zquad.t();
      }
    }
    return sum(loglM);
  }
  
  double logdet(){
    double logdetD = 0;
    for(arma::uword b=0;b<B_;b++){
      //get_block(b,true);
      logdetD += 2*sum(log(DBlocks_[b].diag()));
    }
    return logdetD;
  }
  
  arma::uword B(){
    return B_;
  }
};


inline arma::vec dhdmu(const arma::vec &xb,
                    std::string family,
                    std::string link){
  
  arma::vec wdiag(xb.n_elem, fill::value(1));
  arma::vec p(xb.n_elem, fill::zeros);
  const static std::unordered_map<std::string,int> string_to_case{
    {"poissonlog",1},
    {"poissonidentity",2},
    {"binomiallogit",3},
    {"binomiallog",4},
    {"binomialidentity",5},
    {"binomialprobit",6},
    {"gaussianidentity",7},
    {"gaussianlog",8}
  };
  
  switch (string_to_case.at(family+link)){
  case 1:
    wdiag = 1/exp(xb);
    break;
  case 2:
    wdiag = exp(xb);
    break;
  case 3:
    p = mod_inv_func(xb,"logit");
    wdiag = 1/(p % (1-p));
    break;
  case 4:
    p = mod_inv_func(xb,"logit");
    wdiag = (1-p)/p;
    break;
  case 5:
    p = mod_inv_func(xb,"logit");
    wdiag = p % (1-p);
    break;
  case 6:
    p = mod_inv_func(xb,"probit");
    wdiag = (p % (1-p))/gaussian_pdf_vec(xb);
    break;
  case 7:
    break;
  case 8:
    wdiag = 1/exp(xb);
  }
  
  return wdiag;
}

inline arma::mat blockMatComb(arma::field<arma::mat> matfield){
  arma::uword nmat = matfield.n_rows;
  if(nmat==1){
    return matfield(0);
  } else {
    arma::mat mat1 = matfield(0);
    arma::mat mat2;
    if(nmat==2){
      mat2 = matfield(1);
    } else {
      mat2 = blockMatComb(matfield.rows(1,nmat-1));
    }
    arma::uword n1 = mat1.n_rows;
    arma::uword n2 = mat2.n_rows;
    arma::mat dmat(n1+n2,n1+n2);
    dmat.fill(0);
    dmat.submat(0,0,n1-1,n1-1) = mat1;
    dmat.submat(n1,n1,n1+n2-1,n1+n2-1) = mat2;
    return dmat;
  }
}

#endif
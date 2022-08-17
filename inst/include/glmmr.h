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

// extern arma::mat genBlockD(size_t N_dim,
//                            size_t N_func,
//                            const arma::uvec &func_def,
//                            const arma::uvec &N_var_func,
//                            const arma::umat &col_id,
//                            const arma::uvec &N_par,
//                            const arma::mat &cov_data,
//                            const arma::vec &gamma);
// 
// extern arma::field<arma::mat> genD(const arma::uword &B,
//                                    const arma::uvec &N_dim,
//                                    const arma::uvec &N_func,
//                                    const arma::umat &func_def,
//                                    const arma::umat &N_var_func,
//                                    const arma::ucube &col_id,
//                                    const arma::umat &N_par,
//                                    const arma::uword &sum_N_par,
//                                    const arma::cube &cov_data,
//                                    const arma::vec &gamma);

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
          double val = 1;
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
            
            // if(func_def_(k)==1){
            //   if(dist==0){
            //     val = val*pow(gamma_(gamma_idx),2);
            //   } else {
            //     val = 0;
            //   }
            // } else if(func_def_(k)==2){
            //   val = val*exp(-1*dist/gamma_(gamma_idx));//fexp(dist,gamma(gamma_idx));
            // }else if(func_def_(k)==3){
            //   val = val*pow(gamma_(gamma_idx),dist);
            // } else if(func_def_(k)==4){
            //   val = val*gamma_(gamma_idx)*exp(-1*pow(dist,2)/pow(gamma_(gamma_idx+1),2));
            // } else if(func_def_(k)==5){
            //   double xr = pow(2*gamma_(gamma_idx+1),0.5)*dist/gamma_(gamma_idx);
            //   double ans = 1;
            //   if(xr!=0){
            //     if(gamma_(gamma_idx+1) == 0.5){
            //       ans = exp(-xr);
            //     } else {
            //       double cte = pow(2,-1*(gamma_(gamma_idx+1)-1))/R::gammafn(gamma_(gamma_idx+1));
            //       ans = cte*pow(xr, gamma_(gamma_idx+1))*R::bessel_k(xr,gamma_(gamma_idx+1),1);
            //     }
            //   }
            //   val = val*ans;
            // } else if(func_def_(k)==6){
            //   val = val* R::bessel_k(dist/gamma_(gamma_idx),1,1);
            // }
            
            gamma_idx += N_par_(k);      
          }
          
          D_(i,j) = val;
          D_(j,i) = val;
        }
      }
    }
    
#pragma omp parallel for
    for(arma::uword i=0;i<N_dim_;i++){
      double val = 1;
      size_t gamma_idx_ii = 0;
      for(arma::uword k=0;k<N_func_;k++){
        if(func_def_(k)==1){
          val = val*pow(gamma_(gamma_idx_ii),2);
        } 
      }
      D_(i,i) = val;
    }
    return D_;
  }
};

class DMatrix {
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
  arma::field<arma::mat> DBlocks_;
  arma::uword g_idx_;
  arma::uword sumpar_;
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
  cov_data_(cov_data), gamma_(gamma) {}
  
  arma::field<arma::mat> genD(){
    arma::field<arma::mat> DBlocks(B_);
    arma::uword g_idx = 0;
    arma::uword sumpar;
    for(arma::uword b=0;b<B_;b++){
      sumpar = sum(N_par_.row(b));
      DSubMatrix *dblock;
      dblock = new DSubMatrix(N_dim_(b),
                              N_func_(b),
                              func_def_.row(b).t(),
                              N_var_func_.row(b).t(),
                              col_id_.slice(b),
                              N_par_.row(b).t(),
                              cov_data_.slice(b),
                              gamma_.subvec(g_idx,g_idx+sumpar-1));
      DBlocks[b] = dblock->genSubD();
      delete dblock;
      g_idx += sumpar;
    }
    return(DBlocks);
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
  
  // if(family=="poisson"){
  //   if(link=="log"){
  //     wdiag = 1/exp(xb);
  //   } else if(link =="identity"){
  //     wdiag = exp(xb);
  //   }
  // } else if(family=="binomial"){
  //   p = mod_inv_func(xb,"logit");
  //   if(link=="logit"){
  //     wdiag = 1/(p % (1-p));
  //   } else if(link=="log"){
  //     wdiag = (1-p)/p;
  //   } else if(link=="identity"){
  //     wdiag = p % (1-p);
  //   } else if(link=="probit"){
  //     p = mod_inv_func(xb,"probit");
  //     arma::vec p2(xb.n_elem,fill::zeros);
  //     wdiag = (p % (1-p))/gaussian_pdf_vec(xb);
  //   }
  // } else if(link=="gaussian"){
  //   // if identity do nothin
  //   if(link=="log"){
  //     wdiag = 1/exp(xb);
  //   }
  // } // for gamma- inverse do nothing
  return wdiag;
}

#endif
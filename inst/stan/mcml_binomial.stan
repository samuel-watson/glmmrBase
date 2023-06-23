functions {
  real partial_sum1_lpdf(array[] real y, int start, int end){
    return std_normal_lpdf(y[start:end]);
  }
  real partial_sum2_lpmf(array[] int y,int start, int end,array[] int n, vector mu,int type){
    real out;
    if(type==1) out = binomial_logit_lpmf(y[start:end]|n[start:end],mu[start:end]);
    if(type==2) out = binomial_lpmf(y[start:end]|n[start:end],exp(mu[start:end]));
    if(type==3) out = binomial_lpmf(y[start:end]|n[start:end],mu[start:end]);
    if(type==4) out = binomial_lpmf(y[start:end]|n[start:end],Phi_approx(mu[start:end]));
    return out;
  }
}
data {
  int N; // sample size
  int Q; // columns of Z, size of RE terms
  vector[N] Xb;
  matrix[N,Q] Z;
  array[N] int y;
  array[N] int n;
  int type;
}
parameters {
  array[Q] real gamma;
}
model {
  int grainsize = 1;
  target += reduce_sum(partial_sum1_lpdf,gamma,grainsize);
  target += reduce_sum(partial_sum2_lpmf,y,grainsize,n,Xb + Z*to_vector(gamma),type);
}


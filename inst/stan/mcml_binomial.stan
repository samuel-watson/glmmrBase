functions {
  real partial_sum1_lpdf(array[] real y, int start, int end){
    return std_normal_lpdf(y[start:end]);
  }
  real partial_sum2_lpmf(array[] int y,int start, int end, vector mu,int type){
    real out;
    if(type==1) out = bernoulli_logit_lpmf(y[start:end]|mu[start:end]);
    if(type==2) out = bernoulli_lpmf(y[start:end]|exp(mu[start:end]));
    if(type==3) out = bernoulli_lpmf(y[start:end]|mu[start:end]);
    if(type==4) out = bernoulli_lpmf(y[start:end]|Phi_approx(mu[start:end]));
    return out;
  }
}
data {
  int N; // sample size
  int Q; // columns of Z, size of RE terms
  vector[N] Xb;
  matrix[N,Q] Z;
  array[N] int y;
  real sigma;
  int type;
}
parameters {
  array[Q] real gamma;
}
model {
  int grainsize = 1;
  target += reduce_sum(partial_sum1_lpdf,gamma,grainsize);
  target += reduce_sum(partial_sum2_lpmf,y,grainsize,Xb + Z*to_vector(gamma),type);
}


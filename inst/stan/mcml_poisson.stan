functions {
  real partial_sum1_lpdf(array[] real y, int start, int end){
    return std_normal_lpdf(y[start:end]);
  }
  real partial_sum2_lpmf(array[] int y,int start, int end, vector mu){
    return poisson_log_lpmf(y[start:end]|mu[start:end]);
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
  target += reduce_sum(partial_sum2_lpmf,y,grainsize,Xb + Z*to_vector(gamma));
}


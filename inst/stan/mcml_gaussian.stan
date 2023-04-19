functions {
  real partial_sum1_lpdf(array[] real y, int start, int end){
    return std_normal_lpdf(y[start:end]);
  }
  real partial_sum2_lpdf(array[] real y,int start, int end, vector mu,real sigma){
    return normal_lpdf(y[start:end]|mu[start:end],sigma);
  }
}
data {
  int N; // sample size
  int Q; // columns of Z, size of RE terms
  vector[N] Xb;
  matrix[N,Q] Z;
  array[N] real y;
  real sigma;
  int type;
}
parameters {
  array[Q] real gamma;
}
model {
  //vector[Q] zeroes = rep_vector(0,Q);
  //gamma ~ multi_normal_cholesky(zeroes,L);
  int grainsize = 1;
  target += reduce_sum(partial_sum1_lpdf,gamma,grainsize);
  //gamma ~ std_normal();
  //if(type==1)y ~ normal(Xb + Z*gamma,sigma);
  target += reduce_sum(partial_sum2_lpdf,y,grainsize,Xb + Z*to_vector(gamma),sigma);
}


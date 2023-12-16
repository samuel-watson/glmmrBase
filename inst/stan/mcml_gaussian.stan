functions {
  real partial_sum1_lpdf(array[] real y, int start, int end){
    return std_normal_lpdf(y[1:(1+end-start)]);
  }
  real partial_sum2_lpdf(array[] real y,int start, int end, vector mu,vector sigma){
    return normal_lpdf(y[1:(1+end-start)]|mu[1:(1+end-start)],sigma[1:(1+end-start)]);
  }
}
data {
  int N; // sample size
  int Q; // columns of Z, size of RE terms
  vector[N] Xb;
  matrix[N,Q] Z;
  array[N] real y;
  vector[N] sigma;
  int type;
}
parameters {
  array[Q] real gamma;
}
model {
  int grainsize = 1;
  target += reduce_sum(partial_sum1_lpdf,gamma,grainsize);
  target += reduce_sum(partial_sum2_lpdf,y,grainsize,Xb + Z*to_vector(gamma),sqrt(sigma));
}


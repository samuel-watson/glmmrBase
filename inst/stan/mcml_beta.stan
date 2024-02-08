// functions {
//   real partial_sum1_lpdf(array[] real y, int start, int end){
//     return std_normal_lpdf(y[1:(1+end-start)]);
//   }
//   real partial_sum2_lpdf(array[] real y,int start, int end,vector mu,real phi, int type){
//     real out;
//     if(type==1) out = beta_lpdf(y[1:(1+end-start)]|mu*phi, (1-mu)*phi);
//     return out;
//   }
// }
data {
  int N; // sample size
  int Q; // columns of Z, size of RE terms
  vector[N] Xb;
  matrix[N,Q] Z;
  array[N] real y;
  real var_par;
  int type;
}
parameters {
  array[Q] real gamma;
}
transformed parameters {
  vector[N] logitmu = 1/(1+exp(-1*Xb - Z*to_vector(gamma)));
}
model {
  to_vector(gamma) ~ std_normal();
  if(type==1) to_vector(y) ~ beta(logitmu*var_par, (1-logitmu)*var_par);
  
  // int grainsize = 1;
  // target += reduce_sum(partial_sum1_lpdf,gamma,grainsize);
  // target += reduce_sum(partial_sum2_lpdf,y,grainsize,logitmu,var_par,type);
}


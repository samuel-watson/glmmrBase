// functions {
//   real partial_sum1_lpdf(array[] real y, int start, int end){
//     return std_normal_lpdf(y[1:(1+end-start)]);
//   }
//   real partial_sum2_lpdf(array[] real y,int start, int end,vector mu,real phi, int type){
//     real out;
//     if(type==1) out = gamma_lpdf(y[1:(1+end-start)]|1/phi, 1/(phi*mu[1:(1+end-start)]));
//     if(type==2) out = gamma_lpdf(y[1:(1+end-start)]|1/phi, mu[1:(1+end-start)]/phi);
//     if(type==3) out = gamma_lpdf(y[1:(1+end-start)]|1/phi, 1/(phi*log(mu[1:(1+end-start)])));
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
model {
  to_vector(gamma) ~ std_normal();
  if(type==1) to_vector(y) ~ gamma(1/var_par, 1/(var_par*(Xb + Z*to_vector(gamma))));
  if(type==2) to_vector(y) ~ gamma(1/var_par, (Xb + Z*to_vector(gamma))/var_par);
  if(type==3) to_vector(y) ~ gamma(1/var_par, 1/(var_par*log(Xb + Z*to_vector(gamma))));
  
  // int grainsize = 1;
  // target += reduce_sum(partial_sum1_lpdf,gamma,grainsize);
  // target += reduce_sum(partial_sum2_lpdf,y,grainsize,Xb + Z*to_vector(gamma),var_par,type);
}


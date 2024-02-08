// functions {
//   real partial_sum1_lpdf(array[] real y, int start, int end){
//     return std_normal_lpdf(y[1:(1+end-start)]);
//   }
//   real partial_sum2_lpmf(array[] int y,int start, int end,array[] int n, vector mu,int type){
//     real out;
//     if(type==1) out = binomial_logit_lpmf(y[1:(1+end-start)]|n[1:(1+end-start)],mu[1:(1+end-start)]);
//     if(type==2) out = binomial_lpmf(y[1:(1+end-start)]|n[1:(1+end-start)],exp(mu[1:(1+end-start)]));
//     if(type==3) out = binomial_lpmf(y[1:(1+end-start)]|n[1:(1+end-start)],mu[1:(1+end-start)]);
//     if(type==4) out = binomial_lpmf(y[1:(1+end-start)]|n[1:(1+end-start)],Phi_approx(mu[1:(1+end-start)]));
//     return out;
//   }
// }
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
  to_vector(gamma) ~ std_normal();
  if(type==1) y ~ binomial_logit(n ,Xb + Z*to_vector(gamma));
  if(type==2) y ~ binomial(n,exp(Xb + Z*to_vector(gamma)));
  if(type==3) y ~ binomial(n,Xb + Z*to_vector(gamma));
  if(type==4) y ~ binomial(n,Phi_approx(Xb + Z*to_vector(gamma)));
  
  // int grainsize = 1;
  // target += reduce_sum(partial_sum1_lpdf,gamma,grainsize);
  // target += reduce_sum(partial_sum2_lpmf,y,grainsize,n,Xb + Z*to_vector(gamma),type);
}


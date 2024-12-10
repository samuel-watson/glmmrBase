// functions {
//   real partial_sum1_lpdf(array[] real y, int start, int end){
//     return std_normal_lpdf(y[1:(1+end-start)]);
//   }
//   real partial_sum2_lpdf(array[] real y,int start, int end, vector mu,vector sigma){
//     return normal_lpdf(y[1:(1+end-start)]|mu[1:(1+end-start)],sigma[1:(1+end-start)]);
//   }
// }
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
  to_vector(gamma) ~ std_normal();
  if(type == 1) to_vector(y) ~ normal(Xb + Z*to_vector(gamma), sqrt(sigma));
  if(type == 2) to_vector(log(y)) ~ normal(Xb + Z*to_vector(gamma), sqrt(sigma));
  if(type == 3){
    vector[N] logitmu = 1/(1+exp(-1*Xb - Z*to_vector(gamma)));
    to_vector(y) ~ beta(logitmu*sigma[1], (1-logitmu)*sigma[1]);
  } 
  if(type==4) to_vector(y) ~ gamma(1/sigma[1], 1/(sigma[1]*(Xb + Z*to_vector(gamma))));
  if(type==5) to_vector(y) ~ gamma(1/sigma[1], (Xb + Z*to_vector(gamma))/sigma[1]);
  if(type==6) to_vector(y) ~ gamma(1/sigma[1], 1/(sigma[1]*log(Xb + Z*to_vector(gamma))));

}


// functions {
//   real asymmetric_laplace_lpdf(vector y, vector mu, real sigma, real q){
//     int n = size(y);
//     vector[n] resid = (y - mu)/sigma;
//     vector[n] rho = (abs(resid) + (2*q - 1)*resid)*0.5;
//     real ll = n*log(q) + n*log(1-q) - n*log(sigma) - sum(rho);
//     return ll;
//   }
// }
data {
  int N_cont; // sample size
  int N_int;
  int N_binom;
  int Q; // columns of Z, size of RE terms
  vector[N_cont > N_int ? N_cont : N_int] Xb;
  matrix[N_cont > N_int ? N_cont : N_int,Q] Z;
  array[N_cont] real ycont;
  array[N_int] int yint;
  vector[N_cont] sigma;
  array[N_binom] int n;
  real<lower = 0, upper = 1> q;
  int type;
}
parameters {
  vector[Q] gamma;
}
model {
  gamma ~ std_normal();
  vector[N_cont > N_int ? N_cont : N_int] mu = Xb + Z*gamma;
  if(type == 1) to_vector(ycont) ~ normal(mu, sqrt(sigma));
  if(type == 2) to_vector(log(ycont)) ~ normal(mu, sqrt(sigma));
  if(type == 3){
    vector[N_cont] logitmu;
    vector[N_cont] logitmu1;
    for(i in 1:N_cont){
      logitmu[i] = exp(mu[i])/(1+exp(mu[i]));
      logitmu1[i] = 1 - logitmu[i];
    }
    to_vector(ycont) ~ beta(logitmu*sigma[1], logitmu1*sigma[1]);
  } 
  if(type==4) to_vector(ycont) ~ gamma(1/sigma[1], 1/(sigma[1]*mu));
  if(type==5) to_vector(ycont) ~ gamma(1/sigma[1], mu/sigma[1]);
  if(type==6) to_vector(ycont) ~ gamma(1/sigma[1], 1/(sigma[1]*log(mu)));

  if(type==7) yint ~ bernoulli_logit(mu);
  if(type==8) yint ~ bernoulli(exp(mu));
  if(type==9) yint ~ bernoulli(mu);
  if(type==10) yint ~ bernoulli(Phi_approx(mu));
  
  if(type==11) yint ~ binomial_logit(n ,mu);
  if(type==12) yint ~ binomial(n,exp(mu));
  if(type==13) yint ~ binomial(n,mu);
  if(type==14) yint ~ binomial(n,Phi_approx(mu));
  
  if(type==15) yint ~ poisson_log(mu);
  
  // if(type==16) to_vector(ycont) ~ asymmetric_laplace(mu, sigma[1], q);
  // if(type==17) to_vector(ycont) ~ asymmetric_laplace(exp(mu), sigma[1], q);
  // if(type==18) to_vector(ycont) ~ asymmetric_laplace(inv_logit(mu), sigma[1], q);
  // if(type==19) to_vector(ycont) ~ asymmetric_laplace(Phi_approx(mu), sigma[1], q);
  // if(type==20) to_vector(ycont) ~ asymmetric_laplace(1/(mu), sigma[1], q);
}

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
  if(type==1) y ~ bernoulli_logit(Xb + Z*to_vector(gamma));
  if(type==2) y ~ bernoulli(exp(Xb + Z*to_vector(gamma)));
  if(type==3) y ~ bernoulli(Xb + Z*to_vector(gamma));
  if(type==4) y ~ bernoulli(Phi_approx(Xb + Z*to_vector(gamma)));
  
  if(type==5) y ~ binomial_logit(n ,Xb + Z*to_vector(gamma));
  if(type==6) y ~ binomial(n,exp(Xb + Z*to_vector(gamma)));
  if(type==7) y ~ binomial(n,Xb + Z*to_vector(gamma));
  if(type==8) y ~ binomial(n,Phi_approx(Xb + Z*to_vector(gamma)));
  
  if(type==9) y ~ poisson_log(Xb + Z*to_vector(gamma));
}


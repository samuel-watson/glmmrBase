functions {
  real asymmetric_laplace_lpdf(vector y, vector mu, real sigma, real q){
    int n = size(y);
    vector[n] resid = y - mu;
    vector[n] rho = (abs(resid) + (2*q - 1)*resid)*0.5;
    real ll = n*log(q) + n*log(1-q) - n*log(sigma) - sum(rho);
    return ll;
  }
}
data {
  int N; // sample size
  int Q; // columns of Z, size of RE terms
  vector[N] Xb;
  matrix[N,Q] Z;
  vector[N] y;
  real<lower = 0> var_par;
  real<lower = 0, upper = 1> q;
  int type;
}
parameters {
  array[Q] real gamma;
}
model {
  to_vector(gamma) ~ std_normal();
  if(type==1) y ~ asymmetric_laplace(Xb + Z*to_vector(gamma), var_par, q);
  if(type==2) y ~ asymmetric_laplace(exp(Xb + Z*to_vector(gamma)), var_par, q);
  if(type==3) y ~ asymmetric_laplace(inv_logit(Xb + Z*to_vector(gamma)), var_par, q);
  if(type==4) y ~ asymmetric_laplace(Phi_approx(Xb + Z*to_vector(gamma)), var_par, q);
  if(type==5) y ~ asymmetric_laplace(1/(Xb + Z*to_vector(gamma)), var_par, q);
}


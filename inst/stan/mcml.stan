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
  real constr_zero;
}
parameters {
  vector[Q] gamma;
}
model {
  gamma ~ std_normal();
  //sum(gamma) ~ normal(0, constr_zero*Q);
  //vector[N_cont > N_int ? N_cont : N_int] mu = Xb + Z*gamma;
  if(type == 1){
    target += normal_id_glm_lupdf(to_vector(ycont)|Z,Xb,gamma,sigma);
  } else if(type == 2){
    to_vector(log(ycont)) ~ normal(Xb + Z*gamma, sigma);
  } else if(type == 3){
    vector[N_cont > N_int ? N_cont : N_int] mu = Xb + Z*gamma;
    vector[N_cont] logitmu;
    vector[N_cont] logitmu1;
    for(i in 1:N_cont){
      logitmu[i] = exp(mu[i])/(1+exp(mu[i]));
      logitmu1[i] = 1 - logitmu[i];
    }
    to_vector(ycont) ~ beta(logitmu*sigma[1], logitmu1*sigma[1]);
  } else if(type==4){
    to_vector(ycont) ~ gamma(1/sigma[1], 1/(sigma[1]*(Xb + Z*gamma)));
  } else if(type==5) {
    to_vector(ycont) ~ gamma(1/sigma[1], (Xb + Z*gamma)/sigma[1]);
  } else if(type==6){
    to_vector(ycont) ~ gamma(1/sigma[1], 1/(sigma[1]*log(Xb + Z*gamma)));
  } else if(type==7) {
    target += bernoulli_logit_glm_lupmf(yint|Z,Xb,gamma);
  } else if(type==8) {
    yint ~ bernoulli(exp(Xb + Z*gamma));
  } else if(type==9){
    yint ~ bernoulli(Xb + Z*gamma);
  } else if(type==10){
    yint ~ bernoulli(Phi_approx(Xb + Z*gamma));
  } else if(type==11){
    yint ~ binomial_logit(n ,Xb + Z*gamma);
  } else if(type==12){
    yint ~ binomial(n,exp(Xb + Z*gamma));
  } else if(type==13){
    yint ~ binomial(n,Xb + Z*gamma);
  } else if(type==14){
    yint ~ binomial(n,Phi_approx(Xb + Z*gamma));
  } else if(type==15){
    target += poisson_log_glm_lupmf(yint|Z,Xb,gamma);
  }
}

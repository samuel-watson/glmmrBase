[![cran version](http://www.r-pkg.org/badges/version/glmmrBase)](  https://CRAN.R-project.org/package=glmmrBase)

# glmmrBase
(Version 0.6.1)
R package for the specification, analysis, and fitting of generalised linear mixed models. Includes model fitting using full maximum likelihood with Markov Chain Monte Carlo Maximum Likelihood (MCML) and Laplacian approximation and provides robust and bias-corrected standard error options. Allows for non-linear functions of data and parameters in the fixed effects, and includes a wide range of covariance functions, including autoregressive, exponential, and Matern, which can be arbitrarily combined. The R model classes provide a wide array of functionality including power analysis, data simulation, and generation of a wide range of relevant matrices and products.

The [project home page](https://samuel-watson.github.io/glmmr-web/) is currently being built - the tutorials are out of data with the most recent version of glmmrBase.

## Installation
The package is available on CRAN, or the most up-to-date version can be installed from this repository in R using `devtools::install_github("samuel-watson/glmmrBase")`. A pre-compiled binary is also available with each release on this page. 

### Building from source
It is recommended to build from source with the flags `-fno-math-errno -O3` or `-Ofast` for gcc, this will cut the time to run many functions. One way to do this is to set CPP_FLAGS in `~/.R/Makevars`. Another alternative is to download the package source `.tar.gz` file and run from the command line 
```
R CMD INSTALL --configure-args="CPPFLAGS=-fno-math-errno -O3" glmmrBase_0.5.4.tar.gz
```
Yet another alternative would be to clone the package from this repository and edit the `makevars` or `makevars.win` file to include the flags. The package can then be installed using `devtools::install`.

## Generalised linear mixed models
A generalised linear mixed model (GLMM) has a mean function for observation $i$ is

$$
\mu_i = \mathbf{x}_i\beta + \mathbf{z}_i \mathbf{u}
$$

where $\mathbf{x}_i$ is the $i$th row of matrix $X$, which is a $n \times P$ matrix of covariates, $\beta$ is a vector of parameters, $\mathbf{z}_i$ is the $i$ th row 
of matrix $Z$, which is the $n \times Q$ "design matrix" for the random effects, $\mathbf{u} \sim N(0,D)$, where $D$ is the $Q \times Q$ covariance matrix of the 
random effects terms that depends on parameters $\theta$, and $\mathbf{\mu}$ is the $n$-length vector of mean values. The assumed data generating process for the study 
is 

$$
y_i \sim G(h(\mu_i);\phi)
$$

where $\mathbf{y}$ is a $n$-length vector of outcomes $y_i$, $G$ is a distribution, $h(.)$ is the link function, and $\phi$ additional scale parameters to complete the 
specification. 

## Model specification
A `Model` class object can be created by specifying, for example:
```
R> model <- Model$new(formula = int + b_1*exp(b_2*x) -1 + (1|gr(j)*ar0(t)),
R>                    data = df,
R>                    family = gaussian())
```

A reproducible example is given below:
```
# Example: simulating and fitting a binomial-logit model with AR-1 covariance with continuous time 
# using full maximum likelihood
require(glmmrBase)

# we will assume a binomial-logit model

# simulate data
# this generates data for a study with clusters crossed with time periods with nested
# individuals, i.e. a stepped-wedge cluster trial with cross-sectional sampling
# int is the intervention indicator
df <- nelder(~(cl(6)*t(7))>ind(10))
df$int <- 0
df[df$cl > df$t, 'int'] <- 1
df$time <- runif(nrow(df),df$t-1,df$t)/7

# create a new model - we are providing parameter values for the fixed effects and covariance
#intercept -1.5, random time effects, and a treatment effect of 0.3 (log odds ratio)
mod <- Model$new(
  formula = ~ factor(t) + int -1 + (1|gr(cl)*ar1(time)),
  mean = c(-1.5,rnorm(6,0,0.5),0.3),
  covariance = c(0.2,0.3),
  family = binomial(link="logit"),
  data = df
)

y <- mod$sim_data() # simulates data from the model using model 1 - useful for simulation projects requiring GLMMs easy to specify complex models!

# fit the model
mod$set_trace(1) # for more detailed output
fit <- mod$MCML(y=y) # default fitting algorithm is SAEM
fit

# the above uses the GLS SEs, we can retrieve Kenward Roger for example if we wanted using:
kr <- mod$small_sample_correction(type="KR")
sqrt(diag(kr$vcov_beta))
kr$dof #degrees of freedom

# and if it is of interest, we can use MCMC samples to get marginal effects including marginal log RR
mod$marginal(x = "int",
             type = "dydx",
             re = "average",
             se = "GLS",
             average = c("t_1","t_2","t_3","t_4","t_5","t_6","t_7"))

```


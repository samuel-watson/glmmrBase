[![cran version](http://www.r-pkg.org/badges/version/glmmrBase)](  https://CRAN.R-project.org/package=glmmrBase)

# glmmrBase
(Version 0.4.2)
R package for the specification, analysis, and fitting of generalised linear mixed models. Includes model fitting using a Laplace approximation or full maximum likelihood with Markov Chain Monte Carlo Maximum Likelihood and provides robust and bias-corrected standard error options. Allows for non-linear functions of data and parameters in the fixed effects, and includes a wide range of covariance functions, including autoregressive, exponential, and Matern, which can be arbitrarily combined. The R model classes provide a wide array of functionality including power analysis, data simulation, and generation of a wide range of relevant matrices and products.

The full details and tutorials have moved to the [project home page](https://samuel-watson.github.io/glmmr-web/).

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


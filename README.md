# glmmrBase
R package to support the specification of generalised linear mixed models using the R6 object-orientated class system. 

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

## Generating data
The package includes the function `nelder()`, which we use to generate data for the examples below. Nelder (1965) suggested a simple notation that could express 
a large variety of different blocked designs. The notation was proposed in the context of split-plot experiments for agricultural research, where researchers 
often split areas of land into blocks, sub-blocks, and other smaller divisions, and apply different combinations of treatments. However, the notation is useful 
for expressing a large variety of experimental designs with correlation and clustering including cluster trials, cohort studies, and spatial and temporal 
prevalence surveys. We have included the function \code{nelder()} that generates a data frame of a design using the notation. 

There are two operations:
* `>` (or $\to$ in Nelder's notation) indicates "clustered in".
* `*` (or $\times$ in Nelder's notation) indicates a crossing that generates all combinations of two factors.

The function takes a formula input indicating the name of the variable and a number for the number of levels, such as `abc(12)`. 
So for example `~cl(4) > ind(5)` means in each of five levels of `cl` there are five levels of `ind`, and the individuals are different between clusters. 
The formula `~cl(4) * t(3)` indicates that each of the four levels of `cl` are observed for each of the three levels of `t`. Brackets are used to indicate the 
order of evaluation.

## Specifying covariance
The specification of a covariance object requires three inputs: a formula, data, and parameters. A new instance of each class can be generated with the 
`$new()` function, for example `Covariance\$new(...)`. 

A covariance function is specified as an additive formula made up of components with structure `(1|f(j))`. The left side of the vertical bar specifies the 
covariates in the model that have a random effects structure. The right side of the vertical bar specify the covariance function `f` for that term using variable 
named in the data `j`. Covariance functions on the right side of the vertical bar are multiplied together, i.e., `(1|f(j)*g(t))`. The table below shows the currently 
implemented covariance functions

| Function | $Cov(x_i,x_{i'})$ | $\theta$ | `Code` |
|----------|-------------------|----------|--------|
| Identity/ Group membership | $\theta_1^2 \mathbf{1}(x_i = x_{i'})$ | $\theta_1 > 0$ | `gr(x)` |
| Exponential | $\theta_1 \text{exp}(- \vert x_i - x_{i'}\vert / \theta_2 )$ | $\theta_1,\theta_2 > 0$ | `fexp(x)`|
| | $\text{exp}(- \vert x_i - x_{i'}\vert /\theta_1)$ | $\theta_1 > 0$ | `fexp0(x)` |
| Squared Exponential | $\theta_1 \text{exp}(- (\vert x_i - x_{i'}\vert / \theta_2)^2)$ | $\theta_1,\theta_2 > 0$ | `sqexp(x)` |
| | $\text{exp}(-( \vert x_i - x_{i'}\vert/\theta_1)^2 )$ | $\theta_1 > 0$ | `sqexp0(x)` |
| Autoregressive order 1 | $\theta_1^{\vert x_i - x_{i'} \vert}$ | $0 < \theta_1 < 1$ | `ar1(x)` |
| Bessel | $K_{\theta_1}(x)$ | $\theta_1$ > 0 | `bessel(x)` |
| Matern | $\frac{2^{1-\theta_1}}{\Gamma(\theta_1)}\left( \sqrt{2\theta_1}\frac{x}{\theta_2} \right)^{\theta_1} K_{\theta_1}\left(\sqrt{2\theta_1}\frac{x}{\theta_2})\right)$ | $\theta_1,\theta_2 > 0$ | `matern(x)` |
| Compactly supported* | || |
| Wendland 0 | $\theta_1(1-y)^{\theta_2}, 0 \leq y \leq 1; 0, y \geq 1$ | $\theta_1>0, \theta_2 \geq (d+1)/2$ | `wend0(x)` |
| Wendland 1 | $\theta_1(1+(\theta_2+1)y)(1-y)^{\theta_2+1}, 0 \leq y \leq 1; 0, y \geq 1$ | $\theta_1>0, \theta_2 \geq (d+3)/2$ | `wend1(x)` |
| Wendland 2 | $\theta_1(1+(\theta_2+2)y + \frac{1}{3}((\theta_2+2)^2 - 1)y^2)(1-y)^{\theta_2+2}, 0 \leq y \leq 1 $ | $\theta_1>0,\theta_2 \geq (d+5)/2$ | `wend1(x)` |
| | $0, y \geq 1$ | | |
| Whittle-Matern $\times$ Wendland** | $\theta_1\frac{2^{1-\theta_2}}{\Gamma(\theta_2)}y^{\theta_2}K_{\theta_2}(y)(1+\frac{11}{2}y + \frac{117}{12}y^2)(1-y), 0 \leq y \leq 1; 0, y \geq 1$ | $\theta_1,\theta_2 > 0$ | `prodwm(x)` |
| Cauchy $\times$ Bohman*** | $\theta_1(1-y^{\theta_2})^{-3}\left( (1-y)\text{cos}(\pi y)+\frac{1}{\pi}\text{sin}(\pi y) \right), 0 \leq y \leq 1; 0, y \geq 1$ | $\theta_1>0, 0 \leq \theta_2 \leq 2$ | `prodcb(x)` |
| Exponential $\times$ Kantar**** | $\theta_1\exp{(-y^{\theta_2})}\left( (1-y)\frac{\sin{(2\pi y)}}{2\pi y} + \frac{1}{\pi}\frac{1-\cos{(2\pi y)}}{2\pi y} \right), 0 \leq y \leq 1$ | $\theta_1,\theta_2 > 0$ | `prodek(x)` |
| | $0, y \geq 1$ | | |

$\vert . \vert$ is the Euclidean distance. $K_a$ is the modified Bessel function of the second kind. 
*Variable $y$ is defined as $x/r$ where $r$ is the effective range. For the compactly supported functions $d$ is the number of dimensions in `x`. 
**Permissible in one or two dimensions. ***Only permissible in one dimension. ****Permissible in up to three dimensions.

One combines functions to provide the desired covariance function. For example, for a stepped-wedge cluster trial we could consider the 
standard specification with an exchangeable random effect for the cluster level, and a separate exchangeable random effect for the cluster-period, 
which would be `~(1|gr(j))+(1|gr(j,t))` or `~(1|gr(j))+(1|gr(j)*gr(t))`. Alternatively, we could consider an autoregressive cluster-level random effect 
that decays exponentially over time so that, for persons $i$ in cluster $j$ at time $t$, $Cov(y_{ijt},y_{i'jt}) = \theta_1$, for $i\neq i'$, 
$Cov(y_{ijt},y_{i'jt'}) = \theta_1 \theta_2^{\vert t-t' \vert}$ for $t \neq t'$, and $Cov(y_{ijt},y_{i'j't}) = 0$ for $j \neq j'$. This function would be 
specified as `~(1|gr(j)*ar1(t))`.

Parameters are provided to the covariance function as a vector. The covariance functions described in the Table have different parameters $\theta$, and a value is 
required to be provided to generate the matrix $D$ and related objects for analyses and which serve as starting values for model fitting. The elements of the 
vector correspond to each of the functions in the covariance formula in the order they are written.

A new covariance object can then be created as such

```
R> df <- nelder(~ (j(10)* t(5)) > ind(10))
R> cov <- Covariance$new(formula = ~(1|gr(j)*ar1(t)),
R>                       parameters = c(0.05,0.8),
R>                       data= df)
```

where a compactly supported function is used, then the effective range parameters should be provided in the order the function appears in the formula.

```
R> cov <- Covariance$new(formula = ~(1|prodwm(x,y)),
R>                       parameters = c(0.25,0.5),
R>                       eff_range = 0.5,
R>                       data= df)
```

## Mean function specification
Specification of the mean function follows standard model formulae in R. A vector of values of the mean function parameters is required to complete the model 
specification along with the distribution as a standard R family object. A complete specification is thus:
```
R> mf <- MeanFunction$new(formula = ~ factor(t)+ int - 1,
R>                        data = df,
R>                        parameters = rep(0,6),
R>                        family = gaussian())
```

Note that `factor` in this function does not drop one level, unlike standard R formulae, so removing the intercept is required to prevent a collinearity problem. 

## Model specification
A model is simply a covariance object and a mean function object:
```
R> model <- Model$new(covariance = cov,
R>                   mean.function = mf,
R>                  var_par = 1)
```
For Gaussian models, and other distributions requiring an additional scale parameter $\phi$, one must also specify the option `var_par` which is the 
conditional variance $\phi = \sigma$ at the individual level. The default value is 1. Alternatively, one can specify a design by providing the list of 
arguments directly to `covariance` and `mean.function` instead of model objects.

## Accessing computed elements
Each class holds associated matrices and has member functions to compute basic summaries and analyses. The `Matrix` package is used
for matrix operations in R. For example, a `Covariance` object holds the matrix
$D$
```
R> cov$D[1:10,1:10]
10 x 10 sparse Matrix of class "dsCMatrix"
                                                                                       
 [1,] 0.002500 0.00200 0.0016 0.00128 0.001024 .        .       .      .       .       
 [2,] 0.002000 0.00250 0.0020 0.00160 0.001280 .        .       .      .       .       
 [3,] 0.001600 0.00200 0.0025 0.00200 0.001600 .        .       .      .       .       
 [4,] 0.001280 0.00160 0.0020 0.00250 0.002000 .        .       .      .       .       
 [5,] 0.001024 0.00128 0.0016 0.00200 0.002500 .        .       .      .       .       
 [6,] .        .       .      .       .        0.002500 0.00200 0.0016 0.00128 0.001024
 [7,] .        .       .      .       .        0.002000 0.00250 0.0020 0.00160 0.001280
 [8,] .        .       .      .       .        0.001600 0.00200 0.0025 0.00200 0.001600
 [9,] .        .       .      .       .        0.001280 0.00160 0.0020 0.00250 0.002000
[10,] .        .       .      .       .        0.001024 0.00128 0.0016 0.00200 0.002500

```

## Use of `glmmrBase` in other packages
This package provides a foundation for other packages providing analysis or estimation of generalised linear mixed models. For example, we have the `glmmrMCML` package,
which provides Markov Chain Monte Carlo Maximum Likelihood estimations for these models, and `glmmrOptim`, which finds approximate optimal designs based on these
models. New classes can be defined that inherit from the classes included in this package. `glmmrMCML` defines the `modelMCML` class that adds the member 
function `MCML`. Then the new functions can access the model elements, such as covariance matrices, and benefit from automatic updating of these elements when specifications or parameters change. 
As an example we can define a new class that has a member function that returns the determinant of the matrix $D$:
```
R> CovDet <- R6::R6Class("CovDet",
R>                        inherit = Covariance,
R>                        public = list(
R>                        det = function(){
R>                          return(Matrix::determinant(self$D))
R>                        }))
R> cov <- CovDet$new(formula = ~(1|gr(j)*ar1(t)),
R>                       parameters = c(0.05,0.8),
R>                       data= df)
R> cov$det()
$modulus
[1] -340.4393
attr(,"logarithm")
[1] TRUE

$sign
[1] 1

attr(,"class")
[1] "det"
```


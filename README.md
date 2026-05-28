# glmmrBase v1.4.1

Monte Carlo maximum likelihood estimation and analysis of mixed models. This readme to be updated.
Example code:

```
# Simulated trial data example using REML
# we set starting values to prevent errors on example checking - these are not strictly necessary
set.seed(123)
data(SimTrial,package = "glmmrBase")
fit1 <- Model$new(
  formula = y ~ int + factor(t) - 1 + (1|grlog(cl)*ar0log(t)),
  data = SimTrial,
  family = gaussian(),
  covariance = log(c(0.05,0.7))
)$fit(reml = TRUE)

# Salamanders data example
data(Salamanders,package="glmmrBase")
model <- Model$new(
  mating~fpop:mpop-1+(1|grlog(mnum))+(1|grlog(fnum)),
  data = Salamanders,
  family = binomial(),
  covariance = log(c(0.05,0.7))
)
set.seed(125)
fit2 <- model$fit()

# Example using simulated data
#create example data with six clusters, five time periods, and five people per cluster-period
df <- nelder(~(cl(20)*t(10)) > ind(5))
# parallel trial design intervention indicator
df$int <- 0
df[df$cl > 10, 'int'] <- 1
# specify parameter values in the call for the data simulation below
des <- Model$new(
  formula= ~ factor(t) + int - 1 +(1|grlog(cl)*ar0log(t)),
  covariance = log(c(0.15,0.7)),
  mean = c(rep(0,10),0.2),
  data = df,
  family = binomial()
)
ysim <- des$sim_data() # simulate some data from the model
des$update_y(ysim)
set.seed(123)
fit2 <- des$fit()

# use of Gaussian process approximations
# simulate some data - binomial observation on [-1,1] x [-1,1]
set.seed(123)
df <- data.frame(
x = runif(100, -1, 1),
y = runif(100, -1, 1))
df$z <- rnorm(100)

df$outcome <- Model$new(
  ~ z + (1|matern1log(x, y)),
  data = df,
   family = binomial(),
  mean = c(1, 0.1),
  covariance = c(log(2), log(0.3)),
  trials = rep(10, nrow(df))
)$sim_data()

# we can fit the SPDE approximation using a mesh built by fmesher
df_pred <- expand.grid(x= seq(-1,1,by=0.05), y = seq(-1,1,by=0.05))
df_pred$z <- 0
mesh_data <- mesh_helper(unique(df[,1:2]), df_pred[,1:2], c(0.15, 0.75), 0.075, c(0.1,0.3))

mod <- Model$new(
  outcome ~ z + (1|spde_matern1log(x, y)),
  data = df,
  family = binomial(),
  trials = rep(10, nrow(df)),
  mesh = mesh_data[["data"]],
  covariance = log(c(0.5, 0.3))
)
set.seed(123)
fit1 <- mod$fit(niter = 50)

#generate predictions
pred1 <- mod$predict(newdata = df_pred,mesh_A = mesh_data[["A_pred"]])

#' # Non-linear model fitting example using the example provided by nlmer in lme4
data(Orange, package = "datasets")

# the lme4 example:
startvec <- c(Asym = 200, xmid = 725, scal = 350)
(nm1 <- lme4::nlmer(circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree,
              Orange, start = startvec))

Orange <- as.data.frame(Orange)
Orange$Tree <- as.numeric(Orange$Tree)

# Here we can specify the model as a function.

model <- Model$new(
  circumference ~ Asym/(1 + exp((xmid - (age))/scal)) - 1 + (Asym|grlog(Tree)),
  data = Orange,
  family = gaussian(),
  mean = c(200,725,350),
  covariance = log(c(500)),
  var_par = 50
)
set.seed(123)
nfit <- model$fit(niter = 100)

summary(nfit)
suppressWarnings(summary(nm1)) # lme4 reports Hessian warnings
```

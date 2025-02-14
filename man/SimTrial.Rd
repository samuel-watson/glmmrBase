% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/extrafunctions.R
\docType{data}
\name{SimTrial}
\alias{SimTrial}
\title{Simulated data from a stepped-wedge cluster trial}
\description{
Simulated data from a stepped-wedge cluster trial
}
\examples{
#Data were generated with the following code:
SimTrial <- nelder(~ (cl(10)*t(7))>i(10))
SimTrial$int <- 0
SimTrial[SimTrial$t > SimTrial$cl,'int'] <- 1

model <- Model$new(
  formula = ~ int + factor(t) - 1 + (1|gr(cl)*ar1(t)),
  covariance = c(0.05,0.8),
  mean = rep(0,8),
  data = SimTrial,
  family = gaussian()
)

SimTrial$y <- model$sim_data()
}

\name{glmmrBase-package}
\alias{glmmrBase-package}
\alias{glmmrBase}
\docType{package}
\title{
\packageTitle{glmmrBase}
}
\description{
\packageDescription{glmmrBase}
\code{glmmrBase} provides functions for specifying, analysing, fitting, and simulating mixed models including linear, generalised linear, and models non-linear in fixed effects. 
}
\section{Differences between \pkg{glmmrBase} and \pkg{lme4} and related packages.}{
\pkg{glmmrBase} is intended to be a broad package to support statistical work with generalised linear mixed models. While there are Laplace Approximation methods in the package, it does not intend to replace or supplant popular mixed model packages like \pkg{lme4}. Rather it provides broader functionality around simulation and analysis methods, and a range of model fitting algorithms not found in other mixed model packages. The key features are:
\itemize{
  \item Stochastic maximum likelihood methods. The most widely used methods for mixed model fitting are penalised quasi-likelihood, Laplace approximation, and Gaussian quadrature methods. These methods are widely available in other packages. We provide Markov Chain Monte Carlo (MCMC) Maximum Likelihood and Stochastic Approximation Expectation Maximisation algorithms for model fitting, with various features. These algorithms approximate the intractable GLMM likelihood using MCMC and so can provide an arbitrary level of precision. These methods may provide better maximum likelihood performance than other approximations in settings with high-dimensional or complex random effects, small sample sizes, or non-linear models.
  \item Flexible support for a wide range of covariance functions. The support for different covariance functions can be limited in other packages. For example, \pkg{lme4} only provides exchangeable random effects structures. We include multiple different functions that can be combined arbitrarily.
  \item We similarly use efficient linear algebra methods with the \pkg{Eigen} package along with \pkg{Stan} to provide MCMC sampling.
  \item Gaussian Process approximations. We include Hibert Space and Nearest Neighbour Gaussian Process approximations for high dimensional random effects.
  \item The \code{Model} class includes methods for power estimation, data simulation, MCMC sampling, and calculation of a wide range of matrices and values associated with the models.
  \item We include natively a range of small sample corrections to information matrices, including Kenward-Roger, Box, Satterthwaite, and others, which typically require add-on packages for \pkg{lme4}.
  \item The package provides a flexible class system for specifying mixed models that can be incorporated into other packages and settings. The linked package \pkg{glmmrOptim} provides optimal experimental design algorithms for mixed models.
  \item (New in version 0.9.1) The package includes functions to replicate the functionality of \pkg{lme4}, \link[glmmrBase]{mcml_lmer} and \link[glmmrBase]{mcml_glmer}, which will also accept \pkg{lme4} syntax.
  \item (New in version 0.10.1) The package also provides mixed quantile regression models estimated using the stochastic maximum likelihood algorithms described above. These models specify an asymmetric Laplace distribution for the likelihood and integrate with the other features of the package described above.
}
}
\section{Package development}{
The package is still in development and there may still be bugs and errors. While we do not expect the general user interface to change there may be changes to the underlying library as well as new additions and functionality.
}
\author{
\packageAuthor{glmmrBase}

Maintainer: \packageMaintainer{glmmrBase}
}
\keyword{ package }



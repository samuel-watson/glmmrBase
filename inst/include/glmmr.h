#pragma once

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS 

#include "glmmr/general.h"
#include "glmmr/maths.h"
#include "glmmr/formula.hpp"
#include "glmmr/covariance.hpp"
#include "glmmr/linearpredictor.hpp"
#include "glmmr/model.hpp"
#include "glmmr/modelbits.hpp"
#include "glmmr/openmpheader.h"
#include "glmmr/nngpcovariance.hpp"
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

typedef glmmr::Covariance covariance;
typedef glmmr::nngpCovariance nngp;
typedef glmmr::LinearPredictor xb;
typedef glmmr::ModelBits<covariance, xb> bits;
typedef glmmr::Model<bits > glmm;
typedef glmmr::Model<glmmr::ModelBits<nngp, xb> > glmm_nngp;
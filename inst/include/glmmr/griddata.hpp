#pragma once

#include "general.h"

namespace glmmr {

using namespace Eigen;

class griddata {
public:
  ArrayXXd X = ArrayXXd::Constant(1,1,1); // centroids
  int N; // number of cells
  ArrayXXi NN = ArrayXXi::Constant(1,1,1);
  int m = 10;
  griddata(){};
  griddata(const ArrayXXd& X_) : X(X_), N(X_.rows()) {};
  griddata(const ArrayXXd& X_, int M) : X(X_), N(X_.rows()) {genNN(M);};
  griddata(const glmmr::griddata& g) : X(g.X), N(g.N) {};
  ArrayXi top_i_pq(const ArrayXd& v, int n);
  void genNN(int M);
  void setup(const ArrayXXd& X_);
  void setup(const ArrayXXd& X_, int M);
};
}


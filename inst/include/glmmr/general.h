#ifndef GENERAL_H
#define GENERAL_H

#include <RcppEigen.h>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cmath>
#include "algo.h"

typedef std::vector<std::string> strvec;
typedef std::vector<int> intvec;
typedef std::vector<double> dblvec;
typedef std::vector<strvec> strvec2d;
typedef std::vector<dblvec> dblvec2d;
typedef std::vector<intvec> intvec2d;
typedef std::vector<dblvec2d> dblvec3d;
typedef std::vector<intvec2d> intvec3d;

namespace glmmr {
const static std::unordered_map<std::string, double> nvars = {  
  {"gr", 1},
  {"ar1", 1},
  {"fexp0", 1}
};

const static intvec xvar_rpn = {0,1,4,0,1,4,5};
}

#endif
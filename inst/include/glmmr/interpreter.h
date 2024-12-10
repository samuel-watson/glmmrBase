#pragma once

#include "general.h"
#include "calculator.hpp"

namespace glmmr {

inline std::vector<Do> interpret_re(const CovFunc& fn);

//add in the indexes for each function
inline intvec interpret_re_par(const CovFunc& fn,
                               const int col_idx,
                               const intvec& par_idx);

inline void re_linear_predictor(glmmr::calculator& calc,
                                const int Q);

inline void re_log_likelihood(glmmr::calculator& calc,
                                const int Q);

inline void linear_predictor_to_link(glmmr::calculator& calc,
                                     const Link link);



}

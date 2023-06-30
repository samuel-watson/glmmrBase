#include <glmmr.h>

using namespace Rcpp;

//' Disable or enable parallelised computing
//' 
//' By default, the package will use multithreading for many calculations if OpenMP is 
//' available on the system. For multi-user systems this may not be desired, so parallel
//' execution can be disabled with this function.
//' 
//' @param parallel_ Logical indicating whether to use parallel computation (TRUE) or disable it (FALSE)
//' @param cores_ Number of cores for parallel execution
//' @return None, called for effects
// [[Rcpp::export]]
void setParallel(SEXP parallel_, int cores_ = 2){
  bool parallel = as<bool>(parallel_);
  if(OMP_IS_USED){
    int a, b; // needed for defines on machines without openmp
    if(!parallel){
      a = 0;
      b = 1;
      omp_set_dynamic(a); 
      omp_set_num_threads(b);
      Eigen::setNbThreads(b);
    } else {
      a = 1;
      b = cores_;
      omp_set_dynamic(a); 
      omp_set_num_threads(b);
      Eigen::setNbThreads(b);
    }
  } 
}
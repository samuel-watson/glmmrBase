#include <glmmr.h>

using namespace Rcpp;

//' Disable or enable parallelised computing
//' 
//' By default, the package will use multithreading for many calculations if OpenMP is 
//' available on the system. For multi-user systems this may not be desired, so parallel
//' execution can be disabled with this function.
//' 
//' @param parallel_ Logical indicating whether to use parallel computation (TRUE) or disable it (FALSE)
//' @return None, called for effects
// [[Rcpp::export]]
void setParallel(SEXP parallel_){
  bool parallel = as<bool>(parallel_);
  if(OMP_IS_USED){
    if(!parallel){
      omp_set_dynamic(0); 
      omp_set_num_threads(1);
      Eigen::setNbThreads(1);
    } else {
      omp_set_dynamic(1); 
      omp_set_num_threads(omp_get_max_threads());
      Eigen::setNbThreads(omp_get_max_threads());
    }
  }
}
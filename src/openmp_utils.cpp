#include <glmmr.h>

using namespace Rcpp;

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
  Rcpp::Rcout << "\nThreads: " << omp_get_num_threads() << " OPEN MP: " << OMP_IS_USED << " max: " << omp_get_max_threads();
}
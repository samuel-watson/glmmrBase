#include <glmmr/modelbits.hpp>



template<>
glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor>::ModelBits(const std::string& formula_,
                                                                              const ArrayXXd& data_,
                                                                              const strvec& colnames_,
                                                                              std::string family_, 
                                                                              std::string link_) : 
  formula(formula_), 
  linear_predictor(formula,data_,colnames_),
  covariance(formula,data_,colnames_),
  data(data_.rows()),
  family(family_,link_) {
  covariance.linear_predictor_ptr(&linear_predictor);
};

template<>
glmmr::ModelBits<glmmr::nngpCovariance, glmmr::LinearPredictor>::ModelBits(const std::string& formula_,
                                                                                  const ArrayXXd& data_,
                                                                                  const strvec& colnames_,
                                                                                  std::string family_, 
                                                                                  std::string link_) : 
  formula(formula_), 
  linear_predictor(formula,data_,colnames_),
  covariance(formula,data_,colnames_),
  data(data_.rows()),
  family(family_,link_) {};

template<>
glmmr::ModelBits<glmmr::hsgpCovariance, glmmr::LinearPredictor>::ModelBits(const std::string& formula_,
                                                                                  const ArrayXXd& data_,
                                                                                  const strvec& colnames_,
                                                                                  std::string family_, 
                                                                                  std::string link_) : 
  formula(formula_), 
  linear_predictor(formula,data_,colnames_),
  covariance(formula,data_,colnames_),
  data(data_.rows()),
  family(family_,link_) {};

template<typename cov, typename linpred>
void glmmr::ModelBits<cov, linpred>::make_covariance_sparse(bool amd){
  covariance.set_sparse(true,amd);
}

template<typename cov, typename linpred>
void glmmr::ModelBits<cov, linpred>::make_covariance_dense(){
  covariance.set_sparse(false);
}
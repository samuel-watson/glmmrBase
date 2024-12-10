#include <glmmr.h>

// extern template class glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor>;
// extern template class glmmr::ModelBits<glmmr::hsgpCovariance, glmmr::LinearPredictor>;
// extern template class glmmr::ModelBits<glmmr::nngpCovariance, glmmr::LinearPredictor>;
// extern template class glmmr::MatrixField<Eigen::MatrixXd>;
// extern template class glmmr::RandomEffects<glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor> >;
// extern template class glmmr::RandomEffects<glmmr::ModelBits<glmmr::hsgpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::RandomEffects<glmmr::ModelBits<glmmr::nngpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::MatrixW<glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor> >;
// extern template class glmmr::MatrixW<glmmr::ModelBits<glmmr::hsgpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::MatrixW<glmmr::ModelBits<glmmr::nngpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::ModelMatrix<glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor> >;
// extern template class glmmr::ModelMatrix<glmmr::ModelBits<glmmr::hsgpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::ModelMatrix<glmmr::ModelBits<glmmr::nngpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::ModelOptim<glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor> >;
// extern template class glmmr::ModelOptim<glmmr::ModelBits<glmmr::hsgpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::ModelOptim<glmmr::ModelBits<glmmr::nngpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::ModelMCMC<glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor> >;
// extern template class glmmr::ModelMCMC<glmmr::ModelBits<glmmr::hsgpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::ModelMCMC<glmmr::ModelBits<glmmr::nngpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::Model<glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor> >;
// extern template class glmmr::Model<glmmr::ModelBits<glmmr::hsgpCovariance, glmmr::LinearPredictor> >;
// extern template class glmmr::Model<glmmr::ModelBits<glmmr::nngpCovariance, glmmr::LinearPredictor> >;

namespace Rcpp {
template<>
SEXP wrap(const VectorMatrix& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vec") = Rcpp::wrap(x.vec),
      Rcpp::Named("mat") = Rcpp::wrap(x.mat)
  ));
}

template<>
SEXP wrap(const MatrixMatrix& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("mat1") = Rcpp::wrap(x.mat1),
      Rcpp::Named("mat2") = Rcpp::wrap(x.mat2),
      Rcpp::Named("a") = Rcpp::wrap(x.a),
      Rcpp::Named("b") = Rcpp::wrap(x.b)
  ));
}

template<typename T1, typename T2> SEXP wrap( const std::pair<T1,T2>& _v ) {
  return Rcpp::List::create(
    Rcpp::Named("first")  = Rcpp::wrap<T1>( _v.first ),
    Rcpp::Named("second") = Rcpp::wrap<T2>( _v.second )
  );
};

template<glmmr::SE corr>
SEXP wrap(const CorrectionData<corr>& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vcov_beta") = Rcpp::wrap(x.vcov_beta),
      Rcpp::Named("vcov_theta") = Rcpp::wrap(x.vcov_theta),
      Rcpp::Named("dof") = Rcpp::wrap(x.dof)
  ));
}

template<>
SEXP wrap(const CorrectionData<glmmr::SE::KRBoth>& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("vcov_beta") = Rcpp::wrap(x.vcov_beta),
      Rcpp::Named("vcov_beta_second") = Rcpp::wrap(x.vcov_beta_second),
      Rcpp::Named("vcov_theta") = Rcpp::wrap(x.vcov_theta),
      Rcpp::Named("dof") = Rcpp::wrap(x.dof)
  ));
}

template<>
SEXP wrap(const BoxResults& x){
  return Rcpp::wrap(Rcpp::List::create(
      Rcpp::Named("dof") = Rcpp::wrap(x.dof),
      Rcpp::Named("scale") = Rcpp::wrap(x.scale),
      Rcpp::Named("test_stat") = Rcpp::wrap(x.test_stat),
      Rcpp::Named("p_value") = Rcpp::wrap(x.p_value)
  ));
}


}

using namespace Rcpp;

// [[Rcpp::export]]
SEXP near_semi_pd(SEXP mat_){
  Eigen::MatrixXd mat = as<Eigen::MatrixXd>(mat_);
  glmmr::Eigen_ext::near_semi_pd(mat);
  return wrap(mat);
}

// [[Rcpp::export]]
SEXP Covariance__submatrix(SEXP xp, int i){
  XPtr<nngp> ptr(xp);
  VectorMatrix result = ptr->submatrix(i);
  return wrap(result);
}

// [[Rcpp::export]]
void Model_hsgp__set_approx_pars(SEXP xp, SEXP m_, SEXP L_){
  std::vector<int> m = as<std::vector<int> >(m_);
  Eigen::ArrayXd L = as<Eigen::ArrayXd>(L_);
  XPtr<glmm_hsgp> ptr(xp);
  ptr->model.covariance.update_approx_parameters(m,L);
  ptr->reset_u();
  std::vector<double> theta = ptr->model.covariance.parameters_;
  ptr->model.covariance.update_parameters(theta);
}

// [[Rcpp::export]]
void Covariance_hsgp__set_approx_pars(SEXP xp, SEXP m_, SEXP L_){
  std::vector<int> m = as<std::vector<int> >(m_);
  Eigen::ArrayXd L = as<Eigen::ArrayXd>(L_);
  XPtr<hsgp> ptr(xp);
  ptr->update_approx_parameters(m,L);
}

// [[Rcpp::export]]
SEXP Model_hsgp__dim(SEXP xp){
  XPtr<glmm_hsgp> ptr(xp);
  int dim = ptr->model.covariance.dim;
  return wrap(dim);
}

// [[Rcpp::export]]
SEXP Model__aic(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.aic());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
SEXP Model__residuals(SEXP xp, int rtype = 2, bool conditional = true, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->matrix.residuals(rtype,conditional));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__get_log_likelihood_values(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.current_likelihood_values());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::pair<double,double> >(S));
}

// [[Rcpp::export]]
SEXP Model__u_diagnostic(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.u_diagnostic());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::pair<double,double> >(S));
}

// MarginType type, dydx, diff, ratio
// [[Rcpp::export]]
SEXP Model__marginal(SEXP xp, 
                     std::string x,
                     int margin = 0,
                     int re = 3,
                     int se = 0,
                     int oim = 0,
                     Nullable<std::vector<std::string> > at = R_NilValue,
                     Nullable<std::vector<std::string> > atmeans = R_NilValue,
                     Nullable<std::vector<std::string> > average = R_NilValue,
                     double xvals_first = 1,
                     double xvals_second = 0,
                     Nullable<std::vector<double> > atvals = R_NilValue,
                     Nullable<std::vector<double> > revals = R_NilValue,
                     int type = 0){
  
  glmmrType model(xp,static_cast<Type>(type));
  std::vector<std::string> atvar;
  std::vector<std::string> atmeansvar;
  std::vector<std::string> averagevar;
  std::vector<double> atxvals;
  std::vector<double> atrevals;
  if(at.isNotNull())atvar = as<std::vector<std::string> >(at);
  if(atmeans.isNotNull())atmeansvar = as<std::vector<std::string> >(atmeans);
  if(average.isNotNull())averagevar = as<std::vector<std::string> >(average);
  std::pair<double, double> xvals;
  xvals.first = xvals_first;
  xvals.second = xvals_second;
  if(atvals.isNotNull())atxvals = as<std::vector<double> >(atvals);
  if(revals.isNotNull())atrevals = as<std::vector<double> >(revals);
  
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->marginal(static_cast<glmmr::MarginType>(margin),
                                      x,atvar,atmeansvar,averagevar,
                                      static_cast<glmmr::RandomEffectMargin>(re),
                                      static_cast<glmmr::SE>(se),
                                      static_cast<glmmr::IM>(oim),
                                      xvals,atxvals,atrevals
                                      ));}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::pair<double,double> >(S));
}

// [[Rcpp::export]]
void Model__mcmc_set_lambda(SEXP xp, SEXP lambda_, int type = 0){
  double lambda = as<double>(lambda_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&lambda](auto ptr){ptr->mcmc.mcmc_set_lambda(lambda);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__reset_fn_counter(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->optim.reset_fn_counter();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__get_fn_counter(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->optim.fn_counter);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::pair<int,int> >(S));
}

// [[Rcpp::export]]
void Model__print_names(SEXP xp, bool data, bool parameters, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->model.linear_predictor.calc.print_names(data,parameters);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__mcmc_set_max_steps(SEXP xp, SEXP max_steps_, int type = 0){
  int max_steps = as<int>(max_steps_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&max_steps](auto ptr){ptr->mcmc.mcmc_set_max_steps(max_steps);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__set_sml_parameters(SEXP xp, bool saem_, int block_size = 20, double alpha = 0.8, bool pr_average = true, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){
      ptr->optim.control.saem = saem_;
      ptr->optim.control.alpha = alpha;
      ptr->re.mcmc_block_size = block_size;
      ptr->optim.control.pr_average = pr_average;
      if(!saem_){
        ptr->optim.ll_current.resize(block_size,NoChange);
        // ptr->optim.ll_previous.resize(block_size,NoChange);
      }
    }
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__ll_diff_variance(SEXP xp, bool beta, bool theta, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {return returnType(0);}, 
    [&](auto ptr){
      return returnType(ptr->optim.ll_diff_variance(beta,theta));
    }
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<double>(S));
}

// [[Rcpp::export]]
void Model__mcmc_set_refresh(SEXP xp, SEXP refresh_, int type = 0){
  int refresh = as<int>(refresh_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&refresh](auto ptr){ptr->mcmc.mcmc_set_refresh(refresh);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__mcmc_set_target_accept(SEXP xp, SEXP target_, int type = 0){
  double target = as<double>(target_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&target](auto ptr){ptr->mcmc.mcmc_set_target_accept(target);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__make_sparse(SEXP xp, bool amd = true, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [&](auto ptr){ptr->model.make_covariance_sparse(amd);}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
void Model__make_dense(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {}, 
    [](auto ptr){ptr->model.make_covariance_dense();}
  };
  std::visit(functor,model.ptr);
}

// [[Rcpp::export]]
SEXP Model__beta_parameter_names(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.parameter_names());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<std::string> >(S));
}

// [[Rcpp::export]]
SEXP Model__theta_parameter_names(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.covariance.parameter_names());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<std::string> >(S));
}

// // [[Rcpp::export]]
// SEXP Model__hessian_correction(SEXP xp, int type = 0){
//   glmmrType model(xp,static_cast<Type>(type));
//   auto functor = overloaded {
//     [](int) {  return returnType(0);}, 
//     [](auto ptr){return returnType(ptr->matrix.hessian_nonlinear_correction());}
//   };
//   auto S = std::visit(functor,model.ptr);
//   return wrap(std::get<MatrixXd>(S));
// }

// [[Rcpp::export]]
SEXP Model__any_nonlinear(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->model.linear_predictor.calc.any_nonlinear);}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<bool>(S));
}

// [[Rcpp::export]]
SEXP Model__sandwich(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.sandwich_matrix());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__infomat_theta(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.template information_matrix_theta<glmmr::IM::EIM>());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<Eigen::MatrixXd>(S));
}

// [[Rcpp::export]]
SEXP Model__kenward_roger(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.template small_sample_correction<glmmr::SE::KR, glmmr::IM::EIM>());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<CorrectionData<glmmr::SE::KR> >(S));
}

// [[Rcpp::export]]
SEXP Model__small_sample_correction(SEXP xp, int ss_type = 0, bool oim = false, int type = 0){
  using namespace glmmr;
  glmmrType model(xp,static_cast<Type>(type));
  SE corr = static_cast<SE>(ss_type);
  switch(corr){
    case SE::KR:
      {
        auto functor = overloaded {
          [](int) {  return returnType(0);}, 
          [&](auto ptr){
          if(oim){
            return returnType(ptr->matrix.template small_sample_correction<SE::KR,IM::OIM>());
          } else {
            return returnType(ptr->matrix.template small_sample_correction<SE::KR,IM::EIM>());
          }}
        };
        auto S = std::visit(functor,model.ptr);
        return wrap(std::get<CorrectionData<SE::KR> >(S));
        break;
      }
    case SE::KR2:
    {
      auto functor = overloaded {
        [](int) {  return returnType(0);}, 
        [&](auto ptr){
          if(oim){
            return returnType(ptr->matrix.template small_sample_correction<SE::KR2,IM::OIM>());
          } else {
            return returnType(ptr->matrix.template small_sample_correction<SE::KR2,IM::EIM>());
          }}
      };
      auto S = std::visit(functor,model.ptr);
      return wrap(std::get<CorrectionData<SE::KR2> >(S));
      break;
    }
    case SE::KRBoth:
    {
      auto functor = overloaded {
        [](int) {  return returnType(0);}, 
        [&](auto ptr){
          if(oim){
            return returnType(ptr->matrix.template small_sample_correction<SE::KRBoth,IM::OIM>());
          } else {
            return returnType(ptr->matrix.template small_sample_correction<SE::KRBoth,IM::EIM>());
          }}
      };
      auto S = std::visit(functor,model.ptr);
      return wrap(std::get<CorrectionData<SE::KRBoth> >(S));
      break;
    }
    case SE::Sat:
    {
      auto functor = overloaded {
        [](int) {  return returnType(0);}, 
        [&](auto ptr){
          if(oim){
            return returnType(ptr->matrix.template small_sample_correction<SE::Sat,IM::OIM>());
          } else {
            return returnType(ptr->matrix.template small_sample_correction<SE::Sat,IM::EIM>());
          }}
      };
      auto S = std::visit(functor,model.ptr);
      return wrap(std::get<CorrectionData<SE::Sat> >(S));
      break;
    }
      default:
      {
        Rcpp::stop("Not a valid small sample correction type");
      }
  }
  
}

// [[Rcpp::export]]
SEXP Model__box(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.box());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<BoxResults>(S));
}

// [[Rcpp::export]]
SEXP Model__cov_deriv(SEXP xp, int type = 0){
  glmmrType model(xp,static_cast<Type>(type));
  auto functor = overloaded {
    [](int) {  return returnType(0);}, 
    [](auto ptr){return returnType(ptr->matrix.sigma_derivatives());}
  };
  auto S = std::visit(functor,model.ptr);
  return wrap(std::get<std::vector<Eigen::MatrixXd> >(S));
}

// [[Rcpp::export]]
SEXP Model__predict(SEXP xp, SEXP newdata_,
                    SEXP newoffset_,
                    int m, int type = 0){
  Eigen::ArrayXXd newdata = Rcpp::as<Eigen::ArrayXXd>(newdata_);
  Eigen::ArrayXd newoffset = Rcpp::as<Eigen::ArrayXd>(newoffset_);
  
  glmmrType model(xp,static_cast<Type>(type));
  auto functor_re = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->re.predict_re(newdata,newoffset));}
  };
  auto functor_xb = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->model.linear_predictor.predict_xb(newdata,newoffset));}
  };
  auto S_re = std::visit(functor_re,model.ptr);
  auto S_xb = std::visit(functor_xb,model.ptr);
  VectorMatrix res = std::get<VectorMatrix>(S_re);
  Eigen::VectorXd xb = std::get<Eigen::VectorXd>(S_xb);
  Eigen::MatrixXd samps(newdata.rows(),m>0 ? m : 1);
  if(m>0){
    samps = glmmr::maths::sample_MVN(res,m);
  } else {
    samps.setZero();
  }
  
  return Rcpp::List::create(
    Rcpp::Named("linear_predictor") = wrap(xb),
    Rcpp::Named("re_parameters") = wrap(res),
    Rcpp::Named("samples") = wrap(samps)
  );
}

// [[Rcpp::export]]
SEXP Model__predict_re(SEXP xp, SEXP newdata_,
                    SEXP newoffset_,
                    int m, int type = 0){
  Eigen::ArrayXXd newdata = Rcpp::as<Eigen::ArrayXXd>(newdata_);
  Eigen::ArrayXd newoffset = Rcpp::as<Eigen::ArrayXd>(newoffset_);
  glmmrType model(xp,static_cast<Type>(type));
  auto functor_re = overloaded {
    [](int) {  return returnType(0);}, 
    [&](auto ptr){return returnType(ptr->re.predict_re(newdata,newoffset));}
  };
  auto S_re = std::visit(functor_re,model.ptr);
  VectorMatrix res = std::get<VectorMatrix>(S_re);
  
  return Rcpp::List::create(
    Rcpp::Named("re_parameters") = wrap(res)
  );
}

//' Automatic differentiation of formulae
//' 
//' Exposes the automatic differentiator. Allows for calculation of Jacobian and Hessian matrices 
//' of formulae in terms of specified parameters. Formula specification is as a string. Data items are automatically 
//' multiplied by a parameter unless enclosed in parentheses.
//' @param form_ String. Formula to differentiate specified in terms of data items and parameters. Any string not identifying 
//' a function or a data item names in `colnames` is assumed to be a parameter.
//' @param data_ Matrix. A matrix including the data. Rows represent observations. The number of columns should match the number 
//' of items in `colnames_`
//' @param colnames_ Vector of strings. The names of the columns of `data_`, used to match data named in the formula.
//' @param parameters_ Vector of doubles. The values of the parameters at which to calculate the derivatives. The parameters should be in the 
//' same order they appear in the formula.
//' @return A list including the jacobian and hessian matrices.
//' @examples
//' # obtain the Jacobian and Hessian of the log-binomial model log-likelihood. 
//' # The model is of data from an intervention and control group
//' # with n1 and n0 participants, respectively, with y1 and y0 the number of events in each group. 
//' # The mean is exp(alpha) in the control 
//' # group and exp(alpha + beta) in the intervention group, so that beta is the log relative risk.
//' hessian_from_formula(
//'   form_ = "(y1)*(a+b)+((n1)-(y1))*log((1-exp(a+b)))+(y0)*a+((n0)-(y0))*log((1-exp(a)))",
//'   data_ = matrix(c(10,100,20,100), nrow = 1),
//'   colnames_ = c("y1","n1","y0","n0"),
//'   parameters_ = c(log(0.1),log(0.5)))
//' @export
// [[Rcpp::export]]
SEXP hessian_from_formula(SEXP form_, 
                          SEXP data_,
                          SEXP colnames_,
                          SEXP parameters_){
  std::string form = Rcpp::as<std::string>(form_);
  Eigen::ArrayXXd data = Rcpp::as<Eigen::ArrayXXd>(data_);
  std::vector<std::string> colnames = Rcpp::as<std::vector<std::string> >(colnames_);
  std::vector<double> parameters = Rcpp::as<std::vector<double> >(parameters_);
  glmmr::calculator calc;
  calc.data.conservativeResize(data.rows(),NoChange);
  std::vector<char> formula_as_chars(form.begin(),form.end());
  bool outparse = glmmr::parse_formula(formula_as_chars,
                                       calc,
                                       data,
                                       colnames,
                                       calc.data);
  (void)outparse;
  std::reverse(calc.instructions.begin(),calc.instructions.end());
  std::reverse(calc.indexes.begin(),calc.indexes.end());
  if(calc.parameter_names.size() != parameters.size())throw std::runtime_error("Wrong number of parameters");
  calc.parameters = parameters;
  VectorMatrix result = calc.jacobian_and_hessian();
  return Rcpp::wrap(result);
}

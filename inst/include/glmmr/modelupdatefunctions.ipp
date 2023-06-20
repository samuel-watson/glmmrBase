#ifndef MODELUPDATEFUNCTIONS_IPP
#define MODELUPDATEFUNCTIONS_IPP

inline void glmmr::Model::update_u(const MatrixXd &u){
  if(u.rows()!=Q_)Rcpp::stop("u has wrong number of random effects");
  if(u.cols()!=u_.cols()){
    Rcpp::Rcout << "\nDifferent numbers of random effect samples";
    u_.resize(Q_,u.cols());
    zu_.resize(Q_,u.cols());
    size_m_array.resize(u.cols());
  }
  u_ = u;
  zu_ = ZL_*u_;
}

inline void glmmr::Model::update_W(){
  double nvar_par = 1.0;
  if(family_=="gaussian"){
    nvar_par *= var_par_*var_par_;
  } else if(family_=="Gamma"){
    nvar_par *= 1/var_par_;
  } else if(family_=="beta"){
    nvar_par *= (1+var_par_);
  } else if(family_=="binomial"){
    nvar_par *= 1/var_par_;
  }
  
  if(attenuate_){
    size_n_array = glmmr::maths::attenuted_xb(xb(),covariance_.Z(),covariance_.D(),link_);
  } else {
    size_n_array = xb();
  }
  W_ = glmmr::maths::dhdmu(size_n_array,family_,link_);
  W_.noalias() = (W_.array().inverse()).matrix();
  W_ *= 1/nvar_par;
}

inline void glmmr::Model::update_var_par(const double& v){
  var_par_ = v;
  calc_.var_par = v;
}



inline double glmmr::Model::log_prob(const VectorXd &v){
  VectorXd zu = ZL_ * v;
  VectorXd mu = xb() + zu;
  double lp1 = 0;
  double lp2 = 0;
#pragma omp parallel for reduction (+:lp1) num_threads(n_threads_)
  for(int i = 0; i < n_; i++){
    lp1 += glmmr::maths::log_likelihood(y_(i),mu(i),var_par_,flink);
  }
#pragma omp parallel for reduction (+:lp2) num_threads(n_threads_)
  for(int i = 0; i < v.size(); i++){
    lp2 += -0.5*v(i)*v(i); 
  }
  return lp1+lp2-0.5*v.size()*log(2*M_PI);
}



inline MatrixXd glmmr::Model::Zu(){
  return zu_;
}

inline MatrixXd glmmr::Model::Sigma(bool inverse){
  update_W();
  MatrixXd S = sigma_builder(0,inverse);
  return S;
}

inline MatrixXd glmmr::Model::information_matrix(){
  update_W();
  MatrixXd M = MatrixXd::Zero(P_,P_);
  for(int i = 0; i< sigma_blocks_.size(); i++){
    M += information_matrix_by_block(i);
  }
  return M;
}

inline MatrixXd glmmr::Model::sigma_block(int b,
                                          bool inverse){
  if(b >= sigma_blocks_.size())Rcpp::stop("Index out of range");
  sparse ZLs = submat_sparse(covariance_.ZL_sparse(),sigma_blocks_[b].RowIndexes);
  MatrixXd ZL = sparse_to_dense(ZLs,false);
  MatrixXd S = ZL * ZL.transpose();
  for(int i = 0; i < S.rows(); i++){
    S(i,i)+= 1/W_(sigma_blocks_[b].RowIndexes[i]);
  }
  if(inverse){
    S = S.llt().solve(MatrixXd::Identity(S.rows(),S.cols()));
  }
  return S;
}

inline MatrixXd glmmr::Model::sigma_builder(int b,
                                            bool inverse){
  int B_ = sigma_blocks_.size();
  if (b == B_ - 1) {
    return sigma_block(b,inverse);
  }
  else {
    MatrixXd mat1 = sigma_block(b,inverse);
    MatrixXd mat2;
    if (b == B_ - 2) {
      mat2 = sigma_block(b+1,inverse);
    }
    else {
      mat2 = sigma_builder(b + 1,  inverse);
    }
    int n1 = mat1.rows();
    int n2 = mat2.rows();
    MatrixXd dmat = MatrixXd::Zero(n1+n2, n1+n2);
    dmat.block(0,0,n1,n1) = mat1;
    dmat.block(n1, n1, n2, n2) = mat2;
    return dmat;
  }
}

inline MatrixXd glmmr::Model::information_matrix_by_block(int b){
  ArrayXi rows = Map<ArrayXi,Unaligned>(sigma_blocks_[b].RowIndexes.data(),sigma_blocks_[b].RowIndexes.size());
  MatrixXd X = glmmr::Eigen_ext::submat(linpred_.X(),rows,ArrayXi::LinSpaced(P_,0,P_-1));
  MatrixXd S = sigma_block(b,true);
  MatrixXd M = X.transpose()*S*X;
  return M;
}


inline void glmmr::Model::calculate_var_par(){
  if(family_=="gaussian" || family_=="Gamma" || family_=="beta"){
    // revise this for beta and Gamma re residuals
    int niter = u_.cols();
    ArrayXd sigmas(niter);
    MatrixXd zd = linpred();
#pragma omp parallel for num_threads(n_threads_)
    for(int i = 0; i < niter; ++i){
      VectorXd zdu = glmmr::maths::mod_inv_func(zd.col(i), link_);
      ArrayXd resid = (y_ - zdu);
      sigmas(i) = (resid - resid.mean()).square().sum()/(resid.size()-1);
    }
    update_var_par(sigmas.mean());
  }
}

inline void glmmr::Model::gen_sigma_blocks(){
  int block_counter = 0;
  intvec2d block_ids(n_);
  int block_size;
  sparse Z = covariance_.Z_sparse();
  int i,j,k;
  auto it_begin = Z.Ai.begin();
  for(int b = 0; b < covariance_.B(); b++){
    block_size = covariance_.block_dim(b);
    for(i = 0; i < block_size; i++){
#pragma omp parallel for shared(it_begin, i) num_threads(n_threads_)
      for(j = 0; j < n_; j++){
        auto it = std::find(it_begin + Z.Ap[j], it_begin + Z.Ap[j+1], (i+block_counter));
        if(it != (it_begin + Z.Ap[j+1])){
          block_ids[j].push_back(b);
        }
      }
    }
    block_counter += block_size;
  }
  
  block_counter = 0;
  intvec idx_matches;
  int n_matches;
  for(i = 0; i < n_; i++){
    if(block_counter == 0){
      glmmr::SigmaBlock newblock(block_ids[i]);
      newblock.add_row(0);
      sigma_blocks_.push_back(newblock);
    } else {
      for(j = 0; j < block_counter; j++){
        if(sigma_blocks_[j] == block_ids[i]){
          idx_matches.push_back(j);
        }
      }
      n_matches = idx_matches.size();
      if(n_matches==0){
        glmmr::SigmaBlock newblock(block_ids[i]);
        newblock.add_row(i);
        sigma_blocks_.push_back(newblock);
      } else if(n_matches==1){
        sigma_blocks_[idx_matches[0]].add(block_ids[i]);
        sigma_blocks_[idx_matches[0]].add_row(i);
      } else if(n_matches>1){
        std::reverse(idx_matches.begin(),idx_matches.end());
        for(k = 0; k < (n_matches-1); k++){
          sigma_blocks_[idx_matches[n_matches-1]].merge(sigma_blocks_[idx_matches[k]]);
          sigma_blocks_.erase(sigma_blocks_.begin()+idx_matches[k]);
        }
      }
    }
    idx_matches.clear();
    block_counter = sigma_blocks_.size();
  }
}

#endif
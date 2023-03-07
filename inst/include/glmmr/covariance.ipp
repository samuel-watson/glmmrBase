#ifndef COVARIANCE_IPP
#define COVARIANCE_IPP

inline void glmmr::Covariance::parse(){
  // now process each step of the random effect terms
  if(colnames_.size()!= data_.cols())Rcpp::stop("colnames length != data columns");
  int nre = form_.re_.size();
  for(int i = 0; i < nre; i++){
    strvec fn;
    intvec2d fnvars;
    std::stringstream check1(form_.re_[i]);
    std::string intermediate;
    int iter = 0;
    while(getline(check1, intermediate, '*')){
      intvec fnvars1;
      fnvars.push_back(fnvars1);
      std::stringstream check2(intermediate);
      std::string intermediate2;
      getline(check2, intermediate2, '(');
      fn.push_back(intermediate2);
      getline(check2, intermediate2, ')');

      if(intermediate2.find(",") != std::string::npos){
        std::stringstream check3(intermediate2);
        std::string intermediate3;
        while(getline(check3, intermediate3, ',')){
          auto colidx = std::find(colnames_.begin(),colnames_.end(),intermediate3);
          if(colidx == colnames_.end()){
            Rcpp::stop("variable not in data");
          } else {
            int newidx = colidx - colnames_.begin();
            fnvars[iter].push_back(newidx);
          }
        }
      } else {
        auto colidx = std::find(colnames_.begin(),colnames_.end(),intermediate2);
        if(colidx == colnames_.end()){
          Rcpp::stop("variable not in data");
        } else {
          int newidx = colidx - colnames_.begin();
          fnvars[iter].push_back(newidx);
        }
      }
      iter++;
    }

    // if any of the functions are group, then use block
    // functions
    auto idxgr = std::find(fn.begin(),fn.end(),"gr");
    dblvec2d groups;
    dblvec vals;
    bool isgr;
    int j,k,l,idx,zcol;
    if(form_.z_[i].compare("1")==0){
      zcol = -1;
    } else {
      auto idxz = std::find(colnames_.begin(),colnames_.end(),form_.z_[i]);
      if(idxz == colnames_.end()){
        Rcpp::stop("z variable not in column names");
      } else {
        zcol = idxz - colnames_.begin();
      }
    }


    if(idxgr!=fn.end()){
      idx = idxgr - fn.begin();
      isgr = true;
      vals.resize(fnvars[idx].size());

      for(j = 0; j < data_.rows(); j++){
        for(k = 0; k < fnvars[idx].size(); k++){
          vals[k] = data_(j,fnvars[idx][k]);
        }
        if(std::find(groups.begin(),groups.end(),vals) == groups.end()){
          groups.push_back(vals);
        }
      }
    } else {
      isgr = false;
      vals.push_back(0.0);
      groups.push_back(vals);
    }

    intvec allcols;
    for(j = 0; j< fnvars.size();j++){
      for(k = 0; k < fnvars[j].size();k++){
        allcols.push_back(fnvars[j][k]);
      }
    }

    int total_vars = allcols.size();
    int gridx;
    dblvec allvals;
    allvals.resize(total_vars);
    int currresize = re_data_.size();
    re_data_.resize(currresize+groups.size());

    // // for each group, create a new z data including the group
    for(j = 0; j < groups.size(); j++){
      re_cols_.push_back(fnvars);
      fn_.push_back(fn);
      z_.push_back(zcol);
      re_order_.push_back(form_.re_order_[i]);
      for(k = 0; k < data_.rows(); k++){
        if(isgr){
          for(l = 0; l < vals.size(); l++){
            vals[l] = data_(k,fnvars[idx][l]);
          }
          auto gridx2 = std::find(groups.begin(),groups.end(),vals);
          gridx = gridx2 - groups.begin();
        } else {
          gridx = 0;
        }

        for(l = 0; l<total_vars; l++){
          allvals[l] = data_(k,allcols[l]);
        }
        if(std::find(re_data_[gridx + currresize].begin(),re_data_[gridx + currresize].end(),allvals) ==re_data_[gridx + currresize].end()){
          re_data_[gridx + currresize].push_back(allvals);
        }
      }
    }

  }

  // get parameter indexes
  re_pars_.resize(fn_.size());
  bool firsti;
  npars_ = 0;
  int remidx;
  for(int i = 0; i < nre; i++){
    firsti = true;
    for(int j = 0; j < fn_.size(); j++){
      if(re_order_[j]==i){
        if(firsti){
          intvec2d parcount1;
          for(int k = 0; k < fn_[j].size(); k++){
            auto parget = nvars.find(fn_[j][k]);
            int npars = parget->second;
            intvec parcount2;
            for(int l = 0; l < npars; l++){
              parcount2.push_back(l+npars_);
            }
            parcount1.push_back(parcount2);
            npars_ += npars;
          }
          re_pars_[j] = parcount1;
          firsti = false;
          remidx = j;
        } else {
          re_pars_[j] = re_pars_[remidx];
        }
      }
    }
  }

  //now build the reverse polish notation
  int nvarfn;
  for(int i =0; i<fn_.size();i++){
    intvec fn_instruct;
    intvec fn_par;
    for(int j = 0; j<fn_[i].size();j++){
      intvec A;
      if(fn_[i][j]!="gr"){
        nvarfn = re_cols_[i][j].size();
        for(int k = 0; k< nvarfn; k++){
          A.insert(A.end(),xvar_rpn.begin(),xvar_rpn.end());
        }
        for(int k = 0; k<( nvarfn-1); k++){
          A.push_back(3);
        }
        A.push_back(7);
      }
      intvec B = glmmr::interpret_re(fn_[i][j],A);
      intvec Bpar = glmmr::interpret_re_par(fn_[i][j],re_cols_[i][j],re_pars_[i][j]);
      fn_instruct.insert(fn_instruct.end(),B.begin(),B.end());
      fn_par.insert(fn_par.end(),Bpar.begin(),Bpar.end());
    }
    if(fn_[i].size() > 1){
      for(int j = 0; j < (fn_[i].size()-1); j++){
        fn_instruct.push_back(5);
      }
    }
    re_rpn_.push_back(fn_instruct);
    re_index_.push_back(fn_par);
  }

  //get the number of random effects
  Q_ = 0;
  for(int i = 0; i < re_data_.size(); i++){
    Q_ += re_data_[i].size();
  }

  B_ = re_data_.size();
  n_ = data_.rows();
}

inline double glmmr::Covariance::get_val(int b, int i, int j){
  return glmmr::calculate(re_rpn_[b],re_index_[b],parameters_,re_data_[b],i,j);
}

inline Eigen::MatrixXd glmmr::Covariance::get_block(int b){
  if(b > re_rpn_.size()-1)Rcpp::stop("b larger than number of block");
  if(parameters_.size()==0)Rcpp::stop("no parameters");

  int dim = re_data_[b].size();
  Eigen::MatrixXd D(dim,dim);
  //diagonal
  for(int k = 0; k < dim; k++){
    D(k,k) = get_val(b,k,k);
  }
  if(dim>1){
    for(int i = 0; i < (dim-1); i++){
      for(int j = (i+1); j < dim; j++){
        D(j,i) = get_val(b,j,i);
        D(i,j) = D(j,i);
      }
    }
  }
  return D;
}


inline Eigen::MatrixXd glmmr::Covariance::Z(){
  if(Q_==0)Rcpp::stop("Random effects not initialised");
  Eigen::MatrixXd Z(data_.rows(),Q_);
  Z.setZero();
  int zcount = 0;
  int nvar,nval;
  int i,j,k,l,m;
  re_obs_index_.resize(re_data_.size());
  for(i = 0; i < re_data_.size(); i++){
    for(j = 0; j < re_data_[i].size(); j++){
      dblvec vals(re_data_[i][j].size());
      for(k = 0; k < data_.rows(); k++){
        nval = 0;
        for(l = 0; l < re_cols_[i].size(); l++){
          for(m = 0; m < re_cols_[i][l].size(); m++){
            vals[nval] = data_(k,re_cols_[i][l][m]);
            nval++;
          }
        }
        if(re_data_[i][j]==vals){
          re_obs_index_[i].push_back(k);
          Z(k,zcount) = z_[i]==-1 ? 1.0 : data_(k,z_[i]);
        }
      }
      zcount++;
    }
  }
  return Z;
}

inline Eigen::MatrixXd glmmr::Covariance::get_chol_block(int b,bool upper){
  int n = re_data_[b].size();;
  std::vector<double> L(n * n, 0.0);

  for (int j = 0; j < n; j++) {
    double s = glmmr::algo::inner_sum(&L[j * n], &L[j * n], j);
    L[j * n + j] = sqrt(get_val(b, j, j) - s);
    for (int i = j + 1; i < n; i++) {
      double s = glmmr::algo::inner_sum(&L[j * n], &L[i * n], j);
      L[i * n + j] = (1.0 / L[j * n + j] * (get_val(b, j, i) - s));
    }
  }
  Eigen::MatrixXd M = Eigen::Map<Eigen::MatrixXd>(L.data(), n, n);
  if (upper) {
    return M;
  } else {
    return M.transpose();
  }
}

inline Eigen::VectorXd glmmr::Covariance::sim_re(){
  if(parameters_.size()==0)Rcpp::stop("no parameters");
  Eigen::VectorXd samps(Q_);
  int idx = 0;
  int ndim;
  for(int i=0; i< B(); i++){
    Eigen::MatrixXd L = get_chol_block(i);
    ndim = re_data_[i].size();
    Rcpp::NumericVector z = Rcpp::rnorm(ndim);
    Eigen::Map<Eigen::VectorXd> zz(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(z));
    samps.segment(idx,ndim) = L*zz;
    idx += ndim;
  }
  return samps;
}

inline Eigen::MatrixXd glmmr::Covariance::D_builder(int b,
                                                    bool chol,
                                                    bool upper){
  if (b == B_ - 1) {
    return chol ? get_chol_block(b,upper) : get_block(b);
  }
  else {
    Eigen::MatrixXd mat1 = chol ? get_chol_block(b,upper) : get_block(b);
    Eigen::MatrixXd mat2;
    if (b == B_ - 2) {
      mat2 = chol ? get_chol_block(b+1,upper) : get_block(b+1);
    }
    else {
      mat2 = D_builder(b + 1, chol, upper);
    }
    int n1 = mat1.rows();
    int n2 = mat2.rows();
    Eigen::MatrixXd dmat = Eigen::MatrixXd::Zero(n1+n2, n1+n2);
    dmat.block(0,0,n1,n1) = mat1;
    dmat.block(n1, n1, n2, n2) = mat2;
    return dmat;
  }
}

inline double glmmr::Covariance::log_likelihood(const Eigen::VectorXd &u){
  if(parameters_.size()==0)Rcpp::stop("no parameters");
  double logdet_val=0.0;
  double loglik_val=0.0;
  int obs_counter=0;
  if(!isSparse){
    int blocksize;
    size_B_array.setZero();
    for(int b=0;b<B_;b++){
      blocksize = block_dim(b);
      if(blocksize==1){
        double var = get_val(b,0,0);
        size_B_array[b] = -0.5*log(var*var) -0.5*log(2*M_PI) -
          0.5*u(obs_counter)*u(obs_counter)/(var*var);
      } else {
        dmat_matrix.block(0,0,blocksize,blocksize) = get_chol_block(b);
        logdet_val = 0.0;
        for(int i = 0; i < blocksize; i++){
          logdet_val += 2*log(dmat_matrix(i,i));
        }
        zquad.segment(0,blocksize) = glmmr::algo::forward_sub(dmat_matrix,u.segment(obs_counter,blocksize),blocksize);
        size_B_array[b] = (-0.5*blocksize * log(2*M_PI) - 0.5*logdet_val - 0.5*zquad.transpose()*zquad);
      }
      obs_counter += blocksize;
    }
    loglik_val = size_B_array.sum();
  } else {
    SparseChol chol(&mat);
    int d = chol.ldl_numeric();
    for (auto k : chol.D) logdet_val += log(k);

    dblvec v(u.data(), u.data()+u.size());
    chol.ldl_lsolve(&v[0]);
    chol.ldl_d2solve(&v[0]);
    double quad = glmmr::algo::inner_sum(&v[0],&v[0],Q_);
    loglik_val = (-0.5*Q_ * log(2*M_PI) - 0.5*logdet_val - 0.5*quad);
  }
  return loglik_val;
}

inline double glmmr::Covariance::log_determinant(){
  if(parameters_.size()==0)Rcpp::stop("no parameters");
  int blocksize;
  double logdet_val = 0.0;
  if(!isSparse){
    for(int b=0;b<B_;b++){
      blocksize = block_dim(b);
      dmat_matrix.block(0,0,blocksize,blocksize) = get_chol_block(b);
      for(int i = 0; i < blocksize; i++){
        logdet_val += 2*log(dmat_matrix(i,i));
      }
    }
  } else {
    SparseChol chol(&mat);
    int d = chol.ldl_numeric();
    for (auto k : chol.D) logdet_val += log(k);
  }
  
  return logdet_val;
}

inline void glmmr::Covariance::make_sparse(){
  if(parameters_.size()==0)Rcpp::stop("no parameters");
  isSparse = true;
  int dim;
  double val;
  int col_counter=0;
  mat.Ap.clear();
  mat.Ai.clear();
  mat.Ax.clear();
  // algorithm to generate the sparse matrix representation
  for(int b = 0; b < B(); b++){
    dim = block_dim(b);
    for(int i = 0; i < dim; i++){
      mat.Ap.push_back(mat.Ai.size());
      for(int j = 0; j < (i+1); j++){
        val = get_val(b,i,j);
        if(val!=0){
          mat.Ax.push_back(val);
          mat.Ai.push_back((col_counter+j));
        }
      }
    }
    col_counter += dim;
  }
  mat.n = mat.Ap.size();
  mat.Ap.push_back(mat.Ax.size());
};

inline void glmmr::Covariance::make_dense(){
  isSparse = false;
}

inline void glmmr::Covariance::update_ax(){
  int llim = 0;
  int nj = 0;
  int ulim = mat.Ap[nj+block_dim(0)];
  int j = 0;

  for(int b=0; b < B(); b++){
    for(int i = llim; i<ulim; i++){
      if(i == mat.Ap[j+1])j++;
      mat.Ax[i] = get_val(b,mat.Ai[i]-nj,j-nj);
    }
    llim = ulim;
    if(b<(B()-1)){
      nj += block_dim(b);
      ulim = mat.Ap[nj+block_dim(b+1)];
    }
    if(b == (B()-1)){
      ulim = mat.Ai.size();
    }
  }
};

inline Eigen::MatrixXd glmmr::Covariance::D_sparse_builder(bool chol,
                                 bool upper){
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(Q_,Q_);
  if(!chol){
    for(int i = 0; i < Q_; i++){
      for(int j = mat.Ap[i]; j < mat.Ap[i+1]; j++){
        D(mat.Ai[j],i) = mat.Ax[j];
        if(mat.Ai[j]!=i) D(i,mat.Ai[j]) = D(mat.Ai[j],i);
      }
    }
  } else {
    SparseChol chol(&mat);
    int d = chol.ldl_numeric();
    for(int i = 0; i < Q_; i++){
      for(int j = chol.L->Ap[i]; j < chol.L->Ap[i+1]; j++){
        D(chol.L->Ai[j],i) = chol.L->Ax[j];
      }
    }
    Eigen::VectorXd diag = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(chol.D.data(),chol.D.size());
    diag = diag.array().sqrt().matrix();
    D.diagonal() = Eigen::ArrayXd::Ones(Q_);
    D *= diag.asDiagonal();
    if(upper)D.transposeInPlace();
  }
  return D;
}

inline bool glmmr::Covariance::any_group_re(){
  bool gr = false;
  for(int i = 0; i < fn_.size(); i++){
    for(int j = 0; j < fn_[i].size(); j++){
      if(fn_[i][j]=="gr"){
        gr = true;
        break;
      }
    }
    if(gr)break;
  }
  return gr;
}

#endif
#ifndef COVARIANCE_IPP
#define COVARIANCE_IPP

inline void glmmr::Covariance::parse(){
  intvec3d re_cols_;
  intvec re_order_;
  strvec2d re_par_names_;
  dblvec3d re_temp_data_;

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
    intvec2d newrecols;
    allvals.resize(total_vars);
    newrecols.resize(fnvars.size());
    int currresize = calc_.size();
    
    calc_.resize(currresize+groups.size());
    re_temp_data_.resize(currresize+groups.size());
    re_cols_.resize(currresize+groups.size());
    re_cols_data_.resize(currresize+groups.size());
    // for each group, create a new z data including the group
    
    int fn_var_counter = 0;
    for(int m = 0; m < fnvars.size(); m++){
      intvec iter_fn_var_index;
      for(int p = 0; p < fnvars[m].size(); p++){
        iter_fn_var_index.push_back(p + fn_var_counter);
      }
      fn_var_counter += fnvars[m].size();
      newrecols[m] = iter_fn_var_index;
    }
    
    for(j = 0; j < groups.size(); j++){
      fn_.push_back(fn);
      z_.push_back(zcol);
      re_order_.push_back(form_.re_order_[i]);
      re_cols_[currresize + j] = newrecols;
    }
    
    for(k = 0; k < data_.rows(); k++){
        if(isgr){
          for(int m = 0; m < vals.size(); m++){
            vals[m] = data_(k,fnvars[idx][m]);
          }
          auto gridx2 = std::find(groups.begin(),groups.end(),vals);
          gridx = gridx2 - groups.begin();
        } else {
          gridx = 0;
        }
        for(int m = 0; m<total_vars; m++){
          allvals[m] = data_(k,allcols[m]);
          //newrecols[m] = m;
        }
        
        if(std::find(re_temp_data_[gridx + currresize].begin(),
              re_temp_data_[gridx + currresize].end(),allvals) == re_temp_data_[gridx + currresize].end()){
              re_temp_data_[gridx + currresize].push_back(allvals);
              //calc_[gridx + currresize].data.push_back(allvals);
                //re_cols_[gridx + currresize].push_back(newrecols);
              re_cols_data_[gridx + currresize].push_back(allcols);
        }
      }

  }
  
  
  // get parameter indexes
  re_pars_.resize(fn_.size());
  re_par_names_.resize(fn_.size());
  bool firsti;
  npars_ = 0;
  int remidx;
  for(int i = 0; i < nre; i++){
    firsti = true;
    for(int j = 0; j < fn_.size(); j++){
      if(re_order_[j]==i){
        if(firsti){
          intvec2d parcount1;
          strvec parnames2;
          str fn_name = "";
          for(int k = 0; k < fn_[j].size(); k++) fn_name += fn_[j][k];
          for(int k = 0; k < fn_[j].size(); k++){
           if(glmmr::validate_fn(fn_[j][k]))Rcpp::stop("Function " + fn_[j][k] + " not valid");
            auto parget = nvars.find(fn_[j][k]);
            int npars = parget->second;
            intvec parcount2;
            for(int l = 0; l < npars; l++){
              parcount2.push_back(l+npars_);
              parnames2.push_back(fn_name+"."+std::to_string(i)+".("+fn_[j][k]+")."+std::to_string(l));
              re_fn_par_link_.push_back(i);
            }
            parcount1.push_back(parcount2);
            npars_ += npars;
          }
          re_pars_[j] = parcount1;
          re_par_names_[j] = parnames2;
          firsti = false;
          remidx = j;
        } else {
          re_pars_[j] = re_pars_[remidx];
          re_par_names_[j] = re_par_names_[remidx];
        }
      }
    }
  }
  
  //now build the reverse polish notation
  int nvarfn;
  for(int i =0; i<fn_.size();i++){
    intvec fn_instruct;
    intvec fn_par;
    int minvalue = 100;
    for(int j = 0; j<fn_[i].size();j++){
      auto min_value_iterator = std::min_element(re_pars_[i][j].begin(),re_pars_[i][j].end());
      if(*min_value_iterator < minvalue) minvalue = *min_value_iterator;
    }
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
      intvec re_par_less_min_ = re_pars_[i][j];
      for(int k = 0; k < re_pars_[i][j].size(); k++)re_par_less_min_[k] -= minvalue;
      
      
      
      intvec Bpar = glmmr::interpret_re_par(fn_[i][j],re_cols_[i][j],re_par_less_min_);
      fn_instruct.insert(fn_instruct.end(),B.begin(),B.end());
      fn_par.insert(fn_par.end(),Bpar.begin(),Bpar.end());
    }
    if(fn_[i].size() > 1){
      for(int j = 0; j < (fn_[i].size()-1); j++){
        fn_instruct.push_back(5);
      }
    }
    calc_[i].instructions = fn_instruct;
    calc_[i].indexes = fn_par;
    calc_[i].parameter_names = re_par_names_[i];
    calc_[i].parameter_count = re_par_names_[i].size();
  }
  
  //get the number of random effects
  Q_ = 0;
  for(int i = 0; i < calc_.size(); i++){
    Q_ += re_temp_data_[i].size();
  }
  re_count_.resize(form_.re_terms().size());
  std::fill(re_count_.begin(), re_count_.end(), 0);
  for(int i = 0; i < calc_.size(); i++){
    re_count_[re_order_[i]] += re_temp_data_[i].size();
    MatrixXd newredata(re_temp_data_[i].size(),re_temp_data_[i][0].size());
    for(int j = 0; j < re_temp_data_[i].size(); j++){
      for(int k = 0; k < re_temp_data_[i][0].size(); k++){
        newredata(j,k) = re_temp_data_[i][j][k];
      }
    }
    re_data_.push_back(newredata);
  }

  B_ = calc_.size();
  n_ = data_.rows();
  
}

inline void glmmr::Covariance::update_parameters_in_calculators(){
  if(par_for_calcs_.size()==0)par_for_calcs_.resize(B_);
  for(int i = 0; i < B_; i++){
      dblvec par_for_calc;
      for(int j = 0; j < re_pars_[i].size(); j++){
        for(int k = 0; k < re_pars_[i][j].size(); k++){
          par_for_calc.push_back(parameters_[re_pars_[i][j][k]]);
        }
      }
    par_for_calcs_[i] = par_for_calc;
  }
}


inline void glmmr::Covariance::update_parameters(const dblvec& parameters){
    if(parameters_.size()==0){
      parameters_ = parameters;
      update_parameters_in_calculators();
      make_sparse();
      spchol.update(mat);
      L_constructor();
    } else {
      parameters_ = parameters;
      update_parameters_in_calculators();
      update_ax();
    }
  };
  
inline void glmmr::Covariance::update_parameters_extern(const dblvec& parameters){
    if(parameters_.size()==0){
      parameters_ = parameters;
      update_parameters_in_calculators();
      make_sparse();
      spchol.update(mat);
      L_constructor();
    } else {
      parameters_ = parameters;
      update_parameters_in_calculators();
      update_ax();
    }
  };

inline void glmmr::Covariance::update_parameters(const ArrayXd& parameters){
    if(parameters_.size()==0){
      for(int i = 0; i < parameters.size(); i++){
        parameters_.push_back(parameters(i));
      }
      update_parameters_in_calculators();
    } else if(parameters_.size() == parameters.size()){
      for(int i = 0; i < parameters.size(); i++){
        parameters_[i] = parameters(i);
      }
      update_parameters_in_calculators();
      update_ax();
    } else {
      Rcpp::stop("Wrong number of parameters");
    }
  };

inline double glmmr::Covariance::get_val(int b, int i, int j){
  return calc_[b].calculate(i,par_for_calcs_[b],re_data_[b],j)[0];
}

inline MatrixXd glmmr::Covariance::get_block(int b){
  if(b > calc_.size()-1)Rcpp::stop("b larger than number of blocks");
  if(parameters_.size()==0)Rcpp::stop("no parameters");
  if(b > B_-1)Rcpp::stop("b is too large");

  int dim = re_data_[b].rows();//calc_[b].data.size();
  MatrixXd D(dim,dim);
  D.setZero();
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

inline void glmmr::Covariance::Z_constructor(){

  intvec2d re_obs_index_;
  if(Q_==0)Rcpp::stop("Random effects not initialised");
  MatrixXd Z(data_.rows(),Q_);
  Z.setZero();
  int zcount = 0;
  int nvar,nval;
  int i,j,k,l,m;
  double val;
  re_obs_index_.resize(B_);
  for(i = 0; i < B_; i++){
     for(j = 0; j < re_data_[i].rows(); j++){
       nvar = re_data_[i].cols();
       dblvec vals(nvar);
       dblvec valscomp(nvar);
       for(k = 0; k < data_.rows(); k++){
          for(m = 0; m < nvar; m++){
            vals[m] = data_(k,re_cols_data_[i][j][m]);
            valscomp[m] = re_data_[i](j,m);
          }
          if(valscomp==vals){
            re_obs_index_[i].push_back(k);
            Z(k,zcount) = z_[i]==-1 ? 1.0 : data_(k,z_[i]);
          }
        }
      zcount++;
     }
  }
  matZ = glmmr::dense_to_sparse(Z,false);
}

inline MatrixXd glmmr::Covariance::Z(){
  MatrixXd Z = sparse_to_dense(matZ,false);
  return Z;
}

inline MatrixXd glmmr::Covariance::get_chol_block(int b,bool upper){
  int n = re_data_[b].rows();
  std::vector<double> L(n * n, 0.0);

  for (int j = 0; j < n; j++) {
    double s = glmmr::algo::inner_sum(&L[j * n], &L[j * n], j);
    L[j * n + j] = sqrt(get_val(b, j, j) - s);
    for (int i = j + 1; i < n; i++) {
      double s = glmmr::algo::inner_sum(&L[j * n], &L[i * n], j);
      L[i * n + j] = (1.0 / L[j * n + j] * (get_val(b, j, i) - s));
    }
  }
  MatrixXd M = Map<MatrixXd>(L.data(), n, n);
  if (upper) {
    return M;
  } else {
    return M.transpose();
  }
}

inline VectorXd glmmr::Covariance::sim_re(){
  if(parameters_.size()==0)Rcpp::stop("no parameters");
  VectorXd samps(Q_);
  int idx = 0;
  int ndim;
  if(!isSparse){
    for(int i=0; i< B(); i++){
      MatrixXd L = get_chol_block(i);
      ndim = re_data_[i].rows();
      Rcpp::NumericVector z = Rcpp::rnorm(ndim);
      Map<VectorXd> zz(Rcpp::as<Map<VectorXd> >(z));
      samps.segment(idx,ndim) = L*zz;
      idx += ndim;
    }
  } else {
      Rcpp::NumericVector z = Rcpp::rnorm(Q_);
      VectorXd zz = Rcpp::as<VectorXd>(z);
      samps = matL * zz;
  }
  
  return samps;
}

inline MatrixXd glmmr::Covariance::D_builder(int b,
                                                    bool chol,
                                                    bool upper){
  if (b == B_ - 1) {
    return chol ? get_chol_block(b,upper) : get_block(b);
  }
  else {
    MatrixXd mat1 = chol ? get_chol_block(b,upper) : get_block(b);
    MatrixXd mat2;
    if (b == B_ - 2) {
      mat2 = chol ? get_chol_block(b+1,upper) : get_block(b+1);
    }
    else {
      mat2 = D_builder(b + 1, chol, upper);
    }
    int n1 = mat1.rows();
    int n2 = mat2.rows();
    MatrixXd dmat = MatrixXd::Zero(n1+n2, n1+n2);
    dmat.block(0,0,n1,n1) = mat1;
    dmat.block(n1, n1, n2, n2) = mat2;
    return dmat;
  }
}

inline sparse glmmr::Covariance::ZL_sparse(){
  sparse ZLs = matZ * matL;
  return ZLs;
}

inline sparse glmmr::Covariance::Z_sparse(){
  return matZ;
}

inline MatrixXd glmmr::Covariance::ZL(){
  sparse ZD = ZL_sparse();
  MatrixXd ZL = glmmr::sparse_to_dense(ZD,false);
  return ZL;
}

inline MatrixXd glmmr::Covariance::LZWZL(const VectorXd& w){
  sparse ZL = ZL_sparse();
  sparse ZLt = ZL;
  ZLt.transpose();
  ZLt = ZLt % w;
  ZLt *= ZL;
  // add 1 to diagonal
  for(int i = 0; i < ZLt.n; i++){
    for(int j = ZLt.Ap[i]; j<ZLt.Ap[i+1]; j++){
      if(i == ZLt.Ai[j])ZLt.Ax[j]++;
    }
  }
  return glmmr::sparse_to_dense(ZLt);
}

inline MatrixXd glmmr::Covariance::ZLu(const MatrixXd& u){
  sparse ZL = ZL_sparse();
  //ZL = ;
  //MatrixXd ZLu = glmmr::sparse_to_dense(ZLu);
  return ZL * u;
}

inline MatrixXd glmmr::Covariance::Lu(const MatrixXd& u){
  return matL * u;
}

inline double glmmr::Covariance::log_likelihood(const VectorXd &u){
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
        size_B_array[b] = -0.5*log(var) -0.5*log(2*M_PI) -
          0.5*u(obs_counter)*u(obs_counter)/(var);
      } else {
        zquad.setZero();
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
    //SparseChol chol(mat);
    //int d = chol.ldl_numeric();
    for (auto k : spchol.D) logdet_val += log(k);

    dblvec v(u.data(), u.data()+u.size());
    spchol.ldl_lsolve(&v[0]);
    spchol.ldl_d2solve(&v[0]);
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
    //SparseChol chol(&mat);
    //int d = chol.ldl_numeric();
    for (auto k : spchol.D) logdet_val += log(k);
  }
  
  return logdet_val;
}


inline void glmmr::Covariance::make_sparse(){
  
  if(parameters_.size()==0)Rcpp::stop("no parameters");
  //isSparse = true;
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
  mat.m = mat.Ap.size();
  mat.Ap.push_back(mat.Ax.size());
};

inline void glmmr::Covariance::L_constructor(){
  int d = spchol.ldl_numeric();
  matL = spchol.LD();
}

inline void glmmr::Covariance::set_sparse(bool sparse){
  isSparse = sparse;
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
  spchol.A_ = mat;
  int d = spchol.ldl_numeric(); // assumes structure of D doesn't change
  matL = spchol.LD();
};

inline MatrixXd glmmr::Covariance::D_sparse_builder(bool chol,
                                 bool upper){
  MatrixXd D = MatrixXd::Zero(Q_,Q_);
  if(!chol){
    D = glmmr::sparse_to_dense(mat,true);
  } else {
    D = glmmr::sparse_to_dense(matL,false);
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

inline strvec glmmr::Covariance::parameter_names(){
    strvec parnames;
    for(int i = 0; i < B_; i++){
      parnames.insert(parnames.end(),calc_[i].parameter_names.begin(),calc_[i].parameter_names.end());
    }
    auto last = std::unique(parnames.begin(),parnames.end());
    parnames.erase(last, parnames.end());
    return parnames;
  };



#endif
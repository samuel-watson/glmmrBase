#ifndef MODELOPTIMWEIGHTS_IPP
#define MODELOPTIMWEIGHTS_IPP

inline ArrayXd glmmr::Model::optimum_weights(double N, 
                                             double sigma_sq,
                                             VectorXd C,
                                             double tol,
                                             int max_iter){
  if(C.size()!=P_)Rcpp::stop("C is wrong size");
  VectorXd Cvec(C);
  ArrayXd weights = ArrayXd::Constant(n_,1.0*n_);
  VectorXd holder(n_);
  weights = weights.inverse();
  ArrayXd weightsnew(weights);
  std::vector<MatrixXd> ZDZ;
  std::vector<MatrixXd> Sigmas;
  std::vector<MatrixXd> Xs;
  std::vector<glmmr::SigmaBlock> SB(sigma_blocks_);
  Rcpp::Rcout << "\n### Preparing data ###";
  Rcpp::Rcout << "\nThere are " << SB.size() << " independent blocks and " << n_ << " cells.";
  int maxprint = n_ < 10 ? n_ : 10;
  for(int i = 0 ; i < SB.size(); i++){
    sparse ZLs = submat_sparse(covariance_.ZL_sparse(),SB[i].RowIndexes);
    MatrixXd ZL = sparse_to_dense(ZLs,false);
    MatrixXd S = ZL * ZL.transpose();
    ZDZ.push_back(S);
    Sigmas.push_back(S);
    ArrayXi rows = Map<ArrayXi,Unaligned>(SB[i].RowIndexes.data(),SB[i].RowIndexes.size());
    MatrixXd X = glmmr::Eigen_ext::submat(linpred_.X(),rows,ArrayXi::LinSpaced(P_,0,P_-1));
    Xs.push_back(X);
  }
  
  double diff = 1;
  int block_size;
  MatrixXd M(P_,P_);
  int iter = 0;
  int counter;
  Rcpp::Rcout << "\n### Starting optimisation ###";
  while(diff > tol && iter < max_iter){
    iter++;
    Rcpp::Rcout << "\nIteration " << iter << "\n------------\nweights: [" << weights.segment(0,maxprint).transpose() << " ...]";
    
    //add check to remove weights that are below a certain threshold
    if((weights < 1e-8).any()){
      for(int i = 0 ; i < SB.size(); i++){
        auto it = SB[i].RowIndexes.begin();
        while(it != SB[i].RowIndexes.end()){
          if(weights(*it) < 1e-8){
            weights(*it) = 0;
            int idx = it - SB[i].RowIndexes.begin();
            glmmr::Eigen_ext::removeRow(Xs[i],idx);
            glmmr::Eigen_ext::removeRow(ZDZ[i],idx);
            glmmr::Eigen_ext::removeColumn(ZDZ[i],idx);
            Sigmas[i].conservativeResize(ZDZ[i].rows(),ZDZ[i].cols());
            it = SB[i].RowIndexes.erase(it);
            Rcpp::Rcout << "\n Removing point " << idx << " in block " << i;
          } else {
            it++;
          }
        }
      }
    }
    
    M.setZero();
    for(int i = 0 ; i < SB.size(); i++){
      Sigmas[i] = ZDZ[i];
      for(int j = 0; j < Sigmas[i].rows(); j++){
        Sigmas[i](j,j) += sigma_sq/(N*weights(SB[i].RowIndexes[j]));
      }
      Sigmas[i] = Sigmas[i].llt().solve(MatrixXd::Identity(Sigmas[i].rows(),Sigmas[i].cols()));
      M += Xs[i].transpose() * Sigmas[i] * Xs[i];
    }
    
    //check if positive definite, if not remove the offending column(s)
    bool isspd = glmmr::Eigen_ext::issympd(M);
    if(isspd){
      Rcpp::Rcout << "\n Information matrix not postive definite: ";
      ArrayXd M_row_sums = M.rowwise().sum();
      int fake_it = 0;
      int countZero = 0;
      for(int j = 0; j < M_row_sums.size(); j++){
        if(M_row_sums(j) == 0){
          Rcpp::Rcout << "\n   Removing column " << fake_it;
          for(int k = 0; k < Xs.size(); k++){
            glmmr::Eigen_ext::removeColumn(Xs[k],fake_it);
          }
          glmmr::Eigen_ext::removeElement(Cvec,fake_it);
          countZero++;
        } else {
          fake_it++;
        }
      }
      M.conservativeResize(M.rows()-countZero,M.cols()-countZero);
      M.setZero();
      for(int k = 0; k < SB.size(); k++){
        M += Xs[k].transpose() * Sigmas[k] * Xs[k];
      }
    }
    M = M.llt().solve(MatrixXd::Identity(M.rows(),M.cols()));
    VectorXd Mc = M*Cvec;
    counter = 0;
    weightsnew.setZero();
    for(int i = 0 ; i < SB.size(); i++){
      block_size = SB[i].RowIndexes.size();
      holder.segment(0,block_size) = Sigmas[i] * Xs[i] * Mc;
      for(int j = 0; j < block_size; j++){
        weightsnew(SB[i].RowIndexes[j]) = holder(j);
      }
    }
    weightsnew = weightsnew.abs();
    weightsnew *= 1/weightsnew.sum();
    diff = ((weights-weightsnew).abs()).maxCoeff();
    weights = weightsnew;
    Rcpp::Rcout << "\n(Max. diff: " << diff << ")\n";
  }
  if(iter<max_iter){
    Rcpp::Rcout << "\n### CONVERGED Final weights: [" << weights.segment(0,maxprint).transpose() << "...]";
  } else {
    Rcpp::Rcout << "\n### NOT CONVERGED Reached maximum iterations";
  }
  return weights;
}

#endif
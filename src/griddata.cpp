#include <glmmr/griddata.hpp>

ArrayXi glmmr::griddata::top_i_pq(const ArrayXd& v, int n) {
  typedef std::pair<double, int> Elt;
  struct ComparePair {
    bool operator()(const Elt& elt1, const Elt& elt2) const {
      return elt1.first < elt2.first;
    }
  };
  std::priority_queue< Elt, std::vector<Elt>, ComparePair > pq;
  std::vector<int> result;
  for (int i = 0; i < v.size(); i++) {
    if (pq.size() < n)
      pq.push(Elt(v(i), i));
    else {
      Elt elt = Elt(v(i), i);
      if (pq.top().first > elt.first) {
        pq.pop();
        pq.push(elt);
      }
    }
  }
  
  ArrayXi res(pq.size());
  int iter = 0;
  while (!pq.empty()) {
    res(iter) = pq.top().second;
    pq.pop();
    iter++;
  }
  
  return res;
}

void glmmr::griddata::genNN(int M){
  int n = X.rows();
  m = M;
  NN.conservativeResize(M,n);
  NN = ArrayXXi::Constant(M,n,n);
  for(int i=1; i<n; i++){
    ArrayXd dist = ArrayXd::Zero(i);
    if(i > M){
      for(int j=0; j<i; j++){
        for(int k = 0; k < X.cols(); k++){
          dist(j) += (X(i,k) - X(j,k))*(X(i,k) - X(j,k));
        }
      }
      dist = dist.sqrt();
      NN.col(i) = top_i_pq(dist,M);
    } else {
      for(int j = 0; j<i; j++){
        NN(j,i) = j;
      }
    }
  }
}

void glmmr::griddata::setup(const ArrayXXd& X_){
  X.conservativeResize(X_.rows(),X_.cols());
  X = X_;
  N = X_.rows();
}

void glmmr::griddata::setup(const ArrayXXd& X_, int M){
  X.conservativeResize(X_.rows(),X_.cols());
  X = X_;
  N = X_.rows();
  genNN(M);
}
#ifndef ALGO_H
#define ALGO_H

#include <cmath>
#include <RcppEigen.h>
#include "general.h"

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {
namespace algo {
inline double inner_sum(double* li, double* lj, int n)
{
  double s = 0;
  for (int i = 0; i < n; i++) {
    s += li[i] * lj[i];
  }
  return s;
}

inline int get_flink(const std::string &family,
                     const std::string &link){
  const static std::unordered_map<std::string, int> string_to_case{
    {"poissonlog",1},
    {"poissonidentity",2},
    {"binomiallogit",3},
    {"binomiallog",4},
    {"binomialidentity",5},
    {"binomialprobit",6},
    {"gaussianidentity",7},
    {"gaussianlog",8},
    {"Gammalog",9},
    {"Gammainverse",10},
    {"Gammaidentity",11},
    {"betalogit",12}
  };

  return string_to_case.at(family + link);
}

inline Eigen::VectorXd forward_sub(const Eigen::MatrixXd& U,
                                   const Eigen::VectorXd& u,
                                   const int& n)
{
  Eigen::VectorXd y(n);
  for (int i = 0; i < n; i++) {
    double lsum = 0;
    for (int j = 0; j < i; j++) {
      lsum += U(i,j) * y(j);
    }
    y(i) = (u(i) - lsum) / U(i,i);
  }
  return y;
}

}

namespace Eigen_ext {

template<class ArgType, class RowIndexType, class ColIndexType>
class index_functor {
  const ArgType &m_arg;
  const RowIndexType &m_rowIndices;
  const ColIndexType &m_colIndices;
public:
  typedef Eigen::Matrix<typename ArgType::Scalar,
                        RowIndexType::SizeAtCompileTime,
                        ColIndexType::SizeAtCompileTime,
                        ArgType::Flags&Eigen::RowMajorBit?Eigen::RowMajor:Eigen::ColMajor,
                        RowIndexType::MaxSizeAtCompileTime,
                        ColIndexType::MaxSizeAtCompileTime> MatrixType;
  
  index_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
    : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
  {}
  
  const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
    return m_arg(m_rowIndices[row], m_colIndices[col]);
  }
};

template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<index_functor<ArgType,RowIndexType,ColIndexType>, typename index_functor<ArgType,RowIndexType,ColIndexType>::MatrixType>
submat(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
{
  typedef index_functor<ArgType,RowIndexType,ColIndexType> Func;
  typedef typename Func::MatrixType MatrixType;
  return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}

}

class SigmaBlock {
  public:
    intvec Dblocks;
    intvec RowIndexes;
    SigmaBlock(){};
    SigmaBlock(const intvec& db) : Dblocks(db) {};
    
    bool operator==(const intvec& x){
      bool element_is_in = false;
      for(auto i : x){
        auto it = std::find(Dblocks.begin(),Dblocks.end(),i);
        if(it != Dblocks.end()){
          element_is_in = true;
          break;
        } 
      }
      return element_is_in;
    }
    
    void add(const intvec& x){
      intvec xout;
      bool element_is_in = false;
      for(auto i : x){
        auto it = std::find(Dblocks.begin(),Dblocks.end(),i);
        if(it != Dblocks.end()){
          element_is_in = true;
        } else {
          xout.push_back(i); 
        }
      }
      if(element_is_in){
        Dblocks.insert(Dblocks.end(),xout.begin(),xout.end());
        std::sort(Dblocks.begin(),Dblocks.end());
      }
    }
    
    void merge(const SigmaBlock& x){
      RowIndexes.insert(RowIndexes.end(),x.RowIndexes.begin(),x.RowIndexes.end());
      std::sort(RowIndexes.begin(), RowIndexes.end() );
      RowIndexes.erase( std::unique( RowIndexes.begin(), RowIndexes.end() ), RowIndexes.end() );
      Dblocks.insert(Dblocks.end(),x.Dblocks.begin(),x.Dblocks.end());
      std::sort(Dblocks.begin(), Dblocks.end() );
      Dblocks.erase( std::unique( Dblocks.begin(), Dblocks.end() ), Dblocks.end() );
    }
    
    void add_row(int i){
      RowIndexes.push_back(i);
    }
    
    intvec rows(){
      return RowIndexes;
    }
};

}







#endif
#pragma once

// #define R_BUILD

#include <vector>
#include <functional>
#include <exception>
#include <map>
#include <cmath>
#include <random>
#include <queue>
#include <type_traits>
#include <memory>
#include <tuple>
#include "bobyqa_algo.h"
#include "newuoa.h"
#ifdef R_BUILD
#include <Rcpp.h> // for printing to R
#endif

// this will be used as the basis of adding more funcitons at a later date

class optim_algo {};
class BOBYQA : public optim_algo {};
class NEWUOA : public optim_algo {};



template<typename Signature, class algo>
class optim;

inline size_t random_index(size_t max_index) {
  std::random_device                      rand_dev;
  std::mt19937                            generator(rand_dev());
  std::uniform_int_distribution<size_t>   distr(0, max_index);
  return distr(generator);
}

// class for bobyqa - I am working on a new version of BOBYQA 
template <typename T>
class optim<T(const std::vector<T>&),BOBYQA> {
  using func = T(*)(long, const T*, void*); 
public:
  struct optimControl {
    int npt = 0;
    double rhobeg = 0.0;
    double rhoend = 0.0;
    int trace = 0;
    int maxfun = 0;
  } control;
  
  optim(){};
  optim(const std::vector<T>& start);
  optim(const optim& x) = default;
  auto operator=(const optim& x) -> optim& = default;
  
  // functions
  // the two functions fn() enable capturing of functions to use in the algorithm
  template<auto Function, typename = std::enable_if_t<std::is_invocable_r_v<T, decltype(Function), const std::vector<T>& > > >
  void    fn();
  template<auto Function, typename Class, typename = std::enable_if_t<std::is_invocable_r_v<T, decltype(Function), Class*, const std::vector<T>& > > >
  void    fn(Class* cls);
  
  void    set_bounds(const std::vector<T>& lower, const std::vector<T>& upper);
  void    set_bounds(const std::vector<T>& bound, bool lower = true);
  auto    operator()(const std::vector<T>& vec) const -> T;
  void    minimise(); // the optim algorithm
  std::vector<T> values() const;
  
private:
  [[noreturn]]
  static auto null_fn(long n, const T* x, void* p) -> T {throw std::exception{};}
  
  void*                   optim_instance = nullptr;  // pointer to the class if a member function
  func                    optim_fn = &null_fn;        // pointer to the function
  size_t                  dim;                       // number of dimensions
  std::vector<T>          lower_bound;               // bounds
  std::vector<T>          upper_bound;   
  T                       min_f;                     // current best value
  int                     fn_counter = 0;
  int                     iter = 0;
  std::vector<T>          current_values;
  std::string             msg_;
  
  //functions
  auto            eval(const std::vector<T>& vec) -> T;
  void            update_msg(int res);
};


template <typename T>
inline optim<T(const std::vector<T>&),BOBYQA>::optim(const std::vector<T>& start) {
  dim = start.size();
  current_values.resize(dim);
  current_values = start;
}

template <typename T>
inline std::vector<T> optim<T(const std::vector<T>&),BOBYQA>::values() const {
  return current_values;
}

template <typename T>
inline void optim<T(const std::vector<T>&),BOBYQA>::set_bounds(const std::vector<T>& lower, const std::vector<T>& upper) {
  lower_bound.resize(dim);
  upper_bound.resize(dim);
  lower_bound = lower; 
  upper_bound = upper;
};

template <typename T>
inline void optim<T(const std::vector<T>&),BOBYQA>::set_bounds(const std::vector<T>& bound, bool lower) {
  if(lower){
    lower_bound.resize(dim);
    lower_bound = bound;
  } else {
    upper_bound.resize(dim);
    upper_bound = bound;
  }
};

template <typename T>
template <auto Function, typename>
inline void optim<T(const std::vector<T>&),BOBYQA>::fn() 
{
  optim_instance = nullptr;
  optim_fn = static_cast<func>([](long n, const T* x, void*) -> T {
    return std::invoke(Function, std::vector<T>(x,x+n));
  });
};

template <typename T>
template<auto Function, typename Class, typename>
inline void optim<T(const std::vector<T>&),BOBYQA>::fn(Class* cls)
{
  optim_instance = cls;
  optim_fn = static_cast<func>([](long n, const T* x, void* p) -> T {
    auto* c = static_cast<Class*>(p);
    return std::invoke(Function,c,std::vector<T>(x,x+n));
  });
}

template <typename T>
inline auto optim<T(const std::vector<T>&),BOBYQA>::operator()(const std::vector<T>& vec) const -> T
{
  return std::invoke(optim_fn,vec.size(),vec.data(),optim_instance);
}    

template <typename T>
inline void optim<T(const std::vector<T>&),BOBYQA>::minimise(){
    fn_counter = 0;
#ifndef R_BUILD
    double R_NegInf = -1.0 * std::numeric_limits<double>::infinity();
    double R_PosInf = std::numeric_limits<double>::infinity();
#endif
    if (!control.npt) control.npt = std::min(dim + 2, (dim+2)*(dim+1)/2);     
    if(lower_bound.empty()){
      lower_bound.resize(dim);
      for(int i = 0; i< dim; i++) lower_bound[i] = R_NegInf;
    }    
    if(upper_bound.empty()){
      upper_bound.resize(dim);
      for(int i = 0; i< dim; i++)upper_bound[i] = R_PosInf;
    }
    double max_par = *std::max_element(current_values.begin(),current_values.end(),[](const double& a, const double& b)
    {
      return abs(a) < abs(b);
    });
    if (!control.rhobeg) control.rhobeg = std::min(0.95, std::max(0.2*max_par, 0.2));
    if (!control.rhoend) control.rhoend = 1.0e-6 * control.rhobeg;    
    if (!control.maxfun) control.maxfun = 10000;
    std::vector<double> w;
    w.resize((control.npt + 5) * (control.npt + dim) + (3 * dim * (dim + 5))/2);    
    int res = bobyqa(dim, control.npt, optim_fn, optim_instance, current_values.data(), 
                     lower_bound.data(), upper_bound.data(),
                     control.rhobeg, control.rhoend, control.trace, 
                     control.maxfun, w.data());
    update_msg(res);
    min_f = eval(current_values);
#ifdef R_BUILD
    if(control.trace >= 1)
    {
      Rcpp::Rcout << "\nEND BOBYQA | fn: " << fn_counter << " | " << msg_;  
    }
#endif
}

template <typename T>
inline auto optim<T(const std::vector<T>&),BOBYQA>::eval(const std::vector<T>& vec) -> T
{   
  fn_counter++;
  return std::invoke(optim_fn,vec.size(),vec.data(),optim_instance);
}  

template <typename T>
inline void optim<T(const std::vector<T>&),BOBYQA>::update_msg(int res) {
    switch(res) {
    case 0:
      msg_ = "Normal exit from optim";
      break;
    case -1:
      msg_ = "optim -- NPT is not in the required interval";
      break;
    case -2:
      msg_ = "optim -- one of the box constraint ranges is too small (< 2*RHOBEG)";
      break;
    case -3:
      msg_ = "optim detected too much cancellation in denominator";
      break;
    case -4:
      msg_ = "optim -- maximum number of function evaluations exceeded";
      break;
    case -5:
      msg_ = "optim -- a trust region step failed to reduce q";
      break;
    default: ;
    }
  }


  // class for newuoa
template <typename T>
class optim<T(const std::vector<T>&),NEWUOA> {
  using func = T(*)(void*, long, const T*); 
public:
  struct optimControl {
    int npt = 0;
    double rhobeg = 0.0;
    double rhoend = 0.0;
    int trace = 0;
    int maxfun = 0;
  } control;
  
  optim(){};
  optim(const std::vector<T>& start);
  optim(const optim& x) = default;
  auto operator=(const optim& x) -> optim& = default;
  
  // functions
  // the two functions fn() enable capturing of functions to use in the algorithm
  template<auto Function, typename = std::enable_if_t<std::is_invocable_r_v<T, decltype(Function), const std::vector<T>& > > >
  void    fn();
  template<auto Function, typename Class, typename = std::enable_if_t<std::is_invocable_r_v<T, decltype(Function), Class*, const std::vector<T>& > > >
  void    fn(Class* cls);
  
  void    set_bounds(const std::vector<T>& lower, const std::vector<T>& upper);
  void    set_bounds(const std::vector<T>& bound, bool lower = true);
  auto    operator()(const std::vector<T>& vec) const -> T;
  void    minimise(); // the optim algorithm
  std::vector<T> values() const;
  
private:
  [[noreturn]]
  static auto null_fn(void* p, long n, const T* x) -> T {throw std::exception{};}
  
  void*                   optim_instance = nullptr;  // pointer to the class if a member function
  func                    optim_fn = &null_fn;        // pointer to the function
  size_t                  dim;                       // number of dimensions
  std::vector<T>          lower_bound;               // bounds
  std::vector<T>          upper_bound;   
  T                       min_f;                     // current best value
  int                     fn_counter = 0;
  int                     iter = 0;
  std::vector<T>          current_values;
  
  //functions
  auto            eval(const std::vector<T>& vec) -> T;
};


template <typename T>
inline optim<T(const std::vector<T>&),NEWUOA>::optim(const std::vector<T>& start) {
  dim = start.size();
  current_values.resize(dim);
  current_values = start;
}

template <typename T>
inline std::vector<T> optim<T(const std::vector<T>&),NEWUOA>::values() const {
  return current_values;
}

template <typename T>
inline void optim<T(const std::vector<T>&),NEWUOA>::set_bounds(const std::vector<T>& lower, const std::vector<T>& upper) {
  lower_bound.resize(dim);
  upper_bound.resize(dim);
  lower_bound = lower; 
  upper_bound = upper;
};

template <typename T>
inline void optim<T(const std::vector<T>&),NEWUOA>::set_bounds(const std::vector<T>& bound, bool lower) {
  if(lower){
    lower_bound.resize(dim);
    lower_bound = bound;
  } else {
    upper_bound.resize(dim);
    upper_bound = bound;
  }
};

template <typename T>
template <auto Function, typename>
inline void optim<T(const std::vector<T>&),NEWUOA>::fn() 
{
  optim_instance = nullptr;
  optim_fn = static_cast<func>([](void*, long n, const T* x) -> T {
    return std::invoke(Function, std::vector<T>(x,x+n));
  });
};

template <typename T>
template<auto Function, typename Class, typename>
inline void optim<T(const std::vector<T>&),NEWUOA>::fn(Class* cls)
{
  optim_instance = cls;
  optim_fn = static_cast<func>([](void* p, long n, const T* x) -> T {
    auto* c = static_cast<Class*>(p);
    return std::invoke(Function,c,std::vector<T>(x,x+n));
  });
}

template <typename T>
inline auto optim<T(const std::vector<T>&),NEWUOA>::operator()(const std::vector<T>& vec) const -> T
{
  return std::invoke(optim_fn,optim_instance,vec.size(),vec.data());
}    

template <typename T>
inline void optim<T(const std::vector<T>&),NEWUOA>::minimise(){
    fn_counter = 0;
#ifndef R_BUILD
    double R_NegInf = -1.0 * std::numeric_limits<double>::infinity();
    double R_PosInf = std::numeric_limits<double>::infinity();
#endif
    if (!control.npt) control.npt = std::min(dim + 2, (dim+2)*(dim+1)/2);     
    if(lower_bound.empty()){
      lower_bound.resize(dim);
      for(int i = 0; i< dim; i++) lower_bound[i] = R_NegInf;
    }    
    if(upper_bound.empty()){
      upper_bound.resize(dim);
      for(int i = 0; i< dim; i++)upper_bound[i] = R_PosInf;
    }
    double max_par = *std::max_element(current_values.begin(),current_values.end());
    if (!control.rhobeg) control.rhobeg = std::min(0.95, 0.2*max_par);
    if (!control.rhoend) control.rhoend = 1.0e-6 * control.rhobeg;    
    if (!control.maxfun) control.maxfun = 10000;
    std::vector<double> w;
    w.resize((control.npt + 5) * (control.npt + dim) + (3 * dim * (dim + 5))/2);   
    auto closure = NewuoaClosure{optim_instance, optim_fn};
    fn_counter = 0;
    double result = newuoa_closure( &closure, dim, control.npt, current_values.data(), control.rhobeg, control.rhoend, control.maxfun, w.data(), &fn_counter);
    min_f = eval(current_values);
#ifdef R_BUILD
    if(control.trace >= 1)
    {
      Rcpp::Rcout << "\nEND NEWUOA | fn: " << fn_counter ;  
    }
#endif
}

template <typename T>
inline auto optim<T(const std::vector<T>&),NEWUOA>::eval(const std::vector<T>& vec) -> T
{   
  fn_counter++;
  return std::invoke(optim_fn, optim_instance, vec.size(), vec.data());
}



typedef optim<double(const std::vector<double>&),BOBYQA> bobyqad;
typedef optim<float(const std::vector<float>&),BOBYQA> bobyqaf;
typedef optim<double(const std::vector<double>&),NEWUOA> newuoad;
typedef optim<float(const std::vector<float>&),NEWUOA> newuoaf;



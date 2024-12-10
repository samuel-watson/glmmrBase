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
#include "lbfgs.h"
#include "lbfgsb.h"
#ifdef R_BUILD
#include <Rcpp.h> // for printing to R
#endif

using namespace Eigen;

// this will be used as the basis of adding more funcitons at a later date

class optim_algo {};
class DIRECT : public optim_algo {};
class BOBYQA : public optim_algo {};
class NEWUOA : public optim_algo {};
class LBFGS : public optim_algo {};



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



// class for L-BFGS
template <>
class optim<double(const VectorXd&, VectorXd&),LBFGS> {
  using func = double(*)(void*, const VectorXd&, VectorXd&); 
public:
  struct optimControl {
    double g_epsilon = 1.0e-8;
    double past = 3;
    double delta = 1.0e-8;
    int max_linesearch = 64;
    int trace = 0;
  } control;
  
  optim(const VectorXd& start);
  optim(const optim& x) = default;
  auto operator=(const optim& x) -> optim& = default;
  
  // functions
  // the two functions fn() enable capturing of functions to use in the algorithm
  template<auto Function, typename = std::enable_if_t<std::is_invocable_r_v<double, decltype(Function), const VectorXd&, VectorXd& > > >
  void                fn();
  template<auto Function, typename Class, typename = std::enable_if_t<std::is_invocable_r_v<double, decltype(Function), Class*, const VectorXd&, VectorXd& > > >
  void                fn(Class* cls);

  auto                operator()(const VectorXd& vec, VectorXd& g) -> double;
  void                minimise(); // the optim algorithm
  VectorXd            values() const;
  void                set_bounds(const VectorXd& lower, const VectorXd& upper);
  void                set_bounds(const std::vector<double>& lower, const std::vector<double>& upper);
  
private:
  [[noreturn]]
  static auto null_fn(void* p, const VectorXd& x, VectorXd& g) -> double {throw std::exception{};}
  
  void*            optim_instance = nullptr;  // pointer to the class if a member function
  func             optim_fn = &null_fn;        // pointer to the function
  size_t           dim;                       // number of dimensions
  double           min_f = 0;                     // current best value
  VectorXd         current_values;
  VectorXd         lower_bound;               // bounds
  VectorXd         upper_bound;   
  int              fn_counter = 0;
  int              iter = 0;
  bool             bounded = false;

  //functions
  double           eval(const VectorXd& vec, VectorXd& g);
};


typedef optim<double(const std::vector<double>&),BOBYQA> bobyqad;
typedef optim<float(const std::vector<float>&),BOBYQA> bobyqaf;
typedef optim<double(const std::vector<double>&),NEWUOA> newuoad;
typedef optim<float(const std::vector<float>&),NEWUOA> newuoaf;
typedef optim<double(const VectorXd&, VectorXd&),LBFGS> lbfgsd;


// below is an implementation of the DIRECT algorithm. It works but is much inferior for the maximum likelihood problems!
// I've left it in here as it might be useful for comparison purposes, or if refinements can be identified.

enum class Position {
  Lower,
  Middle,
  Upper
};

// hyperrectangle class - coordinates should be in [0,1]^D
template <typename T>
class Rectangle {
public:
  int                 dim;
  std::vector<T>      min_x;
  std::vector<T>      max_x;
  T fn_value;
  T max_dim_size;
  bool potentially_optimal = false;

  Rectangle(){};
  Rectangle(const int dim_) : dim(dim_), min_x(dim), max_x(dim) {};
  Rectangle(const std::vector<T>& min_x_, const std::vector<T>& max_x_) : dim(min_x_.size()), min_x(min_x_), max_x(max_x_) {};
  Rectangle(const Rectangle<T>& x) : dim(x.dim), min_x(x.min_x), max_x(x.max_x) {};
  auto operator=(const Rectangle<T>& x) -> Rectangle& = default;

  // functions
  std::vector<T>        centroid(); // returns the centroid
  std::vector<T>        centroid(const size_t& dim, const T& delta); // returns the centroid offset by delta in dimension dim
  void                  unit_hyperrectangle(); //sets the rectangle to be the unit hyper-rectangle
  std::pair<T,size_t>   longest_side(); // returns 0.5 times the longest side
  T                     dim_size(const size_t& dim) const; // returns the size of the dimension
  void                  trim_dimension(const size_t& dim_t, const Position pos); // reduces the dimension to a third of the size - either lower, middle or upper
  void                  set_bounds(const std::vector<T>& min, const std::vector<T>& max); //set min x, max x
};

// class for handling function binding
template <typename T>
class optim<T(const std::vector<T>&),DIRECT> {
  using func = T(*)(const void*, const std::vector<T>&);
public:
  struct optimControl {
    T epsilon = 1e-4;
    int max_iter = 1;
    T tol = 1e-4;
    bool select_one = true; //select only one potentially optimal rectangle on each iteration
    bool trisect_once = false; // trisect only one side per division
    int trace = 0;
    int max_eval = 0;
    bool mrdirect = false; // use a multilevel refinement process
    T l2_tol = 1e-2;
    T l1_tol = 1e-4;
    T l2_epsilon = 1e-5;
    T l1_epsilon = 1e-7;
    T l0_epsilon = 0;
  } control;

  optim(){};
  // if starting vals is true, it assumes x is a vector of starting values, and y sets the bounds by adding +/- either side of x
  // otherwise it assumes x is a lower bound, and y is an upper bound
  optim(const std::vector<T>& x, const std::vector<T>& y, bool starting_vals = true);
  optim(const std::vector<T>& x);
  optim(const optim& x) = default;
  auto operator=(const optim& x) -> optim& = default;

  // functions
  // the two functions fn() enable capturing of functions to use in the algorithm
  template<auto Function, typename = std::enable_if_t<std::is_invocable_r_v<T, decltype(Function), const std::vector<T>& > > >
  void    fn();
  template<auto Function, typename Class, typename = std::enable_if_t<std::is_invocable_r_v<T, decltype(Function), Class*, const std::vector<T>& > > >
  void    fn(Class* cls);

  void            set_bounds(const std::vector<T>& lower, const std::vector<T>& upper, bool starting_vals = true);
  auto            operator()(const std::vector<T>& vec) const -> T;
  void            minimise(); // the optim algorithm
  std::vector<T>  values() const;
  T               rect_size() const;

private:
  [[noreturn]]
  static auto null_fn(const void* p, const std::vector<T>& vec) -> T {throw std::exception{};}

  const void*               optim_instance = nullptr;  // pointer to the class if a member function
  func                      optim_fn = &null_fn;        // pointer to the function
  size_t                    dim;                       // number of dimensions
  std::vector<T>            lower_bound;               // bounds
  std::vector<T>            upper_bound;
  std::vector<T>            dim_size;                  // size of each dimension to transform to unit rectangle
  std::vector<std::unique_ptr<Rectangle<T>>> rects;   // the rectangles
  T                         min_f = 0;               // current best value
  int                       fn_counter = 0;
  int                       iter = 0;
  std::vector<T>            current_values;
  std::pair<T,size_t>       current_largest_dim;
  size_t                    mrdirect_level = 2;
  T                         max_diff = control.tol * 1.1;

  //functions
  auto            eval(const std::vector<T>& vec) -> T;
  std::vector<T>  transform(const std::vector<T>& vec);
  size_t          update_map();
  void            filter_rectangles(size_t n_optimal);
  void            divide_rectangles();
};


typedef optim<double(const std::vector<double>&),DIRECT> directd;
typedef optim<float(const std::vector<float>&),DIRECT> directf;

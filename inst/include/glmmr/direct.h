#pragma once

#define R_BUILD

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
#ifdef R_BUILD
#include <Rcpp.h> // for printing to R
#endif

enum class Position {
  Lower,
  Middle,
  Upper
};

template<typename Signature>
class direct;

inline size_t random_index(size_t max_index) {
  std::random_device                      rand_dev;
  std::mt19937                            generator(rand_dev());
  std::uniform_int_distribution<size_t>   distr(0, max_index);
  return distr(generator);
}


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
  T                     dim_size(const size_t& dim); // returns the size of the dimension
  void                  trim_dimension(const size_t& dim_t, const Position pos); // reduces the dimension to a third of the size - either lower, middle or upper
  void                  set_bounds(const std::vector<T>& min, const std::vector<T>& max); //set min x, max x
};

// class for handling function binding
template <typename T>
class direct<T(const std::vector<T>&)> {
  using func = T(*)(const void*, const std::vector<T>&); 
public:
  struct DirectControl {
    T epsilon = 1e-4;
    int max_iter = 1;
    T tol = 1e-4;
    bool select_one = true; //select only one potentially optimal rectangle on each iteration
    bool adaptive = false; // use adaptive epsilon setting
  } control;
  
  direct(){};
  // if starting vals is true, it assumes x is a vector of starting values, and y sets the bounds by adding +/- either side of x
  // otherwise it assumes x is a lower bound, and y is an upper bound
  direct(const std::vector<T>& x, const std::vector<T>& y, bool starting_vals = true);
  direct(const direct& x) = default;
  auto operator=(const direct& x) -> direct& = default;
  
  // functions
  // the two functions fn() enable capturing of functions to use in the algorithm
  template<auto Function, typename = std::enable_if_t<std::is_invocable_r_v<T, decltype(Function), const std::vector<T>& > > >
  void    fn();
  template<auto Function, typename Class, typename = std::enable_if_t<std::is_invocable_r_v<T, decltype(Function), Class*, const std::vector<T>& > > >
  void    fn(Class* cls);
  
  void    set_bounds(const std::vector<T>& lower, const std::vector<T>& upper, bool starting_vals = true);
  auto    operator()(const std::vector<T>& vec) const -> T;
  void    optim(); // the direct algorithm
  std::vector<T> values() const;
  
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
  
  //functions
  auto            eval(const std::vector<T>& vec) -> T;
  std::vector<T>  transform(const std::vector<T>& vec);
  void            update_map();
  void            filter_rectangles();
  void            divide_rectangles();
};

template <typename T>
inline std::vector<T> Rectangle<T>::centroid(){
  std::vector<T> centre(dim);
  for(size_t i = 0; i < dim; i++){
    centre[i] = 0.5*(max_x[i] - min_x[i]);
  }
  return(centre);
};

template <typename T>
inline void Rectangle<T>::unit_hyperrectangle(){
  std::fill(max_x.begin(),max_x.end(),1.0);
  std::fill(min_x.begin(),min_x.end(),0.0);
};

template <typename T>
inline std::vector<T> Rectangle<T>::centroid(const size_t& dim_ex, const T& delta){
  std::vector<T> centre(dim);
  for(size_t i = 0; i < dim; i++){
    centre[i] = 0.5*(max_x[i] - min_x[i]);
    if(i == dim_ex) centre[i] += delta;
  }
  return(centre);
};

template <typename T>
inline void Rectangle<T>::set_bounds(const std::vector<T>& min, const std::vector<T>& max) {
  dim = min.size();
  min_x = min;
  max_x = max;
}

template <typename T>
inline std::pair<T,size_t> Rectangle<T>::longest_side(){
  T long_len = 0;
  size_t which_dim;
  for(size_t i = 0; i < dim; i++){
    T diff = max_x[i] - min_x[i];
    if(diff > long_len) {
      long_len = diff;
      which_dim = i;
    }
  }
  return std::pair<T,size_t>{0.5*long_len,which_dim};
};

template <typename T>
inline T Rectangle<T>::dim_size(const size_t& dim_){
  return max_x[dim_] - min_x[dim_];
};

template <typename T>
inline void Rectangle<T>::trim_dimension(const size_t& dim_t, const Position pos){
  T dsize = dim_size(dim_t);
  switch(pos){
  case Position::Lower:
    max_x[dim_t] -= 2*dsize/3.0;
    break;
  case Position::Middle:
    min_x[dim_t] += dsize/3.0;
    max_x[dim_t] -= dsize/3.0;
    break;
  case Position::Upper:
    min_x[dim_t] += 2*dsize/3.0;
    break;
  }        
};

template <typename T>
inline direct<T(const std::vector<T>&)>::direct(const std::vector<T>& x, const std::vector<T>& y, bool starting_vals) {
  set_bounds(x,y,starting_vals);
};

template <typename T>
inline std::vector<T> direct<T(const std::vector<T>&)>::values() const {
  return current_values;
}

template <typename T>
inline void direct<T(const std::vector<T>&)>::set_bounds(const std::vector<T>& x, const std::vector<T>& y, bool starting_vals) {
  dim = x.size();
  lower_bound.resize(dim);
  upper_bound.resize(dim);
  dim_size.resize(dim);
  if(!starting_vals)
  {
    lower_bound = x; 
    upper_bound = y;
    for(size_t i = 0; i < dim; i++) dim_size[i] = y[i] - x[i];
  } else {
    for(size_t i = 0; i < dim; i++){
      lower_bound[i] = x[i] - y[i];
      upper_bound[i] = x[i] + y[i];
      dim_size[i] = 2*y[i];
    } 
  }
  current_values.resize(dim);
  std::fill(current_values.begin(),current_values.end(),0.0);
  rects.push_back(std::make_unique<Rectangle<T>>(dim));
  rects.back()->unit_hyperrectangle();
  rects.back()->max_dim_size = 0.5;
};

template <typename T>
template <auto Function, typename>
inline void direct<T(const std::vector<T>&)>::fn() 
{
  optim_instance = nullptr;
  optim_fn = static_cast<func>([](const void*, const std::vector<T>& vec) -> T {
    return std::invoke(Function, vec);
  });
};

template <typename T>
template<auto Function, typename Class, typename>
inline void direct<T(const std::vector<T>&)>::fn(Class* cls)
{
  optim_instance = cls;
  optim_fn = static_cast<func>([](const void* p, const std::vector<T>& vec) -> T {
    auto* c = const_cast<Class*>(static_cast<const Class*>(p));
    return std::invoke(Function,c,vec);
  });
}

template <typename T>
inline auto direct<T(const std::vector<T>&)>::operator()(const std::vector<T>& vec) const -> T
{
  return std::invoke(optim_fn,optim_instance,vec);
}  

template <typename T>
inline void direct<T(const std::vector<T>&)>::optim(){
#ifdef R_BUILD
  Rcpp::Rcout << "\nSTARTING DIRECT-L";
  Rcpp::Rcout << "\nTolerance: " << control.tol << " | Max iter : " << control.max_iter << "\n Starting values :";
  std::vector<T> vals = transform(rects.front()->centroid());
  for(const auto& val: vals) Rcpp::Rcout << val << " ";    
#endif
  current_values = transform(rects.back()->centroid());
  rects.back()->fn_value = eval(current_values);
  min_f = rects.back()->fn_value;
  T max_diff = control.tol*1.1;
  iter = 0;
  fn_counter = 0;
  
  while(max_diff > control.tol && iter <= control.max_iter){
    
#ifdef R_BUILD
    Rcpp::Rcout << "\n---------------------------------------------------------------------------------- ";
    Rcpp::Rcout << "\n| Iter: " << iter << " | Evaluations: " << fn_counter << " | Rectangles: " << rects.size() << " | Dimensions: " << dim << " | Start fn: " << min_f << " |";
#endif
    
    update_map();
    if(control.select_one) filter_rectangles();
    divide_rectangles();    
    max_diff = 2*current_largest_dim.first*dim_size[current_largest_dim.second];
    
#ifdef R_BUILD
    Rcpp::Rcout << "\n| New best fn: " << min_f << " | Max difference: " << max_diff << " | New values: ";
    for(const auto& val: current_values) Rcpp::Rcout << val << " ";
    Rcpp::Rcout << " |\n----------------------------------------------------------------------------------";
#endif
    // erase the rectangles from the size map
    iter++;
  }
}

template <typename T>
inline auto direct<T(const std::vector<T>&)>::eval(const std::vector<T>& vec) -> T
{   
  fn_counter++;
  return std::invoke(optim_fn,optim_instance,vec);
}  

template <typename T>
inline std::vector<T> direct<T(const std::vector<T>&)>::transform(const std::vector<T>& vec)
{
  std::vector<T> transformed_vec(dim);
  for(int i = 0; i < dim; i++){
    transformed_vec[i] = vec[i]*dim_size[i] + lower_bound[i];
  }
  return transformed_vec;
};

// after running this function size_fn should contain the potentially optimal rectangles
template <typename T>
inline void direct<T(const std::vector<T>&)>::update_map()
{
  std::sort(rects.begin(), rects.end(), [](const std::unique_ptr<Rectangle<T>>& x, const std::unique_ptr<Rectangle<T>>& y){return x->max_dim_size < y->max_dim_size;});
  
  // variables
  std::pair<T,T>                  coord = {0.0, min_f - control.epsilon*abs(min_f)};
  size_t                          index = 0; 
  size_t                          end = rects.size();
  T                               angle = M_PI*0.5;
  T                               x, y, new_angle;
  bool                            better_value = false;
  
  // function body
  while(index < end){
    if(index == (end-1)){
      rects[index]->potentially_optimal = true;
      index = end;
    } else {
      better_value = false;
      y = abs(rects[index]->fn_value - coord.second);
      x = abs(rects[index]->max_dim_size - coord.first);
      angle = abs(atan(y/x));
      size_t iter = index + 1;
      while(iter < end && !better_value){
        y = abs(rects[iter]->fn_value - coord.second);
        x = abs(rects[iter]->max_dim_size - coord.first);
        new_angle = abs(atan(y/x));
        if(new_angle < angle){
          better_value = true;
        } else {
          iter++;
        }
      }
      if(better_value){
        index = iter;
      } else {
        coord.first = rects[index]->max_dim_size;
        coord.second = rects[index]->fn_value;
        rects[index]->potentially_optimal = true;
        index++;
      }
    }
  }
};


// selects just one potentially optimal rectangle
template <typename T>
inline void direct<T(const std::vector<T>&)>::filter_rectangles()
{
  size_t rect_fn_size = 0;
  for(const auto& r: rects){
    if(r->potentially_optimal) rect_fn_size++;
  }
  if(rect_fn_size > 1){
    size_t keep_index = random_index(rect_fn_size -1);
    size_t counter = 0;
    for(auto& r: rects){
      if(r->potentially_optimal){
        if(counter != keep_index)r->potentially_optimal = false;
        counter++;
      }
    }
  }
}

// divides up the potentially optimal rectangles 
// it adds the rectangles to rects, and then erases the original 
// rectangles, while also clearing size_fn
template <typename T>
inline void direct<T(const std::vector<T>&)>::divide_rectangles(){
  //identify the largest dimensions
  typedef std::pair<T,size_t> dimpair;
  
  struct compare_pair {
    bool operator()(const dimpair& elt1, const dimpair& elt2) const {
      return elt1.first < elt2.first;
    };
  };
  
#ifdef R_BUILD
  Rcpp::Rcout << "\nDIVIDING RECTANGLES ";
#endif
  
  std::vector<size_t> largest_dims;
  std::priority_queue< dimpair, std::vector<dimpair>, compare_pair > pq;
  T curr_dim_size;
  std::vector<std::unique_ptr<Rectangle<T>>> new_rectangles;
  size_t counter = 0;
  
  for(auto& r: rects){
    if(r->potentially_optimal){
      largest_dims.clear();
      curr_dim_size = 2*r->longest_side().first; 
      for(size_t i = 0; i < dim; i++){
        if(r->dim_size(i) == curr_dim_size) largest_dims.push_back(i);
      }
      T delta = curr_dim_size / 3;
      
#ifdef R_BUILD
      if(largest_dims.size()==0)Rcpp::stop("No dimension data");
      Rcpp::Rcout << "\nRECTANGLE " << counter << " | Largest dim size: " << r->max_dim_size << " delta: " << delta << " in dimensions: ";
      for(const auto& val: largest_dims) Rcpp::Rcout << val << " ";
#endif
      
      T fn1, fn2, fnmin;
      for(const auto& dd: largest_dims){
        fn1 = eval(transform(r->centroid(dd, delta)));
        fn2 = eval(transform(r->centroid(dd, -delta)));
        fnmin = std::min(fn1,fn2);
        pq.push(dimpair(fnmin, dd));
      }
      
      if(r->fn_value < min_f){
        current_values = transform(r->centroid());
        current_largest_dim = r->longest_side();  
        min_f = r->fn_value;
      }
      
      while(!pq.empty()){
        size_t dim_vvv = pq.top().second;
        new_rectangles.push_back(std::make_unique<Rectangle<T>>(dim));
        new_rectangles.back()->set_bounds(r->min_x,r->max_x);
        new_rectangles.back()->trim_dimension(dim_vvv,Position::Lower);
        new_rectangles.back()->fn_value = eval(transform(new_rectangles.back()->centroid()));
        new_rectangles.back()->max_dim_size = new_rectangles.back()->longest_side().first;
        if(new_rectangles.back()->fn_value < min_f){
          current_values = transform(new_rectangles.back()->centroid());
          current_largest_dim = new_rectangles.back()->longest_side();  
          min_f = new_rectangles.back()->fn_value;
        }
        
        new_rectangles.push_back(std::make_unique<Rectangle<T>>(dim));
        new_rectangles.back()->set_bounds(r->min_x,r->max_x);
        new_rectangles.back()->trim_dimension(dim_vvv,Position::Upper);
        new_rectangles.back()->fn_value = eval(transform(new_rectangles.back()->centroid()));
        new_rectangles.back()->max_dim_size = new_rectangles.back()->longest_side().first;
        if(new_rectangles.back()->fn_value < min_f){
          current_values = transform(new_rectangles.back()->centroid());
          current_largest_dim = new_rectangles.back()->longest_side();  
          min_f = new_rectangles.back()->fn_value;
        }
        
        r->trim_dimension(dim_vvv, Position::Middle);
        pq.pop();

      }
      r->potentially_optimal = false;
    } 
    counter++;
  }
  
  // insert new rectangles
  for(int i = 0; i < new_rectangles.size(); i++) rects.push_back(std::move(new_rectangles[i]));
  new_rectangles.clear();
};


typedef direct<double(const std::vector<double>&)> directd;
typedef direct<float(const std::vector<float>&)> directf;

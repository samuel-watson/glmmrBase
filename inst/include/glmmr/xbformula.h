#ifndef XBFORMULA_HPP
#define XBFORMULA_HPP

#include <RcppEigen.h>
#include <vector>
#include <string>
#include <cstring>
#include <cctype>
#include <algorithm>

#include "general.h"
#include "interpreter.h"


using namespace Eigen;

namespace glmmr{

class xbFormula {
  public:
    str formula_;
    
    xbFormula(const str& formula,
              const ArrayXXd& data,
              const strvec& colnames) : formula_(formula) {
      parse(data,colnames);
    };
    
    xbFormula(const str& par,
              const ArrayXd& data) : formula_(par) {
      gen_linear(data);
    }
    
    xbFormula(int n) : formula_("[Intercept]") {
      ArrayXd ones = ArrayXd::Ones(n);
      gen_linear(ones);
    }
    
    void update_parameters(const dblvec& parameters){
      parameters_ = parameters;
    }
    
    void update_parameters(const ArrayXd& parameters){
      if(parameters_.size()==0){
        for(int i = 0; i < parameters.size(); i++){
          parameters_.push_back(parameters(i));
        }
      } else if(parameters_.size() == parameters.size()){
        for(int i = 0; i < parameters.size(); i++){
          parameters_[i] = parameters(i);
        }
      } else {
        Rcpp::stop("Wrong number of parameters");
      }
    }
    
    VectorXd xb(){
      VectorXd xb_out(n_);
      for(int i = 0; i<n_; i++){
        xb_out(i) = glmmr::calculate(instructions_,
                                     indexes_,
                                     parameters_,
                                     data_,
                                     i,i);
      }
      return xb_out;
    }
    
    // add a function for creating a linearised X 
    VectorXd X(){
      VectorXd x_out(n_);
      for(int i = 0; i<n_; i++){
        x_out(i) = data_[i][0];
      }
      if(nonlinear_){
        Rcpp::Rcout << "WARNING: Linearisation of nonlinear functions not implemented for X"; 
      }
      return x_out;
    }
    
    int pars(){
      return parnames_.size();
    }
    
    strvec parnames(){
      return parnames_;
    }
    
  private:
    intvec instructions_;
    dblvec2d data_;
    intvec indexes_;
    dblvec parameters_;
    strvec parnames_;
    bool nonlinear_;
    int n_;
    
    void parse(const ArrayXXd& data,
               const strvec& colnames){
      // iterate over the formula and prepare the RPN
      n_ = data.rows();
      data_.resize(n_);
      read_formula(formula_,
                   data,
                   colnames,
                   instructions_,
                   data_,
                   indexes_,
                   parnames_);
    }
    
    void gen_linear(const ArrayXd& data){
      parnames_.push_back(formula_);
      instructions_.push_back(0);
      instructions_.push_back(2);
      instructions_.push_back(5);
      indexes_.push_back(0);
      indexes_.push_back(0);
      data_.resize(data.size());
      for(int i = 0; i < data.size(); i++){
        data_[i].push_back(data(i));
      }
      parameters_.resize(1);
      nonlinear_ = false;
    }
    
    void read_formula(const str& formula_,
                      const ArrayXXd& data,
                      const strvec& colnames,
                      intvec& instructions,
                      dblvec2d& data_vec,
                      intvec& indexes,
                      strvec& parnames){
      int cursor = 0;
      str temp_form;
      int nsize = formula_.size();
      int data_count = 0;
      int parameter_count = 0;
      int n = data_vec.size();
      if(n != data.rows())Rcpp::stop("data and data_vec different dimension");
      int top_of_stack;
      while(cursor<nsize){
        if(formula_[cursor]=='~'){
          cursor++;
        } else if(formula_[cursor]=='('){
          str string_in_brackets;
          int bracket_counter = 0;
          while(!(formula_[cursor+1]==')' && bracket_counter == 0)){
            cursor++;
            if(formula_[cursor]=='(')bracket_counter++;
            if(formula_[cursor]==')')bracket_counter--;
            string_in_brackets.push_back(formula_[cursor]);
          }
          Rcpp::Rcout << "\nTerm in brackets: " << string_in_brackets;
          
          intvec nested_instructions;
          dblvec2d nested_data_vec;
          intvec nested_indexes;
          strvec nested_parnames;
          
          nested_data_vec.resize(n);
          
          read_formula(string_in_brackets,
                       data,
                       colnames,
                       nested_instructions,
                       nested_data_vec,
                       nested_indexes,
                       nested_parnames);
          
          instructions.insert(instructions.end(),nested_instructions.begin(),nested_instructions.end());
          indexes.insert(indexes.end(),nested_indexes.begin(),nested_indexes.end());
          parnames.insert(parnames.end(),nested_parnames.begin(),nested_parnames.end());
          for(int i = 0; i < n; i++){
            data_vec[i].insert(data_vec[i].end(),nested_data_vec[i].begin(),nested_data_vec[i].end());
          }
          
          cursor += 2;
          
        } else {
          if(isalnum(formula_[cursor])){
            temp_form.push_back(formula_[cursor]);
            if(cursor >= (nsize-1) || !isalnum(formula_[cursor+1])){
              Rcpp::Rcout << "\nToken: " << temp_form;
              auto colidx = std::find(colnames.begin(),colnames.end(),temp_form);
              bool token_in_cols = colidx == colnames.end();
              
              if(temp_form == "exp" && token_in_cols){
                instructions.push_back(9);
              } else if(temp_form == "log" && token_in_cols){
                instructions.push_back(16);
              } else if(temp_form == "sqrt" && token_in_cols){
                instructions.push_back(7);
              } else if(temp_form == "sin" && token_in_cols){
                instructions.push_back(13);
              } else if(temp_form == "cos" && token_in_cols){
                instructions.push_back(14);
              } else if(temp_form == "fexp" && token_in_cols){
                // fexp pre-programmed
                intvec B = {2,0,2,5,8,5};
                intvec Bidx = {0,0,1};
                instructions_.insert(instructions_.end(),B.begin(),B.end());
                indexes_.insert(indexes_.end(),Bidx.begin(),Bidx.end());
                
                cursor += 2;
                str string_in_brackets;
                while(!(formula_[cursor+1]==')')){
                  cursor++;
                  if(formula_[cursor]=='(')Rcpp::stop("Extra bracket in function.");
                  string_in_brackets.push_back(formula_[cursor]);
                }
                Rcpp::Rcout << "\nTerm in brackets: " << string_in_brackets;
                
                auto colidx_int = std::find(colnames.begin(),colnames.end(),string_in_brackets);
                if(colidx_int == colnames.end())Rcpp::stop("Variable in fexp not in data");
                int column_index_int = colidx_int - colnames.begin();
                indexes.push_back(data_count);
                data_count++;
                for(int i = 0; i < n_; i++){
                  data_vec[i].push_back(data(i,column_index_int));
                }
                parnames_.push_back("b_fexp_"+string_in_brackets+"_1");
                parnames_.push_back("b_fexp_"+string_in_brackets+"_2");
                parameter_count+=2;
                cursor++;
              } else if(!token_in_cols){
                instructions.push_back(0);
                int column_index = colidx - colnames.begin();
                indexes.push_back(data_count);
                data_count++;
                for(int i = 0; i < n; i++){
                  data_vec[i].push_back(data(i,column_index));
                }
                parnames_.push_back("b_"+temp_form);
              } else if(glmmr::is_number(temp_form)) {
                // add an integer to the stack
                int p = temp_form.size();
                int addint;
                
                if(p > 1){
                  for(int i = 0; i < (p-1); i++){
                    instructions.push_back(3);
                  }
                }
                
                for(int i = 1; i <= p; i++){
                  addint = (int)temp_form[p-i] + 20;
                  if(i==1){
                    instructions.push_back(addint);
                  } else {
                    instructions.push_back(5);
                    instructions.push_back(addint);
                    instructions.push_back(6);
                    instructions.push_back(i-1);
                    instructions.push_back(20);
                  }
                }
              } else {
                // token is a parameter name
                instructions.push_back(2);
                parnames.push_back(temp_form);
                indexes.push_back(parameter_count);
                parameter_count++;
              }
              temp_form.clear();
            }
          } else if(formula_[cursor] == '+'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(3);
            instructions.push_back(top_of_stack);
          } else if(formula_[cursor] == '-'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(4);
            instructions.push_back(top_of_stack);
          } else if(formula_[cursor] == '/'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(6);
            instructions.push_back(top_of_stack);
          } else if(formula_[cursor] == '*'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(5);
            instructions.push_back(top_of_stack);
          } else if(formula_[cursor] == '^'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(8);
            instructions.push_back(top_of_stack);
          } else if(formula_[cursor] == '.'){
            Rcpp::stop("Decimal point/'.' character not currently supported");
          }
          cursor++;
        }
      }
      std::reverse(instructions.begin(),instructions.end());
      std::reverse(indexes.begin(),indexes.end());
      parameters_.resize(parnames_.size());
      nonlinear_ = true;
      auto find_par = std::find(instructions.begin(),instructions.end(),2);
      if(find_par == instructions.end())Rcpp::stop("No parameters in non-linear formula, to include an offset use the offset command.");
    }
};

}


#endif
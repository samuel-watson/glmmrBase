#ifndef XBFORMULA_HPP
#define XBFORMULA_HPP

#include "general.h"
#include "interpreter.h"


using namespace Eigen;

namespace glmmr{

class xbFormula {
  public:
    xbFormula(const str& formula,
              const ArrayXXd& data,
              const strvec& colnames) : formula_(formula.begin(),formula.end()), parameter_count(0), data_count(0) {
      Rcpp::Rcout << "\nFormula: ";
      for(auto ch: formula_)Rcpp::Rcout << ch;
      parse(data,colnames);
    };
    
    xbFormula(int n) : n_(n), parameter_count(0), data_count(0) {
      str intercept = "[Intercept]";
      formula_ = std::vector<char>(intercept.begin(),intercept.end());
      ArrayXd ones = ArrayXd::Ones(n);
      gen_linear(ones);
    }
    
    void update_parameters(const dblvec& parameters){
      if(parameters.size()!=parameters_.size())Rcpp::stop("Wrong length parameter vector (x comp)");
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
    
    str formula(){
      str formula_string(formula_.begin(),formula_.end());
      return formula_string;
    }
    
    dblvec parameters(){
      return parameters_;
    }
    
    bool nonlinear(){
      return nonlinear_;
    }
    
  private:
    std::vector<char> formula_;
    intvec instructions_;
    dblvec2d data_;
    intvec indexes_;
    dblvec parameters_;
    strvec parnames_;
    bool nonlinear_;
    int n_;
    int parameter_count;
    int data_count;
    
    void parse(const ArrayXXd& data,
               const strvec& colnames){
      // iterate over the formula and prepare the RPN
      n_ = data.rows();
      data_.resize(n_);
      str form_as_str(formula_.begin(),formula_.end());
      auto colidx = std::find(colnames.begin(),colnames.end(),form_as_str);
      bool token_in_cols = colidx == colnames.end();

      if(token_in_cols){
        read_formula(formula_,
                     data,
                     colnames,
                     instructions_,
                     data_,
                     indexes_,
                     parnames_);
        std::reverse(instructions_.begin(),instructions_.end());
        std::reverse(indexes_.begin(),indexes_.end());
        if(!glmmr::expect_number_of_unique_elements<str>(parnames_,parameter_count))Rcpp::stop("Not enough uniquely named parameters");
        
        glmmr::print_vec_1d<intvec>(instructions_);
        glmmr::print_vec_1d<strvec>(parnames_);
        glmmr::print_vec_1d<intvec>(indexes_);
      } else {
        int col_number = colidx - colnames.begin();
        gen_linear(data.col(col_number));
      }
    }
    
    void gen_linear(const ArrayXd& data){
      str newparname(formula_.begin(),formula_.end());
      parnames_.push_back(newparname);
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
    
    std::vector<char> extract_term_in_brackets(const std::vector<char>& fvec,
                                               int& cursor){
      std::vector<char> string_in_brackets;
      int bracket_counter = 0;
      while(!(fvec[cursor+1]==')' && bracket_counter == 0)){
        cursor++;
        if(fvec[cursor]=='(')bracket_counter++;
        if(fvec[cursor]==')')bracket_counter--;
        string_in_brackets.push_back(fvec[cursor]);
      }
      Rcpp::Rcout << "\nTerm in brackets: ";
      for(auto ch: string_in_brackets)Rcpp::Rcout << ch;
      
      return string_in_brackets;
    }
    
    void read_formula(const std::vector<char>& formula,
                      const ArrayXXd& data,
                      const strvec& colnames,
                      intvec& instructions,
                      dblvec2d& data_vec,
                      intvec& indexes,
                      strvec& parnames){
      int cursor = 0;
      std::vector<char> temp_form;
      int nsize = formula.size();
      int n = data_vec.size();
      if(n != data.rows())Rcpp::stop("data and data_vec different dimension");
      int top_of_stack;
      while(cursor<nsize){
        if(formula[cursor]=='~'){
          cursor++;
        } else if(formula[cursor]=='('){
          std::vector<char> string_in_brackets = extract_term_in_brackets(formula,cursor);
          
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
          if(glmmr::isalnum_or_uscore(formula[cursor])){
            temp_form.push_back(formula[cursor]);
            if(cursor >= (nsize-1) || !glmmr::isalnum_or_uscore(formula[cursor+1])){
              Rcpp::Rcout << "\nToken: ";
              for(auto ch: temp_form)Rcpp::Rcout << ch;
              str token_as_str(temp_form.begin(),temp_form.end());
              
              auto colidx = std::find(colnames.begin(),colnames.end(),token_as_str);
              bool token_in_cols = colidx == colnames.end();
              
              if(token_as_str == "exp" && token_in_cols){
                instructions.push_back(9);
              } else if(token_as_str == "log" && token_in_cols){
                instructions.push_back(16);
              } else if(token_as_str == "sqrt" && token_in_cols){
                instructions.push_back(7);
              } else if(token_as_str == "sin" && token_in_cols){
                instructions.push_back(13);
              } else if(token_as_str == "cos" && token_in_cols){
                instructions.push_back(14);
              } else if(token_as_str == "fexp" && token_in_cols){
                // fexp pre-programmed
                intvec B = {5,2,9,5,2,0};
                instructions_.insert(instructions_.end(),B.begin(),B.end());
                
                cursor++;
                std::vector<char> string_in_brackets = extract_term_in_brackets(formula,cursor);
                str bracket_as_str(string_in_brackets.begin(),string_in_brackets.end());
                auto colidx_int = std::find(colnames.begin(),colnames.end(),bracket_as_str);
                if(colidx_int == colnames.end())Rcpp::stop("Variable " + bracket_as_str + " not in data");
                int column_index_int = colidx_int - colnames.begin();
                indexes.push_back(data_count);
                indexes.push_back(parameter_count+1);
                indexes.push_back(parameter_count);
                data_count++;
                parameter_count+=2;
                for(int i = 0; i < n_; i++){
                  data_vec[i].push_back(data(i,column_index_int));
                }
                parnames_.push_back("b_fexp_"+bracket_as_str+"_1");
                parnames_.push_back("b_fexp_"+bracket_as_str+"_2");
                cursor++;
              } else if(!token_in_cols){
                // token is the name of a variable
                instructions.push_back(0);
                int column_index = colidx - colnames.begin();
                indexes.push_back(data_count);
                data_count++;
                for(int i = 0; i < n; i++){
                  data_vec[i].push_back(data(i,column_index));
                }
                //parnames_.push_back("b_"+token_as_str);
              } else if(glmmr::is_number(token_as_str)) {
                // add an integer to the stack
                int p = temp_form.size();
                int addint;
                
                if(p > 1){
                  for(int i = 0; i < (p-1); i++){
                    instructions.push_back(3);
                  }
                }
                
                for(int i = 1; i <= p; i++){
                  int number = temp_form[p-i] - '0';
                  addint = number + 20;
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
                parnames.push_back(token_as_str);
                indexes.push_back(parameter_count);
                parameter_count++;
              }
              temp_form.clear();
            }
          } else if(formula[cursor] == '+'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(3);
            instructions.push_back(top_of_stack);
          } else if(formula[cursor] == '-'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(4);
            instructions.push_back(top_of_stack);
          } else if(formula[cursor] == '/'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(6);
            instructions.push_back(top_of_stack);
          } else if(formula[cursor] == '*'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(5);
            instructions.push_back(top_of_stack);
          } else if(formula[cursor] == '^'){
            top_of_stack = instructions.back();
            instructions.pop_back();
            instructions.push_back(8);
            instructions.push_back(top_of_stack);
          } else if(formula[cursor] == '.'){
            Rcpp::stop("Decimal point/'.' character not currently supported");
          }
          cursor++;
        }
      }
      parameters_.resize(parnames_.size());
      nonlinear_ = true;
      auto find_par = std::find(instructions.begin(),instructions.end(),2);
      if(find_par == instructions.end())Rcpp::stop("No parameters in non-linear formula, to include an offset use the offset command.");
      glmmr::print_vec_1d<intvec>(instructions);
      glmmr::print_vec_1d<strvec>(parnames);
    }
};

}


#endif
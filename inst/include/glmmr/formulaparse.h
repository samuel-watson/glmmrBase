#ifndef FORMULAPARSE_H
#define FORMULAPARSE_H

#include "general.h"

namespace glmmr{

inline void parse_formula(std::vector<char>& formula,
                          glmmr::calculator& calc,
                          const ArrayXXd& data,
                          const strvec& colnames){
  int bracket_count = 0;
  int cursor = 0;
  int nchar = formula.size();
  bool has_found_symbol=false;
  int n = data.rows();
  std::vector<char> s1;
  std::vector<char> s2;
  
  // step 1: split at first +
  while(!has_found_symbol && cursor < nchar){
    if(cursor==0 && (formula[cursor]=='+' || formula[cursor]=='-'))Rcpp::stop("Error in formula, plus/minus symbol in wrong place");
    s1.push_back(formula[cursor]);
    if(formula[cursor]=='(')bracket_count++;
    if(formula[cursor]==')')bracket_count--;
    if((formula[cursor+1]=='+' || formula[cursor+1]=='-') && bracket_count == 0)has_found_symbol = true;
    cursor++;
  }
  if(has_found_symbol){
    // split at +/-
    if(formula[cursor]=='+'){
      calc.instructions.push_back(3);
    } else if(formula[cursor]=='-'){
      calc.instructions.push_back(4);
    } else {
      Rcpp::stop("Oops, something has gone wrong (f1)");
    }
    cursor++;
    while(cursor < nchar){
      s2.push_back(formula[cursor]);
      cursor++;
    }
    // check first whether s1 or s2 is the name of a data column
    str s1_as_str(s1.begin(),s1.end());
    auto col_idx = std::find(colnames.begin(),colnames.end(),s1_as_str);
    if(col_idx != colnames.end()){
      str s1_parname = "b_" + s1_as_str;
      s1.push_back('*');
      for(int j = 0; j < s1_parname.size(); j++){
        s1.push_back(s1_parname[j]);
      }
    }
    
    str s2_as_str(s2.begin(),s2.end());
    col_idx = std::find(colnames.begin(),colnames.end(),s2_as_str);
    if(col_idx != colnames.end()){
      str s2_parname = "b_" + s2_as_str;
      s2.push_back('*');
      for(int j = 0; j < s2_parname.size(); j++){
        s2.push_back(s2_parname[j]);
      }
    }
    
    parse_formula(s1,calc,data,colnames);
    parse_formula(s2,calc,data,colnames);
  } else {
    // no +/- to split at, try *//
    s1.clear();
    s2.clear();
    cursor=0;
    bracket_count = 0;
    while(!has_found_symbol && cursor < nchar){
      if(cursor==0 && (formula[cursor]=='*' || formula[cursor]=='/'))Rcpp::stop("Error in formula, multiply/divide symbol in wrong place");
      s1.push_back(formula[cursor]);
      if(formula[cursor]=='(')bracket_count++;
      if(formula[cursor]==')')bracket_count--;
      if((formula[cursor+1]=='*' || formula[cursor+1]=='/') && bracket_count == 0)has_found_symbol = true;
      cursor++;
    }
    if(has_found_symbol){
      // split at *//
      if(formula[cursor]=='*'){
        calc.instructions.push_back(5);
      } else if(formula[cursor]=='/'){
        calc.instructions.push_back(6);
      } else {
        Rcpp::stop("Oops, something has gone wrong (f2)");
      }
      cursor++;
      while(cursor < nchar){
        s2.push_back(formula[cursor]);
        cursor++;
      }
      parse_formula(s1,calc,data,colnames);
      parse_formula(s2,calc,data,colnames);
    } else {
      // no * to split at, try pow
      s1.clear();
      s2.clear();
      cursor=0;
      bracket_count = 0;
      while(!has_found_symbol && cursor < nchar){
        if(cursor==0 && formula[cursor]=='^')Rcpp::stop("Error in formula, power symbol in wrong place");
        s1.push_back(formula[cursor]);
        if(formula[cursor]=='(')bracket_count++;
        if(formula[cursor]==')')bracket_count--;
        if(formula[cursor+1]=='^' && bracket_count == 0)has_found_symbol = true;
        cursor++;
      }
      if(has_found_symbol){
        calc.any_nonlinear = true;
        // split at ^
        if(formula[cursor]=='^'){
          calc.instructions.push_back(8);
        }  else {
          Rcpp::stop("Oops, something has gone wrong (f3)");
        }
        cursor++;
        while(cursor < nchar){
          s2.push_back(formula[cursor]);
          cursor++;
        }
        parse_formula(s1,calc,data,colnames);
        parse_formula(s2,calc,data,colnames);
      } else {
        // no pow, try brackets
        s1.clear();
        s2.clear();
        cursor=0;
        bracket_count = 0;
        while(!has_found_symbol && cursor < nchar){
          if(cursor==0 && formula[cursor]=='(')break;
          s1.push_back(formula[cursor]);
          if(formula[cursor+1]=='(')has_found_symbol = true;
          cursor++;
        }
        if(has_found_symbol){
          calc.any_nonlinear = true;
          cursor++;
          while(!(bracket_count == 0 && formula[cursor]==')') && cursor < nchar){
            s2.push_back(formula[cursor]);
            if(formula[cursor]=='(')bracket_count++;
            if(formula[cursor]==')')bracket_count--;
            cursor++;
          }
          if(formula[cursor]!=')')Rcpp::stop("Matching bracket missing");
          // process s1 as function (if size > 0)
          if(s1.size()>0){
            calc.any_nonlinear = true;
            str token_as_str(s1.begin(),s1.end());
            if(token_as_str == "exp"){
              calc.instructions.push_back(9);
            } else if(token_as_str == "log"){
              calc.instructions.push_back(16);
            } else if(token_as_str == "sqrt"){
              calc.instructions.push_back(7);
            } else if(token_as_str == "sin"){
              calc.instructions.push_back(13);
            } else if(token_as_str == "cos"){
              calc.instructions.push_back(14);
            } else {
              Rcpp::stop("String " + token_as_str + " is not a recognised function");
            }
          }
          parse_formula(s2,calc,data,colnames);
        } else {
          // no brackets - now check data
          str token_as_str(s1.begin(),s1.end());
          auto colidx = std::find(colnames.begin(),colnames.end(),token_as_str);
          if(colidx != colnames.end()){
            // token is the name of a variable
            calc.instructions.push_back(0);
            int column_index = colidx - colnames.begin();
            calc.indexes.push_back(calc.data_count);
            calc.data_count++;
            for(int i = 0; i < n; i++){
              calc.data[i].push_back(data(i,column_index));
            }
          } else {
            // interpret any other string as the name of a parameter
            calc.instructions.push_back(2);
            calc.parameter_names.push_back(token_as_str);
            calc.indexes.push_back(calc.parameter_count);
            calc.parameter_count++;
          }
        }
      }
    }
  }
}

}



#endif
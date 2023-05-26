#ifndef FORMULA_IPP
#define FORMULA_IPP

inline void glmmr::Formula::tokenise(){
  formula_validate();
  
  // remove -1 and set RM_INT is true if its there
  RM_INT = false;
  formula_.erase(std::remove_if(formula_.begin(), formula_.end(), [](unsigned char x) { return std::isspace(x); }), formula_.end());
  auto minone = formula_.find("-1");
  if(minone != str::npos){
    RM_INT = true;
    formula_.erase(minone,2);
  }
  
  // so we have to write our own algorithm to split the tokens
  std::vector<char> formula_as_chars(formula_.begin(),formula_.end());
  int nchar = formula_as_chars.size();
  int cursor = 0;
  int bracket_count = 0; // to deal with opening of brackets
  
  while(cursor < nchar){
    str token;
    while(!(formula_as_chars[cursor]=='+' && bracket_count == 0) && cursor < nchar){
      if(formula_as_chars[cursor]=='(')bracket_count++;
      if(formula_as_chars[cursor]==')')bracket_count--;
      if(!isspace(formula_as_chars[cursor]))token.push_back(formula_as_chars[cursor]);
      cursor++;
    }
    
    if(token.size()>0){
      if(token.front() == '(' && token.back()==')'){
        re_.push_back(token);
      } else{
        fe_.push_back(token);
      }
    }
    
    cursor++;
  }
  
  for(int i =0; i<re_.size(); i++){
    re_order_.push_back(i);
  }

  // // random effects: separate right and left hand sides
  int m = re_.size();
  re_terms_ = re_;

  for(int i = 0; i<m;i++){
    std::stringstream check1(re_[i]);
    str intermediate;
    re_.erase(re_.begin());
    getline(check1, intermediate, '|');
    z_.push_back(intermediate.substr(1,intermediate.length()-1));
    getline(check1, intermediate, '|');
    re_.push_back(intermediate.substr(0,intermediate.length()-1));
  }
  
  for(int i = 0; i < z_.size(); i++){
    z_[i].erase(std::remove_if(z_[i].begin(), z_[i].end(), [](unsigned char x) { return std::isspace(x); }), z_[i].end());
  }
}

inline void glmmr::Formula::formula_validate(){
  //currently only checks if addition inside re term
  int open = 0;
  bool has_a_plus = false;
  bool has_a_vert = false;
  for(auto ch: formula_){
    if(ch=='(')open++;
    if(ch==')'){
      open--;
      if(open == 0){
        has_a_plus = false;
        has_a_vert = false;
      }
    }
    if(ch=='+' && open > 0)has_a_plus = true;
    if(ch=='|' && open > 0)has_a_vert = true;
    
    if(has_a_plus && has_a_vert)Rcpp::stop("Addition inside re term not currently supported");
  }
}

#endif
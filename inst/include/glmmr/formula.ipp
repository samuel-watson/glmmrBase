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
  } else {
    str add_intercept = "b_intercept*1+";
    std::vector<char> add_intercept_vec(add_intercept.begin(),add_intercept.end());
    for(int i = 0; i < add_intercept_vec.size(); i++){
      linear_predictor_.push_back(add_intercept_vec[i]);
    }
  }
  
  // split the tokens
  std::vector<char> formula_as_chars(formula_.begin(),formula_.end());
  // split at plus and then sort
  int nchar = formula_as_chars.size();
  int cursor = 0;
  int bracket_count = 0; // to deal with opening of brackets
  if(formula_as_chars[0]=='+')Rcpp::stop("Cannot start a formula with +");
  std::vector<char> temp_token;
  
  while(cursor <= nchar){
    if((formula_as_chars[cursor]=='+' && bracket_count == 0) || cursor == (nchar)){
      if(temp_token[0]!='('){
        linear_predictor_.insert(linear_predictor_.end(),temp_token.begin(),temp_token.end());
        linear_predictor_.push_back('+');
      } else {
        if(temp_token.back()!=')')Rcpp::stop("Invalid formula, no closing bracket");
        int mm = temp_token.size();
        int cursor_re = 1;
        std::vector<char> temp_token_re;
        while(cursor_re < mm){
          if(temp_token[cursor_re]=='|'){
            str re_new(temp_token_re.begin(),temp_token_re.end());
            z_.push_back(re_new);
            temp_token_re.clear();
          } else if(cursor_re == mm-1){
            str re_new(temp_token_re.begin(),temp_token_re.end());
            re_.push_back(re_new);
          } else {
            temp_token_re.push_back(temp_token[cursor_re]);
          }
          cursor_re++;
        }
      }
      temp_token.clear();
    } else {
      if(formula_as_chars[cursor]=='(')bracket_count++;
      if(formula_as_chars[cursor]==')')bracket_count--;
      temp_token.push_back(formula_as_chars[cursor]);
    }
    cursor++;
  }
  
  if(linear_predictor_.back()=='+')linear_predictor_.pop_back();
  
  for(int i =0; i<re_.size(); i++){
    re_order_.push_back(i);
  }

  // // random effects: separate right and left hand sides
  int m = re_.size();
  re_terms_ = re_;
  
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

inline void glmmr::Formula::calculate_linear_predictor(glmmr::calculator& calculator,const ArrayXXd& data,const strvec& colnames, MatrixXd& Xdata){
  bool outparse = glmmr::parse_formula(linear_predictor_,
                                        calculator,
                                        data,
                                        colnames,
                                        Xdata);
  std::reverse(calculator.instructions.begin(),calculator.instructions.end());
  std::reverse(calculator.indexes.begin(),calculator.indexes.end());
}
 
inline strvec glmmr::Formula::re(){
  return re_;
}

inline strvec glmmr::Formula::z(){
  return z_;
}

inline strvec glmmr::Formula::re_terms(){
  return re_terms_;
}

#endif
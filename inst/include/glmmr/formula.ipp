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
  
  // this code just splits the string at +
  //std::stringstream f1(formula_);
  //str segment;
  //while(std::getline(f1,segment,'+')){
  //  tokens_.push_back(segment);
  //}
  
  // this regex did not work - I can't get it to work!
  // tokenise string - split at + or space except between brackets
  //const std::regex re("([\\+]+)(?![^\\(]*\\|)(?![^\\|]*\\))");
  //std::sregex_token_iterator it{ formula_.begin(), formula_.end(), re, -1 };
  //tokens_ = strvec{ it, {} };
  
  // so we have to write our own algorithm to split the tokens
  int nchar = formula_.size();
  int cursor = 0;
  int bracket_count = 0; // to deal with opening of brackets
  str token;
  while(cursor < nchar){
    Rcpp::Rcout << "\nCursor: " << cursor;
    
    while(!(formula_[cursor]=='+' && bracket_count == 0) || cursor==nchar){
      if(formula_[cursor]=='(')bracket_count++;
      if(formula_[cursor]==')')bracket_count--;
      token.push_back(formula_[cursor]);
      cursor++;
    }
    Rcpp::Rcout << "\nToken: " << token;
    tokens_.push_back(token);
    token.clear();
    cursor++;
  }
  
  tokens_.erase(
    std::remove_if(tokens_.begin(),
                   tokens_.end(),
                   [](std::string const& s) {
                     return s.size() == 0;
                   }),
                   tokens_.end());

  // separate into fixed and random effect components
  int n = tokens_.size();
  for(int i = 0; i<n ; i++){
    if(tokens_[i][0]=='('){
      if(tokens_[i].back()==')'){
        re_.push_back(tokens_[i]);
      } else {
        Rcpp::stop("Error in formula");
      }
    } else {
      fe_.push_back(tokens_[i]);
    }
  }

  
  for(int i =0; i<re_.size(); i++){
    re_order_.push_back(i);
  }

  // // random effects: separate right and left hand sides
  int m = re_.size();
  re_terms_ = re_;

  for(int i = 0; i<m;i++){
    std::stringstream check1(re_[0]);
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

// update this to only capture random effects
inline void glmmr::Formula::formula_validate(){
  //currently only checks if addition inside re term
  int open = 0;
  for(auto ch: formula_){
    if(ch=='(')open++;
    if(ch==')')open--;
    if(ch=='+' && open>0)Rcpp::stop("Addition inside re term not currently supported");
  }
}

#endif
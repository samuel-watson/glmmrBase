#ifndef FORMULA_IPP
#define FORMULA_IPP

inline void glmmr::Formula::tokenise(){
  
  
  // remove -1 and set RM_INT is true if its there
  RM_INT = false;
  const std::regex rmone("-\\s*1");
  if(std::regex_search(formula_,rmone)){
    RM_INT = true;
    std::stringstream result;
    std::regex_replace(std::ostream_iterator<char>(result), formula_.begin(), formula_.end(), rmone, "");
    formula_ = result.str();
  }
  
  // tokenise string - split at + or space except between brackets
  const std::regex re("([\\s\\+]+)(?![^\\(]*\\|)(?![^\\|]*\\))");
  std::sregex_token_iterator it{ formula_.begin(), formula_.end(), re, -1 };
  tokens_ = strvec{ it, {} };
  
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
  
  for(int i = 0; i < fe_.size(); i++){
    fe_[i].erase(std::remove_if(fe_[i].begin(), fe_[i].end(), [](unsigned char x) { return std::isspace(x); }), fe_[i].end());
  }
  
  
  for(int i =0; i<re_.size(); i++){
    re_order_.push_back(i);
  }

  // // random effects: separate right and left hand sides
  int m = re_.size();
  re_terms_ = re_;

  for(int i = 0; i<m;i++){
    std::stringstream check1(re_[0]);
    std::string intermediate;
    re_.erase(re_.begin());
    getline(check1, intermediate, '|');
    z_.push_back(intermediate.substr(1,intermediate.length()-1));
    getline(check1, intermediate, '|');
    re_.push_back(intermediate.substr(0,intermediate.length()-1));
  }
  
  for(int i = 0; i < re_.size(); i++){
    re_[i].erase(std::remove_if(re_[i].begin(), re_[i].end(), [](unsigned char x) { return std::isspace(x); }), re_[i].end());
  }
  
  // // check if multiple values on LHS and split into separate RE terms
  m = z_.size();
  int iter = 0;

  for(int i = 0; i< m; i++){
    if(z_[i].find("+") != std::string::npos){
      std::stringstream check1(z_[iter]);
      std::string intermediate;
      while(getline(check1, intermediate, '+')){
        z_.push_back(intermediate);
        re_.push_back(re_[iter]);
        re_order_.push_back(re_order_[iter]);
      }
      z_.erase(z_.begin()+iter);
      re_.erase(re_.begin()+iter);
      re_order_.erase(re_order_.begin()+iter);
    } else {
      iter++;
    }
  }

  for(int i = 0; i < z_.size(); i++){
    z_[i].erase(std::remove_if(z_[i].begin(), z_[i].end(), [](unsigned char x) { return std::isspace(x); }), z_[i].end());
  }
  
}


#endif
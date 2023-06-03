#ifndef CALCULATOR_IPP
#define CALCULATOR_IPP

inline dblvec glmmr::calculator::calculate(const int i, 
                                           const int j,
                                           int order,
                                           double extraData){
  if(order > 2)Rcpp::stop("Only up to second order derivatives allowed.");
  int idx_iter = 0;
  double a,b,var;
  std::stack<double> stack;
  // for higher order derivatives
  std::vector<std::stack<double> > first_dx;
  std::vector<std::stack<double> > second_dx;
  if(order > 0){
    first_dx.resize(parameter_count);
  }
  if(order == 2){
    second_dx.resize(parameter_count*(parameter_count + 1)/2);
  }
  
  auto addZeroDx = [&] (){
    for(int idx = 0; idx < parameter_count; idx++){
      first_dx[idx].push(0.0);
    }
  };
  
  auto addZeroDx2 = [&] (){
    int index_count = 0;
    for(int idx = 0; idx < parameter_count; idx++){
      for(int jdx = idx; jdx < parameter_count; jdx++){
        second_dx[index_count].push(0.0);
        index_count++;
      }
    }
  };
  
  
  for(int k = 0; k < instructions.size(); k++){
    switch(instructions[k]){
    case 0:
  {
    //push data (i)
    //DEBUG
    
    if(idx_iter >= indexes.size())Rcpp::stop("Index out of range: case 0 idx iter: "+std::to_string(idx_iter)+" versus "+std::to_string(indexes.size()));
    if(indexes[idx_iter] >= data[i].size())Rcpp::stop("Index out of range: case 0 indexes: "+std::to_string(indexes[idx_iter])+" versus "+std::to_string(data[i].size()));
    stack.push(data[i][indexes[idx_iter]]);
    if(order > 0){
      addZeroDx();
    }
    if(order == 2){
      addZeroDx2();
    }
    idx_iter++;
    break;
  }
    case 1:
  {
    // push data (j)
    //DEBUG
    if(idx_iter >= indexes.size())Rcpp::stop("Index out of range: case 0 idx iter: "+std::to_string(idx_iter)+" versus "+std::to_string(indexes.size()));
    if(indexes[idx_iter] >= data[i].size())Rcpp::stop("Index out of range: case 0 indexes: "+std::to_string(indexes[idx_iter])+" versus "+std::to_string(data[j].size()));
    
    stack.push(data[j][indexes[idx_iter]]);
    if(order > 0){
      addZeroDx();
    }
    if(order == 2){
      addZeroDx2();
    }
    idx_iter++;
    break;
  }
    case 2:
  {
    // push parameter
    //DEBUG
    if(idx_iter >= indexes.size())Rcpp::stop("Index out of range: case 0 idx iter: "+std::to_string(idx_iter)+" versus "+std::to_string(indexes.size()));
    if(indexes[idx_iter] >= parameters.size())Rcpp::stop("Index out of range: case 0 indexes: "+std::to_string(indexes[idx_iter])+" versus "+std::to_string(parameters.size()));
    
    stack.push(parameters[indexes[idx_iter]]);
    if(order > 0){
      for(int idx = 0; idx < parameter_count; idx++){
        if(idx == indexes[idx_iter]){
          first_dx[idx].push(1.0);
        } else {
          first_dx[idx].push(0.0);
        }
      }
    }
    if(order == 2){
      addZeroDx2();
    }
    idx_iter++;
    break;
  }
    case 3:
  {
    // add
    if(stack.size()<2)Rcpp::stop("Stack too small (3)");
    a = stack.top();
    stack.pop();
    b = stack.top();
    stack.pop();
    stack.push(a+b);
    if(order > 0){
      for(int idx = 0; idx < parameter_count; idx++){
        a = first_dx[idx].top();
        first_dx[idx].pop();
        b = first_dx[idx].top();
        first_dx[idx].pop();
        first_dx[idx].push(a+b);
      }
    }
    if(order == 2){
      int index_count = 0;
      for(int idx = 0; idx < parameter_count; idx++){
        for(int jdx = idx; jdx < parameter_count; jdx++){
          a = second_dx[index_count].top();
          second_dx[index_count].pop();
          b = second_dx[index_count].top();
          second_dx[index_count].pop();
          second_dx[index_count].push(a+b);
          index_count++;
        }
      }
    }
    break;
  }
    case 4:
  {
    // subtract
    if(stack.size()<2)Rcpp::stop("Stack too small (4)");
    a = stack.top();
    stack.pop();
    b = stack.top();
    stack.pop();
    stack.push(a-b);
    if(order > 0){
      for(int idx = 0; idx < parameter_count; idx++){
        a = first_dx[idx].top();
        first_dx[idx].pop();
        b = first_dx[idx].top();
        first_dx[idx].pop();
        first_dx[idx].push(a-b);
      }
    }
    if(order == 2){
      int index_count = 0;
      for(int idx = 0; idx < parameter_count; idx++){
        for(int jdx = idx; jdx < parameter_count; jdx++){
          a = second_dx[index_count].top();
          second_dx[index_count].pop();
          b = second_dx[index_count].top();
          second_dx[index_count].pop();
          second_dx[index_count].push(a-b);
          index_count++;
        }
      }
    }
    break;
  }
    case 5:
  {
    // multiply
    if(stack.size()<2)Rcpp::stop("Stack too small (5)");
    a = stack.top();
    stack.pop();
    b = stack.top();
    stack.pop();
    stack.push(a*b);
    if(order > 0){
      dblvec a_top_dx;
      dblvec b_top_dx;
      for(int idx = 0; idx < parameter_count; idx++){
        a_top_dx.push_back(first_dx[idx].top());
        first_dx[idx].pop();
        b_top_dx.push_back(first_dx[idx].top());
        first_dx[idx].pop();
        first_dx[idx].push(a*b_top_dx.back() + b*a_top_dx.back());
      }
      if(order == 2){
        int index_count = 0;
        for(int idx = 0; idx < parameter_count; idx++){
          for(int jdx = idx; jdx < parameter_count; jdx++){
            double adx2 = second_dx[index_count].top();
            second_dx[index_count].pop();
            double bdx2 = second_dx[index_count].top();
            second_dx[index_count].pop();
            double result = a*bdx2 + b*adx2 + a_top_dx[idx]*b_top_dx[jdx] + a_top_dx[jdx]*b_top_dx[idx];
            second_dx[index_count].push(result);
            index_count++;
          }
        }
      }
    }
    break;
  }
    case 6:
  {
    //division
    if(stack.size()<2)Rcpp::stop("Stack too small (6)");
    a = stack.top();
    stack.pop();
    b = stack.top();
    if(b == 0)Rcpp::stop("Divide by zero (6)");
    stack.pop();
    stack.push(a/b);
    if(order > 0){
      dblvec a_top_dx;
      dblvec b_top_dx;
      for(int idx = 0; idx < parameter_count; idx++){
        a_top_dx.push_back(first_dx[idx].top());
        first_dx[idx].pop();
        b_top_dx.push_back(first_dx[idx].top());
        first_dx[idx].pop();
        double result = (b*a_top_dx.back() - a*b_top_dx.back())/(b*b);
        first_dx[idx].push(result);
      }
      if(order == 2){
        int index_count = 0;
        for(int idx = 0; idx < parameter_count; idx++){
          for(int jdx = idx; jdx < parameter_count; jdx++){
            double adx2 = second_dx[index_count].top();
            second_dx[index_count].pop();
            double bdx2 = second_dx[index_count].top();
            second_dx[index_count].pop();
            double result = (adx2*b - a_top_dx[idx]*b_top_dx[jdx]- a_top_dx[jdx]*b_top_dx[idx])/(b*b) - (a*bdx2*b - 2*b_top_dx[idx]*b_top_dx[jdx])/(b*b*b);
            second_dx[index_count].push(result);
            index_count++;
          }
        }
      }
    }
    break;
  }
    case 7:
      if(stack.size()==0)Rcpp::stop("Stack too small (7)");
      a = stack.top();
      stack.pop();
      stack.push(sqrt(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = 0.5*pow(a,-0.5)*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = 0.5*pow(a,-0.5)*adx2 - 0.25*a_top_dx[idx]*a_top_dx[jdx]*pow(a,-3/2);
              second_dx[index_count].push(result);
              index_count++;
            }
          }
        }
      }
      break;
    case 8:
      {
        if(stack.size()<2)Rcpp::stop("Stack too small (8)");
        a = stack.top();
        stack.pop();
        b = stack.top();
        stack.pop();
        double out = pow(a,b);
        stack.push(out);
        if(order > 0){
          dblvec a_top_dx;
          dblvec b_top_dx;
          for(int idx = 0; idx < parameter_count; idx++){
            a_top_dx.push_back(first_dx[idx].top());
            first_dx[idx].pop();
            b_top_dx.push_back(first_dx[idx].top());
            first_dx[idx].pop();
            double result = pow(a,b)*b_top_dx.back()*log(a) + pow(a,b-1)*b*a_top_dx.back();
            first_dx[idx].push(result);
          }
          if(order == 2){
            int index_count = 0;
            for(int idx = 0; idx < parameter_count; idx++){
              for(int jdx = idx; jdx < parameter_count; jdx++){
                double adx2 = second_dx[index_count].top();
                second_dx[index_count].pop();
                double bdx2 = second_dx[index_count].top();
                second_dx[index_count].pop();
                double result1 = first_dx[jdx].top()*b_top_dx[idx]*log(a) + stack.top()*(bdx2*log(a) + b_top_dx[idx]*(1/a)*a_top_dx[jdx]);
                double result2 = pow(a,b-1)*b_top_dx[jdx]*log(a) + pow(a,b-2)*(b-1)*a_top_dx[jdx];
                double result3 = result2*b*a_top_dx[idx] + pow(a,b-1)*(b*adx2+b_top_dx[jdx]*a_top_dx[idx]);
                second_dx[index_count].push(result1 + result3);
                index_count++;
              }
            }
          }
        }
        
        break;
      }
    case 9:
      if(stack.size()==0)Rcpp::stop("Stack too small (9)");
      a = stack.top();
      stack.pop();
      stack.push(exp(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = stack.top()*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = stack.top()*(a_top_dx[idx]*a_top_dx[jdx] + adx2);
              second_dx[index_count].push(result);
              index_count++;
            }
          }
        }
      }
      break;
    case 10:
      if(stack.size()==0)Rcpp::stop("Stack too small (10)");
      a = stack.top();
      stack.pop();
      stack.push(-1*a);
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = -1.0*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              second_dx[index_count].push(-1*adx2);
              index_count++;
            }
          }
        }
      }
      break;
    case 11:
      if(stack.size()==0)Rcpp::stop("Stack too small (11)");
      a = stack.top();
      stack.pop();
      b = boost::math::cyl_bessel_k(1,a);
      stack.push(b);
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = -0.5*boost::math::cyl_bessel_k(0,a)-0.5*boost::math::cyl_bessel_k(2,a);
          first_dx[idx].push(result*a_top_dx.back());
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = 0.25*boost::math::cyl_bessel_k(-1,a)+0.5*boost::math::cyl_bessel_k(1,a)+0.25*boost::math::cyl_bessel_k(3,a);
              double result1 = -0.5*boost::math::cyl_bessel_k(0,a)-0.5*boost::math::cyl_bessel_k(2,a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case 12:
      if(stack.size()==0)Rcpp::stop("Stack too small (12)");
      a = stack.top();
      stack.pop();
      stack.push(tgamma(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = stack.top()*boost::math::polygamma(0,a);
          first_dx[idx].push(result*a_top_dx.back());
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = stack.top()*boost::math::polygamma(0,a)*boost::math::polygamma(0,a) + stack.top()*boost::math::polygamma(1,a);
              double result1 = stack.top()*boost::math::polygamma(0,a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case 13:
      if(stack.size()==0)Rcpp::stop("Stack too small (13)");
      a = stack.top();
      stack.pop();
      stack.push(sin(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = cos(a)*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = -1.0*sin(a);
              double result1 = cos(a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case 14:
      if(stack.size()==0)Rcpp::stop("Stack too small (14)");
      a = stack.top();
      stack.pop();
      stack.push(cos(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = -1.0*sin(a)*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = -1.0*cos(a);
              double result1 = -1.0*sin(a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case 15:
      if(stack.size()==0)Rcpp::stop("Stack too small (15)");
      a = stack.top();
      stack.pop();
      b = stack.top();
      stack.pop();
      stack.push(boost::math::cyl_bessel_k(b,a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = -0.5*boost::math::cyl_bessel_k(b-1,a)-0.5*boost::math::cyl_bessel_k(b+1,a);
          first_dx[idx].push(result*a_top_dx.back());
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = 0.25*boost::math::cyl_bessel_k(b-2,a)+0.5*boost::math::cyl_bessel_k(b,a)+0.25*boost::math::cyl_bessel_k(b+2,a);
              double result1 = -0.5*boost::math::cyl_bessel_k(b-1,a)-0.5*boost::math::cyl_bessel_k(b+1,a);
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case 16:
      if(stack.size()==0)Rcpp::stop("Stack too small (16)");
      a = stack.top();
      stack.pop();
      stack.push(log(a));
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = (1/a)*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result = -1.0/(a*a);
              double result1 = 1/a;
              double result2 = result*a_top_dx[idx]*a_top_dx[jdx]+result1*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case 17:
      if(stack.size()==0)Rcpp::stop("Stack too small (17)");
      a = stack.top();
      stack.pop();
      stack.push(a*a);
      if(order > 0){
        dblvec a_top_dx;
        for(int idx = 0; idx < parameter_count; idx++){
          a_top_dx.push_back(first_dx[idx].top());
          first_dx[idx].pop();
          double result = 2*a*a_top_dx.back();
          first_dx[idx].push(result);
        }
        if(order == 2){
          int index_count = 0;
          for(int idx = 0; idx < parameter_count; idx++){
            for(int jdx = idx; jdx < parameter_count; jdx++){
              double adx2 = second_dx[index_count].top();
              second_dx[index_count].pop();
              double result2 = 2*a_top_dx[idx]*a_top_dx[jdx]+2*a*adx2;
              second_dx[index_count].push(result2);
              index_count++;
            }
          }
        }
      }
      break;
    case 18:
      {
        //push data (i)
        stack.push(extraData);
        if(order > 0){
          addZeroDx();
        }
        if(order == 2){
          addZeroDx2();
        }
        break;
      }
    case 20:
      stack.push(10);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 21:
      stack.push(1);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 22:
      stack.push(2);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 23:
      stack.push(3);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 24:
      stack.push(4);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 25:
      stack.push(5);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 26:
      stack.push(6);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 27:
      stack.push(7);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 28:
      stack.push(8);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 29:
      stack.push(9);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 30:
      stack.push(M_PI);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 31:
      stack.push(0.3275911);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 32:
      stack.push(0.254829592);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 33:
      stack.push(-0.284496736);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 34:
      stack.push(1.421413741);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 35:
      stack.push(-1.453152027);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 36:
      stack.push(1.061405429);
      if(order > 0){
        addZeroDx();
      }
      if(order == 2){
        addZeroDx2();
      }
      break;
    case 40:
      {
        //log factorial approximation
        if(stack.size()==0)Rcpp::stop("Stack too small (40)");
        a = stack.top();
        stack.pop();
        if(a == 0){
          stack.push(0);
        } else {
          double result = a*log(a) - a + log(a*(1+4*a*(1+2*a)))/6 + log(3.141593)/2;
          stack.push(result);
        }
        if(order > 0){
          addZeroDx();
        }
        if(order == 2){
          addZeroDx2();
        }
        break;
      }
    case 41:
      {
        //push var par
        stack.push(var_par);
        if(order > 0){
          addZeroDx();
        }
        if(order == 2){
          addZeroDx2();
        }
        break;
      }
      
    }
    
    if(stack.size() == 0)Rcpp::stop("Error stack empty!");
  }
  
  dblvec result;
  result.push_back(stack.top());
  if(order > 0){
    for(int idx = 0; idx < parameter_count; idx++){
      if(first_dx[idx].size()==0)Rcpp::stop("Error derivative stack empty");
      result.push_back(first_dx[idx].top());
    }
  }
  if(order == 2){
    int index_count = 0;
    for(int idx = 0; idx < parameter_count; idx++){
      for(int jdx = idx; jdx < parameter_count; jdx++){
        if(second_dx[index_count].size()==0)Rcpp::stop("Error second derivative stack empty");
        result.push_back(second_dx[index_count].top());
        index_count++;
      }
    }
  }
  
  return result;
}

#endif
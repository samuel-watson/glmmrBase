#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "general.h"

namespace glmmr {

inline intvec interpret_re(const std::string& fn,
                                  const intvec& A){
  intvec B;
  switch(string_to_case.at(fn)){
  case 1:
    B = {2,2,5};
    break;
  case 2:
    B.insert(B.end(), A.begin(), A.end());
    B.push_back(2);
    B.push_back(8);
    break;
  case 3:
     {
      const intvec C = {6,10,9};
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C.begin(), C.end());
      break;
     }
  case 4:
    {
       const intvec C = {6,10,9,2,2,5,5};
       B.push_back(2);
       B.insert(B.end(), A.begin(), A.end());
       B.insert(B.end(), C.begin(), C.end());
       break;
    }
  case 5:
    {
      const intvec C1 = {2,2,5};
      const intvec C2 = {5,6,10,9};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      break;
    }
  case 6:
    {
      const intvec C1 = {2,2,5};
      const intvec C2 = {5,6,10,9,2,2,5,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      break;
    }
  case 7:
    {
      const intvec C = {6,11};
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C.begin(), C.end());
      break;
    }
  case 8:
    {
      const intvec C1 = {2,12,2,21,4,22,8,6,22,2,5,7,2};
      const intvec C2 = {6,5,2,8,5,2,22,2,5,7,2};
      const intvec C3 = {6,5,15,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      break;
    }
  case 9:
    {
      const intvec C = {21,4,8,5};
      B.push_back(2);
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C.begin(), C.end());
      break;
    }
  case 10:
    {
      const intvec C1 = {2,2,21,3};
      const intvec C2 = {5,21,3,5,2,21,3};
      const intvec C3 = {21,4,8,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      break;
    }
  case 11:
    {
      const intvec C1 = {2,21};
      const intvec C2 = {22,2,3,5,3,23,21,4,21,2,22,3,2,22,3,5,4,5};
      const intvec C3 = {5,5,3,5,2,22,3};
      const intvec C4 = {21,4,8,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C4.begin(), C4.end());
      break;
    }
  case 12:
    {
      const intvec C1 = {2,2,12,2,21,4,22,8,6,5,2};
      const intvec C2 = {8,5,2};
      const intvec C3 = {15,5};
      const intvec C4 = {5,20,20,5,20,27,3,3,4,5,22,20,21,3,6};
      const intvec C5 = {5,3,21,3,5};
      const intvec C6 = {21,4,5};
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C4.begin(), C4.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C5.begin(), C5.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C6.begin(), C6.end());
      break;
    }
  case 13:
    {
      const intvec C1 = {8,21,4,23,10,8,2,5};
      const intvec C2 = {30,5,14};
      const intvec C3 = {21,4,5};
      const intvec C4 = {30,13,30,21,6,5,3,5};
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      B.insert(B.end(), A.begin(), A.end());
      break;
    }
  case 14:
    {
      const intvec C1 = {8,10,9,2,22,30};
      const intvec C2 = {5,5,22,30};
      const intvec C3 = {5,5,13,6};
      const intvec C4 = {21,4,5,22,30};
      const intvec C5 = {5,5,22,30};
      const intvec C6 = {5,5,14,21,4,6,30,21,6,5,3,5};
      B.push_back(2);
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C1.begin(), C1.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C2.begin(), C2.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C3.begin(), C3.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C4.begin(), C4.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C5.begin(), C5.end());
      B.insert(B.end(), A.begin(), A.end());
      B.insert(B.end(), C6.begin(), C6.end());
      break;
    }
  }
  return B;
}

//add in the indexes for each function
inline intvec interpret_re_par(const std::string& fn,
                               const intvec& col_idx,
                               const intvec& par_idx){
  intvec B;
  
  
  auto addA = [&] (){
    for(int i = 0; i<col_idx.size();i++){
      B.push_back(col_idx[i]);
      B.push_back(col_idx[i]);
      B.push_back(col_idx[i]);
      B.push_back(col_idx[i]);
    }
  };
  
  auto addPar2 = [&] (int i){
    B.push_back(par_idx[i]);
    B.push_back(par_idx[i]);
  };
  
  
  switch(string_to_case.at(fn)){
  case 1:
    addPar2(0);
    break;
  case 2: 
    addA();
    B.push_back(par_idx[0]);
    break;
  case 3: case 7:
    B.push_back(par_idx[0]);
    addA();
    break;
  case 4:
    B.push_back(par_idx[1]);
    addA();
    addPar2(0);
    break;
  case 5:
    addPar2(0);
    addA();
    addA();
    break;
  case 6:
    addPar2(1);
    addA();
    addA();
    addPar2(0);
    break;
  case 8:
    addPar2(0);
    addPar2(0);
    B.push_back(par_idx[1]);
    addA();
    addPar2(0);
    B.push_back(par_idx[0]);
    B.push_back(par_idx[1]);
    addA();
    break;
  case 9:
    B.push_back(par_idx[0]);
    B.push_back(par_idx[1]);
    addA();
    break;
  case 10:
    addPar2(0);
    B.push_back(par_idx[1]);
    addA();
    break;
  case 11:
    B.push_back(par_idx[0]);
    addA();
    addPar2(1);
    addA();
    addA();
    B.push_back(par_idx[1]);
    addA();
    break;
  case 12:
    B.push_back(par_idx[0]);
    addPar2(1);
    addPar2(0);
    addA();
    addA();
    addA();
    addA();
    addA();
    addA();
    break;
  case 13:
    B.push_back(par_idx[1]);
    addA();
    B.push_back(par_idx[0]);
    addA();
    addA();
    addA();
    break;
  case 14:
    B.push_back(par_idx[1]);
    addA();
    B.push_back(par_idx[0]);
    addA();
    addA();
    addA();
    addA();
    addA();
    break;
  }
  return B;
}

}

#endif
#pragma once

#include "general.h"
#include "calculator.hpp"

namespace glmmr {

inline std::vector<Instruction> interpret_re(const std::string& fn){
  
  using instructs = std::vector<Instruction>;
  using enum Instruction;
  // instruction lists for calculation of covariance functions
  instructs B;
  switch(string_to_case.at(fn)){
  case 1:
    B = {PushParameter}; 
    break;
  case 2:
    B.push_back(PushParameter);
    B.push_back(PushCovData);
    B.push_back(PushParameter);
    B.push_back(Power);
    B.push_back(Multiply);
    break;
  case 3:
     {
      const instructs C = {Divide,Negate,Exp};
      B.push_back(PushParameter);
      B.push_back(PushCovData);
      B.insert(B.end(), C.begin(), C.end());
      break;
     }
  case 4:
    {
       const instructs C = {Divide,Negate,Exp,PushParameter,Multiply};  //var par here
       B.push_back(PushParameter);
       B.push_back(PushCovData);
       B.insert(B.end(), C.begin(), C.end());
       break;
    }
  case 5:
    {
      const instructs C1 = {PushParameter,PushParameter,Multiply};
      const instructs C2 = {Multiply,Divide,Negate,Exp};
      B.insert(B.end(), C1.begin(), C1.end());
      B.push_back(PushCovData);
      B.push_back(PushCovData);
      B.insert(B.end(), C2.begin(), C2.end());
      break;
    }
  case 6:
    {
      const instructs C1 = {PushParameter,PushParameter,Multiply};
      const instructs C2 = {Multiply,Divide,Negate,Exp,PushParameter,Multiply};
      B.insert(B.end(), C1.begin(), C1.end());
      B.push_back(PushCovData);
      B.push_back(PushCovData);
      B.insert(B.end(), C2.begin(), C2.end());
      break;
    }
  case 7:
    {
      const instructs C = {Divide,Bessel};
      B.push_back(PushParameter);
      B.push_back(PushCovData);
      B.insert(B.end(), C.begin(), C.end());
      break;
    }
  case 8:
    {
      const instructs C1 = {PushParameter,Gamma,PushParameter,Int1,Subtract,Int2,Power,Divide,
                          Int2,PushParameter,Multiply,Sqrt,PushParameter};
      const instructs C2 = {Divide,Multiply,PushParameter,Power,Multiply,PushParameter,Int2,PushParameter,
                         Multiply,Sqrt,PushParameter};
      const instructs C3 = {Divide,Multiply,BesselK,Multiply};
      B.insert(B.end(), C1.begin(), C1.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C2.begin(), C2.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C3.begin(), C3.end());
      break;
    }
  case 9:
    {
      const instructs C = {Int1,Subtract,Power,Multiply};
      B.push_back(PushParameter);
      B.push_back(PushParameter);
      B.push_back(PushCovData);
      B.insert(B.end(), C.begin(), C.end());
      break;
    }
  case 10:
    {
      const instructs C1 = {PushParameter,PushParameter,Int1,Add};
      const instructs C2 = {Multiply,Int1,Add,Multiply,PushParameter,Int1,Add};
      const instructs C3 = {Int1,Subtract,Power,Multiply};
      B.insert(B.end(), C1.begin(), C1.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C2.begin(), C2.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C3.begin(), C3.end());
      break;
    }
  case 11:
    {
      const instructs C1 = {PushParameter,Int1};
      const instructs C2 = {Int2,PushParameter,Add,Multiply,Add,Int3,Int1,Subtract,Int1,
                         PushParameter,Int2,Add,PushParameter,Int2,Add,Multiply,Subtract,Multiply};
      const instructs C3 = {Multiply,Multiply,Add,Multiply,PushParameter,Int2,Add};
      const instructs C4 = {Int1,Subtract,Power,Multiply};
      B.insert(B.end(), C1.begin(), C1.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C2.begin(), C2.end());
      B.push_back(PushCovData);
      B.push_back(PushCovData);
      B.insert(B.end(), C3.begin(), C3.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C4.begin(), C4.end());
      break;
    }
  case 12:
    {
      const instructs C1 = {PushParameter,PushParameter,Gamma,PushParameter,Int1,Subtract,Int2,Power,Divide,
                            Multiply,PushParameter};
      const instructs C2 = {Power,Multiply,PushParameter};
      const instructs C3 = {BesselK,Multiply};
      const instructs C4 = {Multiply,Int10,Int10,Multiply,Int10,Int7,Add,Add,Subtract,
                            Multiply,Int2,Int10,Int1,Add,Divide};
      const instructs C5 = {Multiply,Add,Int1,Add,Multiply};
      const instructs C6 = {Int1,Subtract,Multiply};
      B.insert(B.end(), C1.begin(), C1.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C2.begin(), C2.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C3.begin(), C3.end());
      B.push_back(PushCovData);
      B.push_back(PushCovData);
      B.insert(B.end(), C4.begin(), C4.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C5.begin(), C5.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C6.begin(), C6.end());
      break;
    }
  case 13:
    {
      const instructs C1 = {Power,Int1,Subtract,Int3,Negate,Power,PushParameter,Multiply};
      const instructs C2 = {Pi,Multiply,Cos};
      const instructs C3 = {Int1,Subtract,Multiply};
      const instructs C4 = {Pi,Sin,Pi,Int1,Divide,Multiply,Add,Multiply};
      B.push_back(PushParameter);
      B.push_back(PushCovData);
      B.insert(B.end(), C1.begin(), C1.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C2.begin(), C2.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C3.begin(), C3.end());
      B.push_back(PushCovData);
      break;
    }
  case 14:
    {
      const instructs C1 = {Power,Negate,Exp,PushParameter,Int2,Pi};
      const instructs C2 = {Multiply,Multiply,Int2,Pi};
      const instructs C3 = {Multiply,Multiply,Sin,Divide};
      const instructs C4 = {Int1,Subtract,Multiply,Int2,Pi};
      const instructs C6 = {Multiply,Multiply,Cos,Int1,Subtract,Divide,Pi,Int1,Divide,Multiply,Add,Multiply};
      B.push_back(PushParameter);
      B.push_back(PushCovData);
      B.insert(B.end(), C1.begin(), C1.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C2.begin(), C2.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C3.begin(), C3.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C4.begin(), C4.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C2.begin(), C2.end());
      B.push_back(PushCovData);
      B.insert(B.end(), C6.begin(), C6.end());
      break;
    }
  case 15: case 16:
    B.push_back(PushCovData);
    B.push_back(PushParameter);
    B.push_back(Power);
    break;
  case 17:
    B.push_back(PushCovData);
    break;
  }
  return B;
}

//add in the indexes for each function
inline intvec interpret_re_par(const std::string& fn,
                               const int col_idx,
                               const intvec& par_idx){
  intvec B;

  auto addA = [&] (){
    B.push_back(col_idx);
  };
  
  auto addPar2 = [&] (int i){
    B.push_back(par_idx[i]);
    B.push_back(par_idx[i]);
  };
  
  
  switch(string_to_case.at(fn)){
  case 1:
    B.push_back(par_idx[0]);
    break;
  case 2: 
    B.push_back(par_idx[0]);
    addA();
    B.push_back(par_idx[1]);
    break;
  case 3: case 7:
    B.push_back(par_idx[0]);
    addA();
    break;
  case 4:
    B.push_back(par_idx[1]);
    addA();
    B.push_back(par_idx[0]);
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
    B.push_back(par_idx[0]);
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
  case 15: case 16:
    addA();
    B.push_back(par_idx[0]);
    break;
  case 17:
    addA();
    break;
  }
  return B;
}

inline void re_linear_predictor(glmmr::calculator& calc,
                                const int& Q){
  using instructs = std::vector<Instruction>;
  using enum Instruction;
  
  instructs re_instruct;
  instructs re_seq = {PushData,PushParameter,Multiply,Add};
  for(int i = 0; i < Q; i++){
    re_instruct.insert(re_instruct.end(),re_seq.begin(),re_seq.end());
    calc.parameter_names.push_back("v_"+std::to_string(i));
    calc.indexes.push_back(i+calc.data_count);
    calc.indexes.push_back(i+calc.data_count);
  }
  calc.parameter_count += Q;
  calc.instructions.insert(calc.instructions.end(),re_instruct.begin(),re_instruct.end());
  calc.data_count += Q;
}

inline void linear_predictor_to_link(glmmr::calculator& calc,
                                     const str& link){
  using instructs = std::vector<Instruction>;
  using enum Instruction;
  instructs out;
  instructs addzu = {PushExtraData,Add};
  calc.instructions.insert(calc.instructions.end(),addzu.begin(),addzu.end());
  
  switch (link_to_case.at(link)) {
  case 1:
    {
      out = calc.instructions;
      instructs logit_instruct = {Negate,Exp,Int1,Add,Int1,Divide};
      out.insert(out.end(),logit_instruct.begin(),logit_instruct.end());
      break;
    }
  case 2:
    {
      out = calc.instructions;
      out.push_back(Exp);
      break;
    }
  case 3:
    {
      // probit is a pain because of the error function!
      // this uses Abramowitz and Stegun approximation.
      instructs iStar = {Int2,Sqrt};
      iStar.insert(iStar.end(),calc.instructions.begin(),calc.instructions.end());
      iStar.push_back(Divide);
      instructs M = iStar;
      instructs MStar = {Constant1,Multiply,Int1,Add,Int1,Divide};
      M.insert(M.end(),MStar.begin(),MStar.end());
      instructs Ltail = {Power,Multiply,Add};
      instructs L1 = {Constant2};
      L1.insert(L1.end(),M.begin(),M.end());
      L1.push_back(Multiply);
      instructs L2 = {Constant3,Int2};
      L1.insert(L1.end(),L2.begin(),L2.end());
      L1.insert(L1.end(),M.begin(),M.end());
      L1.insert(L1.end(),Ltail.begin(),Ltail.end());
      L2 = {Constant4,Int3};
      L1.insert(L1.end(),L2.begin(),L2.end());
      L1.insert(L1.end(),M.begin(),M.end());
      L1.insert(L1.end(),Ltail.begin(),Ltail.end());
      L2 = {Constant5,Int4};
      L1.insert(L1.end(),L2.begin(),L2.end());
      L1.insert(L1.end(),M.begin(),M.end());
      L1.insert(L1.end(),Ltail.begin(),Ltail.end());
      L2 = {Constant6,Int5};
      L1.insert(L1.end(),L2.begin(),L2.end());
      L1.insert(L1.end(),M.begin(),M.end());
      L1.push_back(Power);
      L1.push_back(Multiply);
      instructs L3 = {Int2};
      L3.insert(L3.end(),iStar.begin(),iStar.end());
      instructs L4 = {Divide,Negate,Power};
      L3.insert(L3.end(),L4.begin(),L4.end());
      out = L1;
      out.insert(out.end(),L3.begin(),L3.end());
      out.push_back(Multiply);
      out.push_back(Int1);
      out.push_back(Subtract);
      break;
    }
  case 4:
    {
      out = calc.instructions;
      break;
    }
  case 5:
    {
      out = calc.instructions;
      instructs inverse_instruct = {Int1,Divide};
      out.insert(out.end(),inverse_instruct.begin(),inverse_instruct.end());
      break;
    }
  }
  
  calc.instructions = out;
}

inline void link_to_likelihood(glmmr::calculator& calc,
                               const str& family){
  using instructs = std::vector<Instruction>;
  using enum Instruction;
  instructs out;
  intvec idx;
  
  switch (family_to_case.at(family)){
    case 1:
      {
        instructs gaus_instruct = {PushY,Subtract,Square,Divide,Int2,Int1,Divide,Multiply,Int2,Pi,Multiply,
                                  Log,Int2,Int1,Divide,Multiply,Add,PushVariance,Log,Int2,Int1,Divide,
                                  Multiply,Add,Negate};
        out.push_back(PushVariance);
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),gaus_instruct.begin(),gaus_instruct.end());
        break;
      }
    case 2:
      {
        instructs binom_instruct = {Log,Multiply,PushY,Int1,Subtract};
        instructs binom_instruct2 = {Int1,Subtract,Log,Multiply,Add};
        out.push_back(PushY);
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),binom_instruct.begin(),binom_instruct.end());
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),binom_instruct2.begin(),binom_instruct2.end());
        break;
      }
    case 3:
      {
        instructs poisson_instruct = {PushY,LogFactorialApprox,Add,PushY};
        instructs poisson_instruct2 = {Log,Multiply,Subtract};
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),poisson_instruct.begin(),poisson_instruct.end());
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),poisson_instruct2.begin(),poisson_instruct2.end());
        break;
      }
    case 4:
      {
        instructs gamma_instruct = {PushVariance,PushY,Multiply,Divide};
        // instructs gamma_instruct2 = {PushVariance,PushY,Multiply,Divide,Log,PushVariance,Multiply,Subtract,
        //                           PushVariance,Gamma,PushY,Multiply,Int1,Divide,PushData,Divide,Add};
        instructs gamma_instruct2 = {Log,PushVariance,Log,Subtract,PushVariance,Multiply,PushY,Log,Int1,PushVariance,
                                     Subtract,Multiply,Add,Subtract};
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),gamma_instruct.begin(),gamma_instruct.end());
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),gamma_instruct2.begin(),gamma_instruct2.end());
        break;
      }
    case 5:
      {
        instructs beta_instruct = {PushVariance,Subtract,PushY,Log,Multiply,Int1};
        instructs beta_instruct2 = {Int1,Subtract,PushVariance,Multiply,Subtract,PushY,Int1,Subtract,Log,
                                    Multiply,Add};
        instructs beta_instruct3 = {PushVariance,Multiply,Gamma,Log,Negate,Add};
        instructs beta_instruct4 = {Int1,Subtract,PushVariance,Multiply,Gamma,Log,Negate,Add,PushVariance,
                                    Gamma,Log,Add};
        out.push_back(Int1);
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),beta_instruct.begin(),beta_instruct.end());
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),beta_instruct2.begin(),beta_instruct2.end());
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),beta_instruct3.begin(),beta_instruct3.end());
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),beta_instruct4.begin(),beta_instruct4.end());
        break;
      }
    case 6:
      {
        instructs binom_instruct = {PushY,LogFactorialApprox,PushY,PushVariance,Subtract,Add,PushVariance,
                                 LogFactorialApprox,Add};
        instructs binom_instruct2 = {Log,PushY,Multiply,Add};
        instructs binom_instruct3 = {Int1,Subtract,Log,PushY,PushVariance,Subtract,Multiply,Add};
        out.insert(out.end(),binom_instruct.begin(),binom_instruct.end());
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),binom_instruct2.begin(),binom_instruct2.end());
        out.insert(out.end(),calc.instructions.begin(),calc.instructions.end());
        idx.insert(idx.end(),calc.indexes.begin(),calc.indexes.end());
        out.insert(out.end(),binom_instruct3.begin(),binom_instruct3.end());
      }
  }
  calc.instructions = out;
  calc.indexes = idx;
}

}

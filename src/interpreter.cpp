#include <glmmr/interpreter.h>

std::vector<Do> glmmr::interpret_re(const CovFunc& fn){
  using instructs = std::vector<Do>;
  instructs B;
  switch(fn){
  case CovFunc::gr:
    B = {Do::PushParameter}; 
    break;
  case CovFunc::ar:
    B.push_back(Do::PushParameter);
    B.push_back(Do::PushCovData);
    B.push_back(Do::PushParameter);
    B.push_back(Do::Power);
    B.push_back(Do::Multiply);
    break;
  case CovFunc::fexp0:
  {
    const instructs C = {Do::Divide,Do::Negate,Do::Exp};
    B.push_back(Do::PushParameter);
    B.push_back(Do::PushCovData);
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::fexp:
  {
    const instructs C = {Do::Divide,Do::Negate,Do::Exp,Do::PushParameter,Do::Multiply};  //var par here
    B.push_back(Do::PushParameter);
    B.push_back(Do::PushCovData);
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::sqexp0:
  {
    const instructs C1 = {Do::PushParameter,Do::Square,Do::PushCovData,
                          Do::Square,Do::Divide,Do::Negate,Do::Exp};
    B.insert(B.end(), C1.begin(), C1.end());
    break;
  }
  case CovFunc::sqexp:
  {
    const instructs C1 = {Do::PushParameter,Do::Square};
    const instructs C2 = {Do::PushCovData,Do::Square,Do::Divide,Do::Negate,Do::Exp,
                          Do::PushParameter,Do::Multiply};
    B.insert(B.end(), C1.begin(), C1.end());
    B.insert(B.end(), C2.begin(), C2.end());
    break;
  }
  case CovFunc::bessel:
  {
    const instructs C = {Do::Divide,Do::Bessel};
    B.push_back(Do::PushParameter);
    B.push_back(Do::PushCovData);
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::matern:
  {
    const instructs C1 = {Do::PushParameter,Do::Gamma,Do::PushParameter,Do::Int1,
                          Do::Subtract,Do::Int2,Do::Power,Do::Divide,
                          Do::Int2,Do::PushParameter,Do::Multiply,Do::Sqrt,Do::PushParameter};
    const instructs C2 = {Do::Divide,Do::Multiply,Do::PushParameter,Do::Power,Do::Multiply,
                          Do::PushParameter,Do::Int2,Do::PushParameter,
                          Do::Multiply,Do::Sqrt,Do::PushParameter};
    const instructs C3 = {Do::Divide,Do::Multiply,Do::BesselK,Do::Multiply};
    B.insert(B.end(), C1.begin(), C1.end());
    B.push_back(Do::PushCovData);
    B.insert(B.end(), C2.begin(), C2.end());
    B.push_back(Do::PushCovData);
    B.insert(B.end(), C3.begin(), C3.end());
    break;
  }
  case CovFunc::truncpow2:
  {
    const instructs C = {Do::PushParameter,Do::Int2,Do::PushParameter,Do::PushCovData,Do::Power,Do::Int1,Do::Subtract,Do::Power,Do::Multiply};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::truncpow3:
  {
    const instructs C = {Do::PushParameter,Do::Int3,Do::PushParameter,Do::PushCovData,Do::Power,Do::Int1,Do::Subtract,Do::Power,Do::Multiply};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::truncpow4:
  {
    const instructs C = {Do::PushParameter,Do::Int4,Do::PushParameter,Do::PushCovData,Do::Power,Do::Int1,Do::Subtract,Do::Power,Do::Multiply};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::cauchy3:
  {
    const instructs C = {Do::PushParameter,Do::Int3,Do::Negate,Do::PushParameter,Do::PushCovData,Do::Power,Do::Int1,Do::Add,Do::Power,Do::Multiply};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::cauchy:
  {
    const instructs C = {Do::PushParameter,Do::PushParameter,Do::PushParameter,Do::Divide,Do::Negate,Do::PushParameter,Do::PushCovData,Do::Power,
                         Do::Int1,Do::Add,Do::Power,Do::Multiply};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::truncpow20:
  {
    const instructs C = {Do::Int2,Do::PushParameter,Do::PushCovData,Do::Power,Do::Int1,Do::Subtract,Do::Power};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::truncpow30:
  {
    const instructs C = {Do::Int3,Do::PushParameter,Do::PushCovData,Do::Power,Do::Int1,Do::Subtract,Do::Power};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::truncpow40:
  {
    const instructs C = {Do::Int4,Do::PushParameter,Do::PushCovData,Do::Power,Do::Int1,Do::Subtract,Do::Power};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::cauchy30:
  {
    const instructs C = {Do::Int3,Do::Negate,Do::PushParameter,Do::PushCovData,Do::Power,Do::Int1,Do::Add,Do::Power};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::cauchy0:
  {
    const instructs C = {Do::PushParameter,Do::PushParameter,Do::Divide,Do::Negate,Do::PushParameter,Do::PushCovData,Do::Power,
                         Do::Int1,Do::Add,Do::Power};
    B.insert(B.end(), C.begin(), C.end());
    break;
  }
  case CovFunc::ar1: case CovFunc::ar0:
    B.push_back(Do::PushCovData);
    B.push_back(Do::PushParameter);
    B.push_back(Do::Power);
    break;
  case CovFunc::dist:
    B.push_back(Do::PushCovData);
    break;
  }
  return B;
}

//add in the indexes for each function
intvec glmmr::interpret_re_par(const CovFunc& fn,
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
  
  
  switch(fn){
  case CovFunc::gr:
    B.push_back(par_idx[0]);
    break;
  case CovFunc::ar: 
    B.push_back(par_idx[0]);
    addA();
    B.push_back(par_idx[1]);
    break;
  case CovFunc::fexp0: case CovFunc::bessel: case CovFunc::sqexp0:
    B.push_back(par_idx[0]);
    addA();
    break;
  case CovFunc::fexp: case CovFunc::sqexp:
    B.push_back(par_idx[1]);
    addA();
    B.push_back(par_idx[0]);
    break;
  case CovFunc::matern:
    addPar2(0);
    addPar2(0);
    B.push_back(par_idx[1]);
    addA();
    addPar2(0);
    B.push_back(par_idx[0]);
    B.push_back(par_idx[1]);
    addA();
    break;
  case CovFunc::truncpow2: case CovFunc::truncpow3: case CovFunc::truncpow4: case CovFunc::cauchy3:
    B.push_back(par_idx[0]);
    B.push_back(par_idx[1]);
    addA();
    break;
  case CovFunc::cauchy:
    B.push_back(par_idx[0]);
    B.push_back(par_idx[1]);
    B.push_back(par_idx[2]);
    B.push_back(par_idx[1]);
    addA();
    break;
  case CovFunc::truncpow20: case CovFunc::truncpow30: case CovFunc::truncpow40: case CovFunc::cauchy30:
    B.push_back(par_idx[0]);
    addA();
    break;
  case CovFunc::cauchy0:
    B.push_back(par_idx[0]);
    B.push_back(par_idx[1]);
    B.push_back(par_idx[0]);
    addA();
    break;
  case CovFunc::ar1: case CovFunc::ar0:
    addA();
    B.push_back(par_idx[0]);
    break;
  case CovFunc::dist:
    addA();
    break;
  }
  return B;
}

void glmmr::re_linear_predictor(glmmr::calculator& calc,
                                const int Q){
  using instructs = std::vector<Do>;
  
  instructs re_seq = {Do::PushData,Do::PushParameter,Do::Multiply,Do::Add};
  for(int i = 0; i < Q; i++){
    calc.instructions.insert(calc.instructions.end(),re_seq.begin(),re_seq.end());
    calc.parameter_names.push_back("v_"+std::to_string(i));
    calc.data_names.push_back("z_"+std::to_string(i));
    calc.indexes.push_back(calc.data_count);
    calc.indexes.push_back(calc.parameter_count);
    calc.parameter_count++;
    calc.data_count++;
  }
}

void glmmr::re_log_likelihood(glmmr::calculator& calc,
                              const int Q){
  using instructs = std::vector<Do>;
  instructs re_seq = {Do::PushParameter,Do::Square,Do::Add};
  for(int i = 0; i < Q; i++){
    calc.instructions.insert(calc.instructions.end(),re_seq.begin(),re_seq.end());
    auto uidx = std::find(calc.parameter_names.begin(),calc.parameter_names.end(),"v_"+std::to_string(i));
    if(uidx == calc.parameter_names.end()){
      throw std::runtime_error("Error finding name of random effect in calculator");
    } else {
      int idx_to_add = uidx - calc.parameter_names.begin();
      calc.indexes.push_back(idx_to_add);
    }
  }
}

void glmmr::linear_predictor_to_link(glmmr::calculator& calc,
                                     const Link link){
  using instructs = std::vector<Do>;
  instructs out;
  instructs addzu = {Do::PushExtraData,Do::Add};
  calc.instructions.insert(calc.instructions.end(),addzu.begin(),addzu.end());
  
  switch (link) {
  case Link::logit:
  {
    out = calc.instructions;
    instructs logit_instruct = {Do::Negate,Do::Exp,Do::Int1,Do::Add,Do::Int1,Do::Divide};
    out.insert(out.end(),logit_instruct.begin(),logit_instruct.end());
    break;
  }
  case Link::loglink:
  {
    out = calc.instructions;
    out.push_back(Do::Exp);
    break;
  }
  case Link::probit:
  {
    out.push_back(Do::SqrtTwo);
    out = calc.instructions;
    instructs probit_instruct = {Do::Divide, Do::ErrorFunc, Do::Int1, Do::Add, Do::Half, Do::Multiply};
    out.insert(out.end(),probit_instruct.begin(),probit_instruct.end());
    break;
  }
  case Link::identity:
  {
    out = calc.instructions;
    break;
  }
  case Link::inverse:
  {
    out = calc.instructions;
    instructs inverse_instruct = {Do::Int1,Do::Divide};
    out.insert(out.end(),inverse_instruct.begin(),inverse_instruct.end());
    break;
  }
  }
  
  calc.instructions = out;
}

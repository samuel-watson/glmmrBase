#include <glmmr/algo.h>

bool glmmr::SigmaBlock::operator==(const intvec& x){
  bool element_is_in = false;
  for(auto i : x){
    auto it = std::find(Dblocks.begin(),Dblocks.end(),i);
    if(it != Dblocks.end()){
      element_is_in = true;
      break;
    } 
  }
  return element_is_in;
}

void glmmr::SigmaBlock::add(const intvec& x){
  intvec xout;
  bool element_is_in = false;
  for(auto i : x){
    auto it = std::find(Dblocks.begin(),Dblocks.end(),i);
    if(it != Dblocks.end()){
      element_is_in = true;
    } else {
      xout.push_back(i); 
    }
  }
  if(element_is_in){
    Dblocks.insert(Dblocks.end(),xout.begin(),xout.end());
    std::sort(Dblocks.begin(),Dblocks.end());
  }
}

void glmmr::SigmaBlock::merge(const SigmaBlock& x){
  RowIndexes.insert(RowIndexes.end(),x.RowIndexes.begin(),x.RowIndexes.end());
  std::sort(RowIndexes.begin(), RowIndexes.end() );
  RowIndexes.erase( std::unique( RowIndexes.begin(), RowIndexes.end() ), RowIndexes.end() );
  Dblocks.insert(Dblocks.end(),x.Dblocks.begin(),x.Dblocks.end());
  std::sort(Dblocks.begin(), Dblocks.end() );
  Dblocks.erase( std::unique( Dblocks.begin(), Dblocks.end() ), Dblocks.end() );
}

void glmmr::SigmaBlock::add_row(int i){
  RowIndexes.push_back(i);
}

intvec glmmr::SigmaBlock::rows(){
  return RowIndexes;
}


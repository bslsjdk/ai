#include "ornith35_tokenizer.h"
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <cstring>

namespace {
static bool u32(std::ifstream &f,uint32_t &v){
 unsigned char b[4]; if(!f.read((char*)b,4)) return false;
 v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24); return true;
}
static bool str(std::ifstream &f,std::string &s){
 uint32_t n; if(!u32(f,n)||n>1000000) return false; s.resize(n);
 return !n || (bool)f.read(s.data(),n);
}
}
bool ornith35_tokenizer_load(const std::string &path,Ornith35Tokenizer &out,std::string &error){
 std::ifstream f(path,std::ios::binary);
 if(!f){error="tokenizer_open_failed";return false;}
 char m[4]; if(!f.read(m,4)||std::string(m,4)!="OTOK"){error="bad_tokenizer_magic";return false;}
 uint32_t ver,n; if(!u32(f,ver)||ver!=1||!u32(f,n)||n>300000){error="bad_tokenizer_header";return false;}
 out.tokens.resize(n); out.scores.resize(n);
 for(uint32_t i=0;i<n;i++){
   if(!str(f,out.tokens[i])){error="bad_tokenizer_token";return false;}
   uint32_t bits; if(!u32(f,bits)){error="bad_tokenizer_score";return false;}
   float x; std::memcpy(&x,&bits,4); out.scores[i]=x;
 }
 out.loaded=true; return true;
}
bool ornith35_tokenizer_encode(const Ornith35Tokenizer &tok,const std::string &text,
                               std::vector<int32_t>&ids,std::string&error){
 if(!tok.loaded){error="tokenizer_not_loaded";return false;}
 ids.clear();
 for(unsigned char c: text){
   std::string s(1,(char)c);
   auto it=std::find(tok.tokens.begin(),tok.tokens.end(),s);
   if(it==tok.tokens.end()){error="token_not_found";return false;}
   ids.push_back((int32_t)(it-tok.tokens.begin()));
 }
 return true;
}

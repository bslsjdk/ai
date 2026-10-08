#include "mlx_safetensors.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace {
bool is_ws(char c) { return c==' ' || c=='\\t' || c=='\\r' || c=='\\n'; }

void skip_ws(const std::string &s, size_t &p) { while (p<s.size() && is_ws(s[p])) ++p; }

bool json_string(const std::string &s, size_t &p, std::string &out) {
    skip_ws(s,p);
    if (p>=s.size() || s[p]!='"') return false;
    ++p; out.clear();
    while (p<s.size()) {
        char c=s[p++];
        if (c=='"') return true;
        if (c=='\\\\') {
            if (p>=s.size()) return false;
            char e=s[p++];
            switch(e) { case '"': case '\\\\': case '/': out.push_back(e); break;
                case 'b': out.push_back('\\b'); break; case 'f': out.push_back('\\f'); break;
                case 'n': out.push_back('\\n'); break; case 'r': out.push_back('\\r'); break;
                case 't': out.push_back('\\t'); break; default: out.push_back(e); break; }
        } else out.push_back(c);
    }
    return false;
}
bool json_uint(const std::string &s, size_t &p, uint64_t &v) {
    skip_ws(s,p); if (p>=s.size() || !std::isdigit((unsigned char)s[p])) return false;
    v=0; while(p<s.size() && std::isdigit((unsigned char)s[p])) {
        uint64_t d=(uint64_t)(s[p++]-'0'); if(v>(UINT64_MAX-d)/10) return false; v=v*10+d;
    } return true;
}
bool skip_value(const std::string &s, size_t &p, int depth=0) {
    if(depth>32) return false; skip_ws(s,p); if(p>=s.size()) return false;
    if(s[p]=='"') { std::string x; return json_string(s,p,x); }
    if(s[p]=='[') { ++p; skip_ws(s,p); if(p<s.size()&&s[p]==']'){++p;return true;}
        while(true){ if(!skip_value(s,p,depth+1)) return false; skip_ws(s,p);
            if(p<s.size()&&s[p]==']'){++p;return true;} if(p>=s.size()||s[p++]!=',') return false; } }
    if(s[p]=='{') { ++p; skip_ws(s,p); if(p<s.size()&&s[p]=='}'){++p;return true;}
        while(true){std::string k; if(!json_string(s,p,k)) return false; skip_ws(s,p);
            if(p>=s.size()||s[p++]!=':'||!skip_value(s,p,depth+1)) return false; skip_ws(s,p);
            if(p<s.size()&&s[p]=='}'){++p;return true;} if(p>=s.size()||s[p++]!=',') return false;} }
    while(p<s.size() && s[p]!=',' && s[p]!=']' && s[p]!='}') ++p; return true;
}
bool tensor_object(const std::string &s, size_t &p, MlxTensorInfo &t) {
    skip_ws(s,p); if(p>=s.size()||s[p]!='{') return false; ++p;
    bool have_dtype=false,have_shape=false,have_data=false;
    while(true) {
        skip_ws(s,p); if(p<s.size()&&s[p]=='}'){++p;return have_dtype&&have_shape&&have_data;}
        std::string key; if(!json_string(s,p,key)) return false; skip_ws(s,p);
        if(p>=s.size()||s[p++]!=':') return false;
        if(key=="dtype") { if(!json_string(s,p,t.dtype)) return false; have_dtype=true; }
        else if(key=="shape") {
            skip_ws(s,p); if(p>=s.size()||s[p++]!='[') return false; skip_ws(s,p);
            if(p<s.size()&&s[p]==']'){++p;have_shape=true;}
            else { while(true){uint64_t x; if(!json_uint(s,p,x)) return false; t.shape.push_back(x); skip_ws(s,p);
                    if(p<s.size()&&s[p]==']'){++p;have_shape=true;break;} if(p>=s.size()||s[p++]!=',') return false;} }
        } else if(key=="data_offsets") {
            skip_ws(s,p); if(p>=s.size()||s[p++]!='[') return false;
            uint64_t a,b; if(!json_uint(s,p,a)) return false; skip_ws(s,p);
            if(p>=s.size()||s[p++]!=',') return false; if(!json_uint(s,p,b)) return false; skip_ws(s,p);
            if(p>=s.size()||s[p++]!=']') return false; t.data_begin=a; t.data_end=b; have_data=true;
        } else if(!skip_value(s,p)) return false;
        skip_ws(s,p); if(p<s.size()&&s[p]==','){++p;continue;} if(p<s.size()&&s[p]=='}'){++p;return have_dtype&&have_shape&&have_data;} return false;
    }
}
bool find_matching_object(const std::string &s, size_t &p, MlxSafetensorsInfo &out) {
    skip_ws(s,p); if(p>=s.size()||s[p++]!='{') return false;
    while(true) {
        skip_ws(s,p); if(p<s.size()&&s[p]=='}'){++p;return true;}
        std::string key; if(!json_string(s,p,key)) return false; skip_ws(s,p);
        if(p>=s.size()||s[p++]!=':') return false;
        if(key=="__metadata__") { if(!skip_value(s,p)) return false; }
        else {
            MlxTensorInfo t; t.name=key;
            if(!tensor_object(s,p,t)) return false;
            if(t.data_end<t.data_begin) return false;
            out.tensors.push_back(t);
            out.total_tensor_bytes += t.data_end-t.data_begin;
            if(t.name.find("scales")!=std::string::npos || t.name.find("biases")!=std::string::npos ||
               t.dtype=="I4" || t.dtype=="U4") ++out.quantized_tensor_count;
        }
        skip_ws(s,p); if(p<s.size()&&s[p]==','){++p;continue;} if(p<s.size()&&s[p]=='}'){++p;return true;} return false;
    }
}
}

bool mlx_safetensors_probe(const std::string &path, MlxSafetensorsInfo &out, std::string &error) {
    out=MlxSafetensorsInfo{};
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f){error="open_failed";return false;}
    auto end=f.tellg(); if(end<16){error="file_too_small";return false;} out.file_bytes=(uint64_t)end;
    f.seekg(0);
    unsigned char h[8]; if(!f.read((char*)h,8)){error="header_length_read_failed";return false;}
    uint64_t n=0; for(int i=0;i<8;i++) n|=(uint64_t)h[i]<<(8*i);
    if(n==0 || n>64ull*1024ull*1024ull || 8+n>(uint64_t)out.file_bytes){error="invalid_header_length";return false;}
    out.header_bytes=n;
    std::string json((size_t)n,'\\0'); if(!f.read(json.data(),(std::streamsize)n)){error="header_read_failed";return false;}
    size_t p=0; if(!find_matching_object(json,p,out)){error="header_json_parse_failed";return false;}
    out.tensor_count=out.tensors.size();
    std::ostringstream q; q<<"safetensors tensors="<<out.tensor_count<<" quantized_marked="<<out.quantized_tensor_count;
    out.quantization_summary=q.str();
    for(auto &t:out.tensors){t.data_begin += 8+n; t.data_end += 8+n;}
    return true;
}

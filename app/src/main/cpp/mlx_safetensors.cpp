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

bool mlx_infer_affine4(const MlxSafetensorsInfo & info, MlxQuantInfo & out, std::string & error) {
    out = MlxQuantInfo{};
    for (const auto & w : info.tensors) {
        if (w.dtype != "U32" || w.shape.size() != 2) continue;
        if (w.name.size() < 7 || w.name.compare(w.name.size()-7,7,".weight") != 0) continue;
        std::string base = w.name.substr(0,w.name.size()-7);
        const MlxTensorInfo * sc=nullptr, * bi=nullptr;
        for (const auto & t : info.tensors) {
            if (t.name == base + ".scales") sc=&t;
            else if (t.name == base + ".biases") bi=&t;
        }
        if (!sc || !bi || sc->shape.size()!=2 || bi->shape!=sc->shape) continue;
        uint64_t packed_k=w.shape[1];
        uint64_t logical_k=packed_k*8ull;
        if (logical_k==0 || sc->shape[1]==0 || logical_k % sc->shape[1] != 0) {
            error="invalid_affine4_group_shape:"+w.name; return false;
        }
        uint64_t group=logical_k/sc->shape[1];
        if (group != 64) {
            error="unexpected_group_size="+std::to_string(group)+":"+w.name; return false;
        }
        if (sc->dtype!="F16" && sc->dtype!="BF16" && sc->dtype!="F32") {
            error="unexpected_scale_dtype="+sc->dtype+":"+w.name; return false;
        }
        if (bi->dtype!=sc->dtype) {
            error="scale_bias_dtype_mismatch:"+w.name; return false;
        }
        ++out.quantized_weight_count; ++out.scale_count; ++out.bias_count;
    }
    if (out.quantized_weight_count==0) { error="no_mlx_affine4_weight_triplets"; return false; }
    out.valid=true; out.bits=4; out.group_size=64; return true;
}

bool mlx_read_tensor_range(const std::string & path, const MlxTensorInfo & tensor,
                           uint64_t relative_offset, void * dst, size_t bytes, std::string & error) {
    const uint64_t tensor_bytes = tensor.data_end - tensor.data_begin;
    if (relative_offset > tensor_bytes || bytes > tensor_bytes-relative_offset) {
        error="tensor_range_oob"; return false;
    }
    std::ifstream f(path,std::ios::binary);
    if(!f){error="open_failed";return false;}
    f.seekg((std::streamoff)(tensor.data_begin + relative_offset),std::ios::beg);
    if(!f){error="seek_failed";return false;}
    if(bytes && !f.read(reinterpret_cast<char*>(dst),(std::streamsize)bytes)){error="read_failed";return false;}
    return true;
}

bool mlx_decode_affine4_tile(const uint32_t * packed, size_t packed_words,
                             const float * scales, const float * biases,
                             size_t rows, size_t cols, size_t group_size,
                             float * out, size_t out_capacity) {
    if (!packed || !scales || !biases || !out || !rows || !cols ||
        group_size != 64 || (cols & 7u) != 0u) return false;
    const size_t count = rows * cols;
    if (count > out_capacity || packed_words < count / 8u) return false;
    const size_t groups_per_row = cols / group_size;
    if (groups_per_row == 0) return false;
    // MLX affine4 stores eight 4-bit values in each uint32 word. Keep this
    // decoder tile-local so the 4-bit model never needs a full F32 expansion.
    for (size_t r=0; r<rows; ++r) {
        for (size_t k=0; k<cols; ++k) {
            const uint32_t word = packed[r * (cols/8u) + (k/8u)];
            const uint32_t q = (word >> ((k & 7u) * 4u)) & 0xFu;
            const size_t g = r * groups_per_row + (k / group_size);
            out[r * cols + k] = (float)q * scales[g] + biases[g];
        }
    }
    return true;
}

#include "mcnpu_backend.h"

MlxNpuTileResult mlx_affine4_npu_matmul_tile(
        const float * activation,
        const uint32_t * packed_weight,
        const float * scales,
        const float * biases,
        uint32_t m, uint32_t k, uint32_t n,
        uint32_t group_size) {
    MlxNpuTileResult r;
    r.m=m; r.k=k; r.n=n;
    if (!activation || !packed_weight || !scales || !biases ||
        !m || !k || !n || group_size != 64 || (k & 7u) != 0u ||
        !mcnpu_backend_ready()) {
        r.status = "ERR MLX_NPU_TILE invalid_or_backend_not_ready";
        return r;
    }

    const size_t na=(size_t)m*k, nw=(size_t)n*k, nc=(size_t)m*n;
    std::vector<float> wf(nw);
    if (!mlx_decode_affine4_tile(packed_weight, nw/8u, scales, biases,
                                 n, k, group_size, wf.data(), wf.size())) {
        r.status="ERR MLX_NPU_TILE decode_failed";
        return r;
    }

    float ma=0.0f, mw=0.0f;
    for(size_t i=0;i<na;i++) ma=std::max(ma,std::fabs(activation[i]));
    for(size_t i=0;i<nw;i++) mw=std::max(mw,std::fabs(wf[i]));
    if(ma==0.0f || mw==0.0f) {
        r.ok=true; r.status="OK MLX_NPU_TILE zero"; return r;
    }

    const float sa=ma/127.0f, sw=mw/127.0f;
    std::vector<int8_t> qa(na), qw(nw), qc(nc);
    for(size_t i=0;i<na;i++) {
        const float x=activation[i]/sa;
        qa[i]=(int8_t)std::max(-127.0f,std::min(127.0f,std::lrintf(x)));
    }
    for(size_t i=0;i<nw;i++) {
        const float x=wf[i]/sw;
        qw[i]=(int8_t)std::max(-127.0f,std::min(127.0f,std::lrintf(x)));
    }

    float so=1.0f;
    r.status=mcnpu_backend_matmul_int8(
        qa.data(), qw.data(), qc.data(), m, k, n, so);
    if(r.status.rfind("OK",0)!=0) return r;
    r.ok=true; r.scale=sa*sw*so;
    r.status += " path=MLX_AFFINE4_TILE_NPU";
    return r;
}

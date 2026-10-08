#include "mlx_safetensors.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <cstring>
#include <utility>

namespace {
std::atomic<uint64_t> g_affine4_read_bytes{0};
std::atomic<uint64_t> g_affine4_read_count{0};
std::atomic<uint64_t> g_affine4_read_us{0};

bool is_ws(char c) { return c==' ' || c=='\t' || c=='\r' || c=='\n'; }

void skip_ws(const std::string &s, size_t &p) { while (p<s.size() && is_ws(s[p])) ++p; }

bool json_string(const std::string &s, size_t &p, std::string &out) {
    skip_ws(s,p);
    if (p>=s.size() || s[p]!='"') return false;
    ++p; out.clear();
    while (p<s.size()) {
        char c=s[p++];
        if (c=='"') return true;
        if (c=='\\') {
            if (p>=s.size()) return false;
            char e=s[p++];
            switch(e) { case '"': case '\\': case '/': out.push_back(e); break;
                case 'b': out.push_back('\b'); break; case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break; case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break; default: out.push_back(e); break; }
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
struct PersistentReader {
    std::string path;
    std::ifstream file;

    bool open(const std::string &next) {
        if (path == next && file.is_open()) return true;
        file.close();
        file.clear();
        path.clear();
        file.open(next, std::ios::binary);
        if (!file) return false;
        path = next;
        return true;
    }
};

// All model reads for one inference thread share this descriptor. The executor
// repeatedly asks for small weight tiles; reopening a 5-GB file for every tile
// adds avoidable syscall/path-resolution overhead. thread_local keeps the helper
// race-free even if a diagnostic thread reads another model path.
thread_local PersistentReader g_model_reader;

bool open_model_reader(const std::string &path) {
    return g_model_reader.open(path);
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
    std::string json((size_t)n,'\0'); if(!f.read(json.data(),(std::streamsize)n)){error="header_read_failed";return false;}
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
    if(!open_model_reader(path)){error="open_failed";return false;}
    std::ifstream &f=g_model_reader.file;
    f.clear();
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

static float decode_half(uint16_t h) {
    const uint32_t sign=(uint32_t)(h&0x8000u)<<16;
    const uint32_t exp=(h>>10)&0x1fu;
    const uint32_t frac=h&0x3ffu;
    uint32_t bits;
    if(exp==0) {
        if(frac==0) bits=sign;
        else {
            float v=std::ldexp((float)frac,-24);
            return (sign ? -v : v);
        }
    } else if(exp==31) {
        bits=sign|0x7f800000u|(frac<<13);
    } else {
        bits=sign|((exp+112u)<<23)|(frac<<13);
    }
    float v;
    std::memcpy(&v,&bits,sizeof(v));
    return v;
}

static float decode_scalar(const unsigned char *p, const std::string &dtype) {
    if(dtype=="F32") {
        float v; std::memcpy(&v,p,4); return v;
    }
    if(dtype=="F16") {
        uint16_t v; std::memcpy(&v,p,2); return decode_half(v);
    }
    if(dtype=="BF16") {
        uint16_t v; std::memcpy(&v,p,2);
        uint32_t bits=((uint32_t)v)<<16;
        float out; std::memcpy(&out,&bits,4); return out;
    }
    return 0.0f;
}

static const MlxTensorInfo * find_tensor(const MlxSafetensorsInfo &info,
                                         const std::string &name) {
    for(const auto &t: info.tensors) if(t.name==name) return &t;
    return nullptr;
}

#include "mcnpu_backend.h"


bool mlx_read_affine4_tile(const std::string & path,
                           const MlxSafetensorsInfo & info,
                           const std::string & tensor_suffix,
                           uint32_t row0, uint32_t rows,
                           uint32_t col0, uint32_t cols,
                           MlxAffine4Tile & out,
                           std::string & error) {
    out = MlxAffine4Tile{};
    if(!rows || !cols || (col0 & 7u) || (cols & 7u) ||
       (col0 + cols < col0) || (row0 + rows < row0)) {
        error="bad_tile";
        return false;
    }
    const MlxTensorInfo * w=nullptr;
    for(const auto &t: info.tensors) {
        if(t.dtype=="U32" && t.shape.size()==2 && t.name == tensor_suffix) {
            w=&t; break;
        }
    }
    if(!w) { error="affine4_weight_not_found:"+tensor_suffix; return false; }
    if(row0 + rows > w->shape[0]) { error="row_oob"; return false; }
    const uint64_t logicalK=w->shape[1]*8ull;
    if((uint64_t)col0 + cols > logicalK || logicalK%64ull!=0) {
        error="col_oob_or_bad_group"; return false;
    }
    const std::string base=w->name.substr(0,w->name.size()-7);
    const MlxTensorInfo *sc=find_tensor(info,base+".scales");
    const MlxTensorInfo *bi=find_tensor(info,base+".biases");
    if(!sc || !bi || sc->shape.size()!=2 || bi->shape!=sc->shape ||
       sc->shape[0]!=w->shape[0] || sc->shape[1]!=(logicalK/64ull) ||
       sc->dtype!=bi->dtype ||
       (sc->dtype!="F16" && sc->dtype!="BF16" && sc->dtype!="F32")) {
        error="affine4_metadata_mismatch:"+w->name; return false;
    }
    if(col0%64u || cols%64u) { error="tile_must_align_to_group64"; return false; }

    const size_t packedPerRow=(size_t)cols/8u;
    out.rows=rows; out.cols=cols; out.tensor_name=w->name;
    out.packed_weight.resize((size_t)rows*packedPerRow);

    // Reuse one persistent descriptor for the whole tile and across subsequent
    // projection calls in the same inference thread. This keeps the hot path
    // from repeatedly opening/closing the 5-GB model file.
    if(!open_model_reader(path)){ error="open_failed"; return false; }
    std::ifstream &f=g_model_reader.file;
    uint64_t tileReadBytes = 0;
    const auto tileRead0 = std::chrono::steady_clock::now();
    auto read_open = [&](const MlxTensorInfo &t, uint64_t rel, void *dst, size_t bytes)->bool {
        const uint64_t tensor_bytes=t.data_end-t.data_begin;
        if(rel>tensor_bytes || bytes>tensor_bytes-rel){ error="tensor_range_oob"; return false; }
        f.clear();
        f.seekg((std::streamoff)(t.data_begin+rel),std::ios::beg);
        if(!f){ error="seek_failed"; return false; }
        if(bytes==0) return true;
        if(!f.read(reinterpret_cast<char*>(dst),(std::streamsize)bytes)){ error="read_failed"; return false; }
        tileReadBytes += bytes;
        return true;
    };

    const size_t rowBytes=(size_t)w->shape[1]*sizeof(uint32_t);
    for(uint32_t r=0;r<rows;r++){
        const uint64_t off=(uint64_t)(row0+r)*rowBytes+(uint64_t)(col0/8u)*sizeof(uint32_t);
        if(!read_open(*w,off,out.packed_weight.data()+(size_t)r*packedPerRow,
                      packedPerRow*sizeof(uint32_t))) return false;
    }

    const size_t scalarBytes=sc->dtype=="F32"?4u:2u;
    const size_t group0=col0/64u;
    const size_t groups=cols/64u;
    std::vector<unsigned char> rawS((size_t)rows*groups*scalarBytes);
    std::vector<unsigned char> rawB(rawS.size());
    const size_t scaleRowBytes=(size_t)sc->shape[1]*scalarBytes;
    for(uint32_t r=0;r<rows;r++) {
        const uint64_t off=(uint64_t)(row0+r)*scaleRowBytes+(uint64_t)group0*scalarBytes;
        if(!read_open(*sc,off,rawS.data()+(size_t)r*groups*scalarBytes,groups*scalarBytes) ||
           !read_open(*bi,off,rawB.data()+(size_t)r*groups*scalarBytes,groups*scalarBytes))
            return false;
    }
    out.scales.resize((size_t)rows*groups);
    out.biases.resize(out.scales.size());
    for(size_t i=0;i<out.scales.size();i++) {
        out.scales[i]=decode_scalar(rawS.data()+i*scalarBytes,sc->dtype);
        out.biases[i]=decode_scalar(rawB.data()+i*scalarBytes,bi->dtype);
        if(!std::isfinite(out.scales[i]) || !std::isfinite(out.biases[i])) {
            error="nonfinite_affine_params";
            return false;
        }
    }
    const uint64_t elapsedUs = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tileRead0).count();
    g_affine4_read_bytes.fetch_add(tileReadBytes, std::memory_order_relaxed);
    g_affine4_read_count.fetch_add(1, std::memory_order_relaxed);
    g_affine4_read_us.fetch_add(elapsedUs, std::memory_order_relaxed);
    return true;
}

bool mlx_reset_affine4_io_stats() {
    g_affine4_read_bytes.store(0, std::memory_order_relaxed);
    g_affine4_read_count.store(0, std::memory_order_relaxed);
    g_affine4_read_us.store(0, std::memory_order_relaxed);
    return true;
}

std::string mlx_affine4_io_status() {
    const auto bytes = g_affine4_read_bytes.load(std::memory_order_relaxed);
    const auto count = g_affine4_read_count.load(std::memory_order_relaxed);
    const auto us = g_affine4_read_us.load(std::memory_order_relaxed);
    return "reads=" + std::to_string(count) +
           " bytes=" + std::to_string(bytes) +
           " read_us=" + std::to_string(us);
}


std::string mlx_affine4_npu_probe(const std::string & path,
                                  const MlxSafetensorsInfo & info) {
    if(!mcnpu_backend_ready())
        return "ERR MLX_NPU_PROBE backend_not_ready " + mcnpu_backend_status();

    constexpr uint32_t M=32, K=64, N=32;
    for(const auto &w: info.tensors) {
        if(w.dtype!="U32" || w.shape.size()!=2 || w.shape[0]<N || w.shape[1]<K/8u)
            continue;
        if(w.name.size()<7 || w.name.compare(w.name.size()-7,7,".weight")!=0)
            continue;
        const std::string base=w.name.substr(0,w.name.size()-7);
        const MlxTensorInfo *sc=find_tensor(info,base+".scales");
        const MlxTensorInfo *bi=find_tensor(info,base+".biases");
        if(!sc || !bi || sc->shape.size()!=2 || bi->shape!=sc->shape)
            continue;
        const uint64_t packedK=w.shape[1];
        const uint64_t logicalK=packedK*8ull;
        if(logicalK< K || logicalK%64ull!=0 || sc->shape[0]<N ||
           sc->shape[1]!=(logicalK/64ull))
            continue;
        if(sc->dtype!="F16" && sc->dtype!="BF16" && sc->dtype!="F32")
            continue;
        if(bi->dtype!=sc->dtype) continue;

        std::string err;
        MlxAffine4Tile tile;
        if(!mlx_read_affine4_tile(path,info,w.name,0,N,0,K,tile,err))
            return "ERR MLX_NPU_PROBE tile_read="+err+" tensor="+w.name;
        std::vector<uint32_t> packed=std::move(tile.packed_weight);
        std::vector<float> scales=std::move(tile.scales);
        std::vector<float> biases=std::move(tile.biases);

        std::vector<float> activation((size_t)M*K);
        for(size_t i=0;i<activation.size();++i)
            activation[i]=std::sin((float)i*0.017f)*0.5f;

        std::vector<float> decoded((size_t)N*K);
        if(!mlx_decode_affine4_tile(packed.data(),packed.size(),
                                    scales.data(),biases.data(),
                                    N,K,64,decoded.data(),decoded.size()))
            return "ERR MLX_NPU_PROBE decode_failed tensor="+w.name;

        const auto t0=std::chrono::steady_clock::now();
        auto result=mlx_affine4_npu_matmul_tile(
            activation.data(),packed.data(),scales.data(),biases.data(),
            M,K,N,64);
        const auto t1=std::chrono::steady_clock::now();
        const auto us=std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
        if(!result.ok)
            return "ERR MLX_NPU_PROBE npu="+result.status+
                   " tensor="+w.name+" elapsed_us="+std::to_string(us);

        float checksum=0.0f;
        for(float x:decoded) checksum+=x;
        return "OK MLX_NPU_PROBE/1 tensor="+w.name+
               " shape=32x64x32 elapsed_us="+std::to_string(us)+
               " decoded_checksum="+std::to_string(checksum)+
               " max_abs_error="+std::to_string(result.max_abs_error)+
               " max_relative_error="+std::to_string(result.max_relative_error)+
               " npu="+result.status;
    }
    return "ERR MLX_NPU_PROBE no_compatible_affine4_tensor";
}

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
    std::vector<int8_t> qa(na), qw((size_t)k*n), qc(nc);
    for(size_t i=0;i<na;i++) {
        const float x=activation[i]/sa;
        qa[i]=(int8_t)std::max(-127.0f,std::min(127.0f,std::lrintf(x)));
    }
    // The MLX tile is stored [N,K], while the MCNPU bridge consumes the
    // conventional [M,K] x [K,N] row-major layout. Transpose while quantizing
    // so the NPU never sees a silently swapped B matrix.
    for(uint32_t row=0; row<n; ++row) {
        for(uint32_t col=0; col<k; ++col) {
            const float x=wf[(size_t)row*k+col]/sw;
            qw[(size_t)col*n+row]=(int8_t)std::max(
                -127.0f,std::min(127.0f,std::lrintf(x)));
        }
    }

    float npuScale=0.0f;
    r.status=mcnpu_backend_matmul_int8(
        qa.data(), qw.data(), qc.data(), m, k, n, npuScale);
    if(r.status.rfind("OK",0)!=0) return r;
    r.ok=true; r.scale=npuScale*(sa*sw/1.0e-6f);

    float max_abs=0.0f, max_rel=0.0f;
    for(uint32_t i=0;i<m;i++) {
        for(uint32_t j=0;j<n;j++) {
            float ref=0.0f;
            for(uint32_t d=0;d<k;d++)
                ref += activation[(size_t)i*k+d] * wf[(size_t)j*k+d];
            const float got=(float)qc[(size_t)i*n+j]*r.scale;
            const float abs_err=std::fabs(got-ref);
            max_abs=std::max(max_abs,abs_err);
            max_rel=std::max(max_rel,abs_err/(std::fabs(ref)+1.0e-4f));
        }
    }
    r.max_abs_error=max_abs;
    r.max_relative_error=max_rel;
    r.status += " path=MLX_AFFINE4_TILE_NPU";
    return r;
}


bool mlx_read_affine4_row(const std::string &path, const MlxSafetensorsInfo &info,
                          const std::string &weight_name, uint32_t row,
                          std::vector<float> &out, std::string &error) {
    const MlxTensorInfo *w = find_tensor(info, weight_name);
    if (!w || w->dtype != "U32" || w->shape.size() != 2) {
        error = "row_weight_not_found";
        return false;
    }
    const uint64_t k64 = w->shape[1] * 8ull;
    if (k64 == 0 || k64 > UINT32_MAX || k64 % 64ull != 0 || row >= w->shape[0]) {
        error = "row_bad_shape";
        return false;
    }
    MlxAffine4Tile tile;
    if (!mlx_read_affine4_tile(path, info, weight_name, row, 1, 0,
                               (uint32_t)k64, tile, error)) return false;
    out.resize((size_t)k64);
    if (!mlx_decode_affine4_tile(tile.packed_weight.data(), tile.packed_weight.size(),
                                 tile.scales.data(), tile.biases.data(), 1, (size_t)k64,
                                 64, out.data(), out.size())) {
        error = "row_decode_failed";
        return false;
    }
    return true;
}

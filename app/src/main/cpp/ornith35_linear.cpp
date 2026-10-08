#include "ornith35_linear.h"
#include "mlx_safetensors.h"
#include "mcnpu_backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
static bool quantize_i8(const float *src, size_t count, std::vector<int8_t> &dst, float &scale) {
    float mx=0.0f;
    for(size_t i=0;i<count;i++) mx=std::max(mx,std::fabs(src[i]));
    if(mx==0.0f) { scale=1.0f; dst.assign(count,0); return true; }
    scale=mx/127.0f;
    dst.resize(count);
    for(size_t i=0;i<count;i++) {
        float q=src[i]/scale;
        dst[i]=(int8_t)std::max(-127.0f,std::min(127.0f,std::lrintf(q)));
    }
    return true;
}
}

bool ornith35_run_projection_tile(
    const std::string & model_path,
    const MlxSafetensorsInfo & info,
    const std::string & weight_name,
    const float * input,
    uint32_t m, uint32_t k,
    float * output, uint32_t n,
    Ornith35ProjectionStats & stats) {
    if(!input || !output || !m || !k || !n) { stats.status="ERR invalid_args"; return false; }
    if(!mcnpu_backend_ready()) { stats.status="ERR "+mcnpu_backend_status(); return false; }
    if(k%64u || n%32u || m%32u) { stats.status="ERR tile_alignment_m32_n32_k64"; return false; }

    std::fill(output,output+(size_t)m*n,0.0f);
    std::vector<int8_t> qa, qw, qc;
    const uint32_t row_tile=32, col_tile=32, k_tile=64;
    for(uint32_t r0=0;r0<m;r0+=row_tile) {
        const uint32_t mr=std::min(row_tile,m-r0);
        for(uint32_t n0=0;n0<n;n0+=col_tile) {
            const uint32_t nn=std::min(col_tile,n-n0);
            if(mr!=32 || nn!=32) { stats.status="ERR nonbucket_edge"; return false; }
            for(uint32_t k0=0;k0<k;k0+=k_tile) {
                MlxAffine4Tile tile;
                std::string err;
                if(!mlx_read_affine4_tile(model_path,info,weight_name,n0,nn,k0,k_tile,tile,err)) {
                    stats.status="ERR tile_read="+err; return false;
                }
                float sa=1.0f, sw=1.0f;
                quantize_i8(input+(size_t)r0*k+k0,mr*k_tile,qa,sa);
                std::vector<float> wf((size_t)nn*k_tile);
                if(!mlx_decode_affine4_tile(tile.packed_weight.data(),tile.packed_weight.size(),
                                            tile.scales.data(),tile.biases.data(),
                                            nn,k_tile,64,wf.data(),wf.size())) {
                    stats.status="ERR decode"; return false;
                }
                // MLX affine4 decodes each tile as [N,K]. MCNPU expects
                // B in [K,N], so transpose at the quantization boundary.
                qw.resize((size_t)k_tile*nn);
                float wmx=0.0f;
                for(size_t i=0;i<wf.size();++i) wmx=std::max(wmx,std::fabs(wf[i]));
                if(wmx==0.0f) {
                    sw=1.0f;
                    std::fill(qw.begin(),qw.end(),0);
                } else {
                    sw=wmx/127.0f;
                    for(uint32_t row=0; row<nn; ++row)
                        for(uint32_t col=0; col<k_tile; ++col) {
                            float q=wf[(size_t)row*k_tile+col]/sw;
                            qw[(size_t)col*nn+row]=(int8_t)std::max(
                                -127.0f,std::min(127.0f,std::lrintf(q)));
                        }
                }
                qc.resize((size_t)mr*nn);
                std::string s=mcnpu_backend_matmul_int8(
                    qa.data(),qw.data(),qc.data(),mr,k_tile,nn,1.0f);
                if(s.rfind("OK",0)!=0) { stats.status=s; return false; }
                const float so=sa*sw;
                for(uint32_t i=0;i<mr;i++)
                    for(uint32_t j=0;j<nn;j++)
                        output[(size_t)(r0+i)*n+n0+j]+=((float)qc[(size_t)i*nn+j])*so;
                stats.tiles++; stats.npu_calls++;
            }
        }
    }
    stats.rows=m; stats.cols=n; stats.status="OK ORNITH35_PROJECTION_NPU_TILED";
    return true;
}

bool ornith35_run_projection_token(
    const std::string & model_path,
    const MlxSafetensorsInfo & info,
    const std::string & weight_name,
    const float * input,
    uint32_t k,
    float * output,
    uint32_t n,
    Ornith35ProjectionStats & stats) {
    if (!input || !output || !k || !n) {
        stats.status = "ERR invalid_args";
        return false;
    }

    const uint32_t kp = (k + 63u) & ~63u;
    const uint32_t np = (n + 31u) & ~31u;
    std::vector<float> padded((size_t)32 * kp, 0.0f);
    std::copy(input, input + k, padded.begin());

    std::vector<float> tmp((size_t)32 * np, 0.0f);
    Ornith35ProjectionStats inner;
    if (!ornith35_run_projection_tile(
            model_path, info, weight_name,
            padded.data(), 32, kp,
            tmp.data(), np, inner)) {
        stats = inner;
        return false;
    }

    std::copy(tmp.begin(), tmp.begin() + n, output);
    stats = inner;
    stats.status = "OK ORNITH35_PROJECTION_TOKEN_BUCKET32";
    return true;
}


bool ornith35_run_mlp(
    const float *hidden, float *out, uint32_t hidden_size,
    uint32_t intermediate_size, const std::string &layer_prefix,
    const MlxSafetensorsInfo &model, Ornith35MlpStats &stats) {
    if(!hidden || !out || hidden_size != 4096u || intermediate_size != 12288u) {
        stats.status="ERR mlp_shape";
        return false;
    }
    std::vector<float> gate(intermediate_size), up(intermediate_size), fused(intermediate_size);
    Ornith35ProjectionStats ps;
    const std::string gate_name=layer_prefix+"mlp.gate_proj.weight";
    const std::string up_name=layer_prefix+"mlp.up_proj.weight";
    const std::string down_name=layer_prefix+"mlp.down_proj.weight";
    if(!ornith35_run_projection_token("",model,gate_name,hidden,hidden_size,
                                      gate.data(),intermediate_size,ps)) {
        stats.status="ERR gate="+ps.status; return false;
    }
    stats.projection_calls++;
    if(!ornith35_run_projection_token("",model,up_name,hidden,hidden_size,
                                      up.data(),intermediate_size,ps)) {
        stats.status="ERR up="+ps.status; return false;
    }
    stats.projection_calls++;
    for(uint32_t i=0;i<intermediate_size;i++) {
        const float x=gate[i];
        const float silu=x/(1.0f+std::exp(-x));
        fused[i]=silu*up[i];
    }
    if(!ornith35_run_projection_token("",model,down_name,fused.data(),intermediate_size,
                                      out,hidden_size,ps)) {
        stats.status="ERR down="+ps.status; return false;
    }
    stats.projection_calls++;
    stats.npu=true;
    stats.status="OK ORNITH35_MLP_NPU";
    return true;
}

#include "ornith15_linear.h"
#include "mlx_safetensors.h"
#include "mcnpu_backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
static uint32_t projection_col_tile(uint32_t n) {
    // Choose the largest stable bucket that does not exceed 4096. Small
    // projections such as beta/alpha (32 outputs) must not trigger a 4096-wide
    // matmul just because the largest matrix benefits from it.
    if (n <= 32u) return 32u;
    if (n <= 64u) return 64u;
    if (n <= 128u) return 128u;
    if (n <= 256u) return 256u;
    if (n <= 512u) return 512u;
    if (n <= 1024u) return 1024u;
    if (n <= 2048u) return 2048u;
    return 4096u;
}
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

bool ornith15_run_projection_tile(
    const std::string & model_path,
    const MlxSafetensorsInfo & info,
    const std::string & weight_name,
    const float * input,
    uint32_t m, uint32_t k,
    float * output, uint32_t n,
    Ornith15ProjectionStats & stats) {
    if(!input || !output || !m || !k || !n) { stats.status="ERR invalid_args"; return false; }
    if(!mcnpu_backend_ready()) { stats.status="ERR "+mcnpu_backend_status(); return false; }
    if(k%64u || n%32u || m%32u) { stats.status="ERR tile_alignment_m32_n32_k64"; return false; }

    std::fill(output,output+(size_t)m*n,0.0f);
    std::vector<int8_t> qa, qw, qc;
    // Use the largest empirically stable HTP V73 buckets to keep the
    // per-token projection call count practical. M stays 32 for decode-time
    // padding; K/N use 1024-byte-aligned buckets observed to be stable.
    // Decode-time M is only 32, while K/N can use the larger measured HTP
    // buckets. Keeping both matrix axes at 4096 dramatically cuts graph-execute
    // count without materializing the full 5-GB model.
    const uint32_t row_tile=32, k_tile=4096;
    for(uint32_t r0=0;r0<m;r0+=row_tile) {
        const uint32_t mr=std::min(row_tile,m-r0);
        const uint32_t col_tile=projection_col_tile(n);
        for(uint32_t n0=0;n0<n;n0+=col_tile) {
            const uint32_t actual_nn=std::min(col_tile,n-n0);
            if(mr!=32 || actual_nn==0) { stats.status="ERR nonbucket_edge"; return false; }
            const uint32_t nn=col_tile;
            for(uint32_t k0=0;k0<k;k0+=k_tile) {
                MlxAffine4Tile tile;
                std::string err;
                if(!mlx_read_affine4_tile(model_path,info,weight_name,n0,actual_nn,k0,k_tile,tile,err)) {
                    stats.status="ERR tile_read="+err; return false;
                }
                if(actual_nn != nn) {
                    const size_t packed_per_row=(size_t)k_tile/8u;
                    const size_t groups_per_row=(size_t)k_tile/64u;
                    MlxAffine4Tile padded;
                    padded.rows=nn; padded.cols=k_tile; padded.tensor_name=tile.tensor_name;
                    padded.packed_weight.assign((size_t)nn*packed_per_row,0u);
                    padded.scales.assign((size_t)nn*groups_per_row,0.0f);
                    padded.biases.assign((size_t)nn*groups_per_row,0.0f);
                    for(uint32_t rr=0;rr<actual_nn;rr++) {
                        std::copy(tile.packed_weight.begin()+(size_t)rr*packed_per_row,
                                  tile.packed_weight.begin()+(size_t)(rr+1)*packed_per_row,
                                  padded.packed_weight.begin()+(size_t)rr*packed_per_row);
                        std::copy(tile.scales.begin()+(size_t)rr*groups_per_row,
                                  tile.scales.begin()+(size_t)(rr+1)*groups_per_row,
                                  padded.scales.begin()+(size_t)rr*groups_per_row);
                        std::copy(tile.biases.begin()+(size_t)rr*groups_per_row,
                                  tile.biases.begin()+(size_t)(rr+1)*groups_per_row,
                                  padded.biases.begin()+(size_t)rr*groups_per_row);
                    }
                    tile=std::move(padded);
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
                float npu_scale = 0.0f;
                std::string s=mcnpu_backend_matmul_int8(
                    qa.data(),qw.data(),qc.data(),mr,k_tile,nn,npu_scale);
                if(s.rfind("OK",0)!=0) { stats.status=s; return false; }
                // The QNN bridge calibrates integer dot products with fixed
                // 0.001 input scales. Convert that result back to this tile's
                // actual dynamic activation and weight scales.
                const float so=npu_scale*(sa*sw/1.0e-6f);
                for(uint32_t i=0;i<mr;i++)
                    for(uint32_t j=0;j<nn;j++)
                        output[(size_t)(r0+i)*n+n0+j]+=((float)qc[(size_t)i*nn+j])*so;
                stats.tiles++; stats.npu_calls++;
            }
        }
    }
    stats.rows=m; stats.cols=n; stats.status="OK ORNITH15_PROJECTION_NPU_TILED";
    return true;
}

bool ornith15_run_projection_token(
    const std::string & model_path,
    const MlxSafetensorsInfo & info,
    const std::string & weight_name,
    const float * input,
    uint32_t k,
    float * output,
    uint32_t n,
    Ornith15ProjectionStats & stats) {
    if (!input || !output || !k || !n) {
        stats.status = "ERR invalid_args";
        return false;
    }

    const uint32_t kp = (k + 63u) & ~63u;
    const uint32_t np = (n + 31u) & ~31u;
    std::vector<float> padded((size_t)32 * kp, 0.0f);
    std::copy(input, input + k, padded.begin());

    std::vector<float> tmp((size_t)32 * np, 0.0f);
    Ornith15ProjectionStats inner;
    if (!ornith15_run_projection_tile(
            model_path, info, weight_name,
            padded.data(), 32, kp,
            tmp.data(), np, inner)) {
        stats = inner;
        return false;
    }

    std::copy(tmp.begin(), tmp.begin() + n, output);
    stats = inner;
    stats.status = "OK ORNITH15_PROJECTION_TOKEN_BUCKET32";
    return true;
}


bool ornith15_run_mlp(
    const std::string &model_path, const float *hidden, float *out, uint32_t hidden_size,
    uint32_t intermediate_size, const std::string &layer_prefix,
    const MlxSafetensorsInfo &model, Ornith15MlpStats &stats) {
    if(!hidden || !out || hidden_size != 4096u || intermediate_size != 12288u) {
        stats.status="ERR mlp_shape";
        return false;
    }
    std::vector<float> gate(intermediate_size), up(intermediate_size), fused(intermediate_size);
    Ornith15ProjectionStats ps;
    const std::string gate_name=layer_prefix+"gate_proj.weight";
    const std::string up_name=layer_prefix+"up_proj.weight";
    const std::string down_name=layer_prefix+"down_proj.weight";
    if(!ornith15_run_projection_token(model_path,model,gate_name,hidden,hidden_size,
                                      gate.data(),intermediate_size,ps)) {
        stats.status="ERR gate="+ps.status; return false;
    }
    stats.projection_calls++;
    if(!ornith15_run_projection_token(model_path,model,up_name,hidden,hidden_size,
                                      up.data(),intermediate_size,ps)) {
        stats.status="ERR up="+ps.status; return false;
    }
    stats.projection_calls++;
    for(uint32_t i=0;i<intermediate_size;i++) {
        const float x=gate[i];
        const float silu=x/(1.0f+std::exp(-x));
        fused[i]=silu*up[i];
    }
    if(!ornith15_run_projection_token(model_path,model,down_name,fused.data(),intermediate_size,
                                      out,hidden_size,ps)) {
        stats.status="ERR down="+ps.status; return false;
    }
    stats.projection_calls++;
    stats.npu=true;
    stats.status="OK ORNITH15_MLP_NPU";
    return true;
}

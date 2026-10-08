#include "ornith15_linear.h"
#include "mlx_safetensors.h"
#include "mcnpu_backend.h"
#include "runtime_memory_budget.h"
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
static bool quantize_i8_rows(const float *src,
                                uint32_t rows,
                                uint32_t row_stride,
                                uint32_t cols,
                                uint32_t active_rows,
                                std::vector<int8_t> &dst,
                                float &scale) {
    if (!src || !rows || !cols || row_stride < cols || active_rows > rows) return false;
    float mx=0.0f;
    for(uint32_t r=0;r<active_rows;r++)
        for(uint32_t i=0;i<cols;i++)
            mx=std::max(mx,std::fabs(src[(size_t)r*row_stride+i]));
    scale=mx>0.0f ? mx/127.0f : 1.0f;
    dst.assign((size_t)rows*cols,0);
    if(mx==0.0f) return true;
    for(uint32_t r=0;r<active_rows;r++)
        for(uint32_t i=0;i<cols;i++) {
            const float q=src[(size_t)r*row_stride+i]/scale;
            dst[(size_t)r*cols+i]=(int8_t)std::max(-127.0f,std::min(127.0f,std::lrintf(q)));
        }
    return true;
}

static bool affine4_quantize_transposed(const MlxAffine4Tile &tile,
                                        uint32_t rows,
                                        uint32_t cols,
                                        std::vector<int8_t> &out,
                                        float &scale) {
    if (!tile.packed_weight.data() || !rows || !cols ||
        tile.rows!=rows || tile.cols!=cols || (cols&63u)) return false;
    const size_t groups=cols/64u;
    float wmax=0.0f;
    for(uint32_t r=0;r<rows;r++)
        for(uint32_t k=0;k<cols;k++) {
            const uint32_t word=tile.packed_weight[(size_t)r*(cols/8u)+(k/8u)];
            const uint32_t q=(word>>((k&7u)*4u))&0xFu;
            const size_t g=(size_t)r*groups+(k/64u);
            const float w=(float)q*tile.scales[g]+tile.biases[g];
            wmax=std::max(wmax,std::fabs(w));
        }
    scale=wmax>0.0f ? wmax/127.0f : 1.0f;
    out.assign((size_t)rows*cols,0);
    if(wmax==0.0f) return true;
    for(uint32_t r=0;r<rows;r++)
        for(uint32_t k=0;k<cols;k++) {
            const uint32_t word=tile.packed_weight[(size_t)r*(cols/8u)+(k/8u)];
            const uint32_t q=(word>>((k&7u)*4u))&0xFu;
            const size_t g=(size_t)r*groups+(k/64u);
            const float w=(float)q*tile.scales[g]+tile.biases[g];
            out[(size_t)k*rows+r]=(int8_t)std::max(-127.0f,std::min(127.0f,std::lrintf(w/scale)));
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

    const uint32_t active_rows = logical_rows ? std::min(logical_rows,m) : m;
    if(active_rows==0 || active_rows>m) { stats.status="ERR logical_rows"; return false; }
    // Decode-time M=32 only needs the logical rows. Keeping the caller buffer
    // at active_rows*n is especially important for the 248320-column LM head.
    std::fill(output,output+(size_t)active_rows*n,0.0f);
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
                const uint32_t active_in_tile = active_rows > r0 ? std::min<uint32_t>(active_rows-r0,mr) : 0;
                if(!quantize_i8_rows(input+(size_t)r0*k+k0,
                                     mr,k,k_tile,active_in_tile,qa,sa)) {
                    stats.status="ERR activation_quantize"; return false;
                }
                const uint64_t tile_temp_bytes =
                    (uint64_t)k_tile * nn * sizeof(int8_t) +
                    (uint64_t)mr * k_tile * sizeof(int8_t) +
                    (uint64_t)mr * nn * sizeof(int8_t);
                std::string mem_error;
                if (!ornith15_memory_headroom(tile_temp_bytes, "projection_tile_temp", mem_error)) {
                    stats.status = "ERR " + mem_error;
                    return false;
                }
                if(!affine4_quantize_transposed(tile,nn,k_tile,qw,sw)) {
                    stats.status="ERR weight_quantize"; return false;
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
                // The last N tile may be padded to the next HTP bucket.
                // Only accumulate the real columns; writing all nn columns would
                // overflow the caller's output when n is not a multiple of 4096
                // (LM head is 248320 rows, i.e. 60 full tiles + 2560 real columns).
                for(uint32_t i=0;i<mr;i++)
                    for(uint32_t j=0;j<actual_nn;j++)
                        output[(size_t)i*n+n0+j]+=((float)qc[(size_t)i*nn+j])*so;
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
    Ornith15ProjectionStats & stats,
    uint32_t logical_rows) {
    if (!input || !output || !k || !n) {
        stats.status = "ERR invalid_args";
        return false;
    }

    const uint32_t kp = (k + 63u) & ~63u;
    const uint32_t np = (n + 31u) & ~31u;
    std::vector<float> padded;
    const float *source = input;
    if (kp != k) {
        padded.assign((size_t)32 * kp, 0.0f);
        std::copy(input, input + k, padded.begin());
        source = padded.data();
    }

    std::vector<float> tmp((size_t)np, 0.0f);
    Ornith15ProjectionStats inner;
    if (!ornith15_run_projection_tile(
            model_path, info, weight_name,
            source, 32, kp,
            tmp.data(), np, inner, 1)) {
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

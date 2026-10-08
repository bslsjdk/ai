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
                quantize_i8(wf.data(),wf.size(),qw,sw);
                qc.resize((size_t)mr*nn);
                std::string s=mcnpu_backend_matmul_int8(qa.data(),qw.data(),qc.data(),mr,k_tile,nn,1.0f);
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

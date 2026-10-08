#include "ornith35_executor.h"
#include "mlx_safetensors.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
static uint16_t rd16(const unsigned char *p){ return (uint16_t)p[0] | ((uint16_t)p[1]<<8); }
static float half_to_f(uint16_t h){
    const uint32_t s=(h>>15)&1u, e=(h>>10)&31u, f=h&1023u;
    uint32_t out;
    if(e==0) out=s<<31;
    else if(e==31) out=(s<<31)|0x7f800000u|(f<<13);
    else out=(s<<31)|((e-15+127)<<23)|(f<<13);
    float v; std::memcpy(&v,&out,sizeof(v)); return v;
}
static float bf16_to_f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float v; std::memcpy(&v,&u,4); return v; }
static bool read_vec(const std::string &path,const MlxTensorInfo &t,uint32_t n,std::vector<float>&out,std::string&err){
    if(t.shape.size()!=1 || t.shape[0]!=n){err="vector_shape";return false;}
    const size_t bytes = t.dtype=="F32" ? (size_t)n*4 : (size_t)n*2;
    std::vector<unsigned char>b(bytes);
    if(!mlx_read_tensor_range(path,t,0,b.data(),bytes,err)) return false;
    out.resize(n);
    if(t.dtype=="F32") std::memcpy(out.data(),b.data(),bytes);
    else if(t.dtype=="F16") for(uint32_t i=0;i<n;i++) out[i]=half_to_f(rd16(b.data()+2*i));
    else if(t.dtype=="BF16") for(uint32_t i=0;i<n;i++) out[i]=bf16_to_f(rd16(b.data()+2*i));
    else {err="vector_dtype";return false;}
    return true;
}
static const MlxTensorInfo *tx(const MlxSafetensorsInfo&i,const std::string&n){
    for(const auto&t:i.tensors) if(t.name==n) return &t; return nullptr;
}
static void zero_centered_rms(float *x, uint32_t n, const std::vector<float> &w, float eps) {
    float ss=0.0f;
    for(uint32_t i=0;i<n;i++) ss += x[i]*x[i];
    const float inv=1.0f/std::sqrt(ss/(float)n+eps);
    for(uint32_t i=0;i<n;i++) x[i]=x[i]*inv*(1.0f+(w.empty()?0.0f:w[i]));
}

}

static const MlxTensorInfo *find_tensor(const MlxSafetensorsInfo &m,const std::string &n){
    for(const auto &t:m.tensors) if(t.name==n) return &t;
    return nullptr;
}
static bool scalar_f32(const MlxTensorInfo &t,const std::string &path,uint64_t off,float &v){
    std::vector<unsigned char> b(4);
    std::string e;
    if(!mlx_read_tensor_range(path,t,off,b.data(),4,e)) return false;
    std::memcpy(&v,b.data(),4); return std::isfinite(v);
}
}

bool ornith35_executor_validate(const MlxSafetensorsInfo &info,
                                 const Ornith35TextConfig &cfg,
                                 std::string &error) {
    if (!ornith35_validate_config(cfg, error)) return false;
    if (info.tensors.empty()) { error="empty_safetensors"; return false; }
    auto plan=ornith35_make_layer_plan();
    if(plan.size()!=cfg.num_layers){error="layer_plan_size_mismatch";return false;}
    uint32_t affine=0;
    for(const auto&t:info.tensors)
        if(t.dtype=="U32"&&t.shape.size()==2&&t.name.size()>=7&&
           t.name.compare(t.name.size()-7,7,".weight")==0) ++affine;
    if(!affine){error="no_affine4_weights";return false;}
    return true;
}

bool ornith35_executor_greedy_step(const std::string &model_path,
                                    const MlxSafetensorsInfo &info,
                                    const Ornith35TextConfig &cfg,
                                    uint32_t token_id,
                                    Ornith35DecoderStep &step,
                                    std::string &error) {
    if(!ornith35_executor_validate(info,cfg,error)) return false;
    if(token_id>=cfg.vocab_size){error="token_id_oob";return false;}
    const std::string prefix="language_model.model.";
    const MlxTensorInfo *emb=find_tensor(info,prefix+"embed_tokens.weight");
    if(!emb){error="embedding_not_found";return false;}
    if(emb->shape.size()!=2 || emb->shape[0]!=cfg.vocab_size || emb->shape[1]*8ull!=cfg.hidden_size){
        error="embedding_shape_mismatch"; return false;
    }
    // The MLX checkpoint uses affine4 row-packed embedding weights. Read only
    // one token row; no full embedding matrix is ever resident.
    std::vector<float> row;
    if(!mlx_read_affine4_row(model_path,info,emb->name,token_id,row,error)) return false;
    step.hidden=std::move(row);
    if(step.hidden.size()!=cfg.hidden_size){error="embedding_decode_size";return false;}
    step.position = step.position == 0 ? 1 : step.position + 1;
    step.stats.layers_done=0;
    step.stats.status="EMBEDDING_READY_EXECUTOR_NEXT";
    return true;
}


bool ornith35_executor_init_runtime(const Ornith35TextConfig &cfg,
                                     uint32_t max_attention_tokens,
                                     Ornith35LayerRuntime &runtime,
                                     std::string &error) {
    if (!ornith35_validate_config(cfg, error)) return false;
    if (!max_attention_tokens) { error = "max_attention_tokens_zero"; return false; }
    // Never preallocate the full 262K attention cache. The runtime is deliberately
    // bounded; a paged KV cache will extend context later without blowing RAM.
    const uint32_t bounded_attention_tokens = std::min(max_attention_tokens, 4096u);
    runtime.delta.clear();
    runtime.attention.clear();
    runtime.delta.resize(cfg.num_layers);
    runtime.attention.resize(cfg.num_layers);
    for (uint32_t i = 0; i < cfg.num_layers; ++i) {
        const auto plan = ornith35_make_layer_plan();
        if (plan[i] == Ornith35LayerType::LinearAttention) {
            if (!ornith35_deltanet_init(runtime.delta[i], 32u * 128u, cfg.linear_conv_kernel)) {
                error = "deltanet_state_init_failed";
                return false;
            }
        } else {
            if (!ornith35_attention_init(runtime.attention[i], bounded_attention_tokens)) {
                error = "attention_state_init_failed";
                return false;
            }
        }
    }
    runtime.work_a.resize(cfg.hidden_size);
    runtime.work_b.resize(cfg.hidden_size);
    runtime.work_c.resize(cfg.hidden_size);
    runtime.initialized_layers = cfg.num_layers;
    return true;
}

bool ornith35_executor_apply_mlp(const std::string &model_path,
                                  const MlxSafetensorsInfo &info,
                                  const Ornith35TextConfig &cfg,
                                  uint32_t layer_index,
                                  const float *hidden,
                                  float *out,
                                  Ornith35ExecutorStats &stats,
                                  std::string &error) {
    if (!hidden || !out || layer_index >= cfg.num_layers) { error = "mlp_args"; return false; }
    const std::string base = "language_model.model.layers." + std::to_string(layer_index) + ".mlp.";
    Ornith35MlpStats ms;
    if (!ornith35_run_mlp(model_path, hidden, out, cfg.hidden_size, cfg.intermediate_size, base, info, ms)) {
        error = ms.status.empty() ? "mlp_failed" : ms.status;
        return false;
    }
    stats.npu_calls += ms.projection_calls;
    return true;
}


bool ornith35_executor_run_attention_layer(const std::string &model_path,
                                           const MlxSafetensorsInfo &info,
                                           const Ornith35TextConfig &cfg,
                                           uint32_t layer_index,
                                           const float *hidden,
                                           float *out,
                                           Ornith35AttentionState &state,
                                           uint32_t position,
                                           Ornith35ExecutorStats &stats,
                                           std::string &error) {
    if(!hidden||!out||layer_index>=cfg.num_layers){error="attention_args";return false;}
    const std::string b="language_model.model.layers."+std::to_string(layer_index)+".self_attn.";
    std::vector<float> q(8192),k(1024),v(1024),qnorm(256),knorm(256),att(4096),gate(4096);
    Ornith35ProjectionStats ps;
    if(!ornith35_run_projection_token(model_path,info,b+"q_proj.weight",hidden,4096,q.data(),8192,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith35_run_projection_token(model_path,info,b+"k_proj.weight",hidden,4096,k.data(),1024,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith35_run_projection_token(model_path,info,b+"v_proj.weight",hidden,4096,v.data(),1024,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    const auto *qt=tx(info,b+"q_norm.weight"),*kt=tx(info,b+"k_norm.weight");
    if(!qt||!kt||!read_vec(model_path,*qt,256,qnorm,error)||!read_vec(model_path,*kt,256,knorm,error)) return false;
    if(!ornith35_attention_step(state,q.data(),k.data(),v.data(),16,4,256,position,10000000.0f,att,error,qnorm.data(),knorm.data())) return false;
    for(uint32_t i=0;i<4096;i++) gate[i]=q[4096+i];
    for(uint32_t i=0;i<4096;i++) att[i]*=1.0f/(1.0f+std::exp(-gate[i]));
    if(!ornith35_run_projection_token(model_path,info,b+"o_proj.weight",att.data(),4096,out,4096,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    return true;
}


bool ornith35_executor_run_delta_layer(const std::string &model_path,
                                       const MlxSafetensorsInfo &info,
                                       const Ornith35TextConfig &cfg,
                                       uint32_t layer_index,
                                       const float *hidden,
                                       float *out,
                                       Ornith35DeltaState &state,
                                       Ornith35ExecutorStats &stats,
                                       std::string &error) {
    if(!hidden||!out||layer_index>=cfg.num_layers){error="delta_args";return false;}
    const std::string b="language_model.model.layers."+std::to_string(layer_index)+".linear_attn.";
    std::vector<float> qkv(8192),z(4096),ba(64),q(2048),k(2048),v(4096),beta(32),a(32),decay(32),core(4096),normw(128);
    Ornith35ProjectionStats ps;
    if(!ornith35_run_projection_token(model_path,info,b+"in_proj_qkv.weight",hidden,4096,qkv.data(),8192,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith35_run_projection_token(model_path,info,b+"in_proj_z.weight",hidden,4096,z.data(),4096,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith35_run_projection_token(model_path,info,b+"in_proj_b.weight",hidden,4096,beta.data(),32,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith35_run_projection_token(model_path,info,b+"in_proj_a.weight",hidden,4096,a.data(),32,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;

    const auto *at=tx(info,b+"A_log"), *dt=tx(info,b+"dt_bias");
    if(!at||!dt||!read_vec(model_path,*at,32,decay,error)) return false;
    std::vector<float> dtv;
    if(!read_vec(model_path,*dt,32,dtv,error)) return false;
    for(uint32_t i=0;i<32;i++){
        beta[i]=1.0f/(1.0f+std::exp(-beta[i]));
        float sp=std::log1p(std::exp(std::min(a[i]+dtv[i],20.0f)));
        decay[i]=-std::exp(decay[i])*sp;
    }

    const auto *ct=tx(info,b+"conv1d.weight");
    if(!ct){error="conv1d_weight_not_found";return false;}
    size_t conv_elems=1; for(auto d:ct->shape) conv_elems*=d;
    if(conv_elems!=8192u*4u){error="conv1d_shape";return false;}
    std::vector<float> cw; if(!read_vec(model_path,*ct,(uint32_t)conv_elems,cw,error)) return false;
    if(state.conv.size()!=8192u*4u) state.conv.assign(8192u*4u,0.0f);
    std::vector<float> conv(8192);
    for(uint32_t ch=0;ch<8192;ch++){
        float y=0.0f;
        float *hist=state.conv.data()+(size_t)ch*4;
        for(uint32_t j=0;j<3;j++) hist[j+1]=hist[j];
        hist[0]=qkv[ch];
        for(uint32_t j=0;j<4;j++) y+=hist[j]*cw[(size_t)ch*4+j];
        conv[ch]=y/(1.0f+std::exp(-y));
    }
    std::copy(conv.begin(),conv.begin()+2048,q.begin());
    std::copy(conv.begin()+2048,conv.begin()+4096,k.begin());
    std::copy(conv.begin()+4096,conv.end(),v.begin());

    if(!ornith35_deltanet_step(state,q.data(),k.data(),v.data(),beta.data(),decay.data(),16,32,128,128,core,error)) return false;
    const auto *nw=tx(info,b+"norm.weight");
    if(nw && nw->shape.size()==1 && nw->shape[0]==128) {
        if(!read_vec(model_path,*nw,128,normw,error)) return false;
    } else { error="delta_norm_weight_not_found"; return false; }
    for(uint32_t h=0;h<32;h++){
        float ss=0.0f; for(uint32_t d=0;d<128;d++){float x=core[h*128+d];ss+=x*x;}
        float inv=1.0f/std::sqrt(ss/128.0f+cfg.rms_norm_eps);
        for(uint32_t d=0;d<128;d++){float x=core[h*128+d]*inv*normw[d]; float g=z[h*128+d]; core[h*128+d]=x*(g/(1.0f+std::exp(-g)));}
    }
    if(!ornith35_run_projection_token(model_path,info,b+"out_proj.weight",core.data(),4096,out,4096,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    return true;
}

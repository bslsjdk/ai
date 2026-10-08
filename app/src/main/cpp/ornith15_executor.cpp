#include "ornith15_executor.h"
#include "ornith15_linear.h"
#include "mlx_safetensors.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

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
static bool read_flat_tensor(const std::string &path,
                             const MlxTensorInfo &t,
                             uint64_t elements,
                             std::vector<float> &out,
                             std::string &err) {
    uint64_t shape_elements=1;
    for(uint64_t d:t.shape) {
        if(!d || shape_elements>UINT64_MAX/d) { err="flat_tensor_shape"; return false; }
        shape_elements*=d;
    }
    if(shape_elements!=elements || !elements || elements>(uint64_t)SIZE_MAX/4u) {
        err="flat_tensor_shape";
        return false;
    }
    const size_t bytes=t.dtype=="F32" ? (size_t)elements*4u :
                       (t.dtype=="F16" || t.dtype=="BF16") ? (size_t)elements*2u : 0u;
    if(!bytes) { err="flat_tensor_dtype"; return false; }
    std::vector<unsigned char>b(bytes);
    if(!mlx_read_tensor_range(path,t,0,b.data(),bytes,err)) return false;
    out.resize((size_t)elements);
    if(t.dtype=="F32") std::memcpy(out.data(),b.data(),bytes);
    else if(t.dtype=="F16") for(uint64_t i=0;i<elements;i++) out[(size_t)i]=half_to_f(rd16(b.data()+2*i));
    else for(uint64_t i=0;i<elements;i++) out[(size_t)i]=bf16_to_f(rd16(b.data()+2*i));
    return true;
}

static const MlxTensorInfo *tx(const MlxSafetensorsInfo&i,const std::string&n){
    for(const auto&t:i.tensors) if(t.name==n) return &t; return nullptr;
}
static void zero_centered_rms(float *x, uint32_t n, const std::vector<float> &w, float eps) {
    float ss=0.0f;
    for(uint32_t i=0;i<n;i++) ss += x[i]*x[i];
    const float inv=1.0f/std::sqrt(ss/(float)n+eps);
    for(uint32_t i=0;i<n;i++) x[i]=x[i]*inv*(w.empty()?1.0f:w[i]);
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

bool ornith15_executor_validate(const MlxSafetensorsInfo &info,
                                 const Ornith15TextConfig &cfg,
                                 std::string &error) {
    if (!ornith15_validate_config(cfg, error)) return false;
    if (info.tensors.empty()) { error="empty_safetensors"; return false; }
    auto plan=ornith15_make_layer_plan();
    if(plan.size()!=cfg.num_layers){error="layer_plan_size_mismatch";return false;}
    uint32_t affine=0;
    for(const auto&t:info.tensors)
        if(t.dtype=="U32"&&t.shape.size()==2&&t.name.size()>=7&&
           t.name.compare(t.name.size()-7,7,".weight")==0) ++affine;
    if(!affine){error="no_affine4_weights";return false;}

    auto has=[&](const std::string &name)->bool {
        return find_tensor(info,name)!=nullptr;
    };
    auto check_affine=[&](const std::string &name,uint32_t rows,uint32_t cols)->bool {
        const MlxTensorInfo *w=find_tensor(info,name);
        if(!w || w->dtype!="U32" || w->shape.size()!=2 ||
           w->shape[0]!=rows || w->shape[1]*8ull!=cols) {
            error="affine_shape_mismatch="+name; return false;
        }
        const uint32_t groups=cols/64u;
        const MlxTensorInfo *sc=find_tensor(info,name.substr(0,name.size()-7)+".scales");
        const MlxTensorInfo *bi=find_tensor(info,name.substr(0,name.size()-7)+".biases");
        if(!sc || !bi || sc->shape.size()!=2 || sc->shape[0]!=rows ||
           sc->shape[1]!=groups || bi->shape!=sc->shape ||
           (sc->dtype!="F16" && sc->dtype!="BF16" && sc->dtype!="F32") ||
           bi->dtype!=sc->dtype) {
            error="affine_quant_shape_mismatch="+name; return false;
        }
        return true;
    };
    auto check_vec=[&](const std::string &name,uint32_t n)->bool {
        const MlxTensorInfo *t=find_tensor(info,name);
        if(!t || t->shape.size()!=1 || t->shape[0]!=n ||
           (t->dtype!="F16" && t->dtype!="BF16" && t->dtype!="F32")) {
            error="vector_shape_mismatch="+name; return false;
        }
        return true;
    };
    const std::string prefix="language_model.model.";
    const char *globalRequired[] = {
        "language_model.model.embed_tokens.weight",
        "language_model.model.embed_tokens.scales",
        "language_model.model.embed_tokens.biases",
        "language_model.lm_head.weight",
        "language_model.lm_head.scales",
        "language_model.lm_head.biases",
        "language_model.model.norm.weight"
    };
    for(const char *name:globalRequired) {
        if(!has(name)) { error=std::string("required_tensor_missing=")+name; return false; }
    }
    if(!check_affine("language_model.model.embed_tokens.weight",cfg.vocab_size,cfg.hidden_size)) return false;
    if(!check_affine("language_model.lm_head.weight",cfg.vocab_size,cfg.hidden_size)) return false;
    if(!check_vec("language_model.model.norm.weight",cfg.hidden_size)) return false;
    for(uint32_t i=0;i<cfg.num_layers;i++) {
        const std::string b=prefix+"layers."+std::to_string(i)+".";
        const std::string common[] = {
            b+"input_layernorm.weight",
            b+"post_attention_layernorm.weight",
            b+"mlp.gate_proj.weight", b+"mlp.gate_proj.scales", b+"mlp.gate_proj.biases",
            b+"mlp.up_proj.weight", b+"mlp.up_proj.scales", b+"mlp.up_proj.biases",
            b+"mlp.down_proj.weight", b+"mlp.down_proj.scales", b+"mlp.down_proj.biases"
        };
        for(const auto &name:common) {
            if(!has(name)) { error="required_tensor_missing="+name; return false; }
        }
        if(!check_vec(lb+"input_layernorm.weight",cfg.hidden_size)) return false;
        if(!check_vec(lb+"post_attention_layernorm.weight",cfg.hidden_size)) return false;
        if(!check_affine(lb+"mlp.gate_proj.weight",cfg.intermediate_size,cfg.hidden_size)) return false;
        if(!check_affine(lb+"mlp.up_proj.weight",cfg.intermediate_size,cfg.hidden_size)) return false;
        if(!check_affine(lb+"mlp.down_proj.weight",cfg.hidden_size,cfg.intermediate_size)) return false;
        if(plan[i].type==Ornith15LayerType::LinearAttention) {
            const std::string linear[] = {
                b+"linear_attn.in_proj_qkv.weight", b+"linear_attn.in_proj_qkv.scales", b+"linear_attn.in_proj_qkv.biases",
                b+"linear_attn.in_proj_z.weight", b+"linear_attn.in_proj_z.scales", b+"linear_attn.in_proj_z.biases",
                b+"linear_attn.in_proj_b.weight", b+"linear_attn.in_proj_b.scales", b+"linear_attn.in_proj_b.biases",
                b+"linear_attn.in_proj_a.weight", b+"linear_attn.in_proj_a.scales", b+"linear_attn.in_proj_a.biases",
                b+"linear_attn.A_log", b+"linear_attn.dt_bias", b+"linear_attn.conv1d.weight",
                b+"linear_attn.norm.weight",
                b+"linear_attn.out_proj.weight", b+"linear_attn.out_proj.scales", b+"linear_attn.out_proj.biases"
            };
            for(const auto &name:linear) {
                if(!has(name)) { error="required_tensor_missing="+name; return false; }
            }
            if(!check_affine(b+"linear_attn.in_proj_qkv.weight",8192u,cfg.hidden_size)) return false;
            if(!check_affine(b+"linear_attn.in_proj_z.weight",4096u,cfg.hidden_size)) return false;
            if(!check_affine(b+"linear_attn.in_proj_b.weight",32u,cfg.hidden_size)) return false;
            if(!check_affine(b+"linear_attn.in_proj_a.weight",32u,cfg.hidden_size)) return false;
            if(!check_affine(b+"linear_attn.out_proj.weight",cfg.hidden_size,4096u)) return false;
            if(!check_vec(b+"linear_attn.A_log",32u)) return false;
            if(!check_vec(b+"linear_attn.dt_bias",32u)) return false;
            if(!check_vec(b+"linear_attn.norm.weight",128u)) return false;
        } else {
            const std::string attn[] = {
                b+"self_attn.q_proj.weight", b+"self_attn.q_proj.scales", b+"self_attn.q_proj.biases",
                b+"self_attn.k_proj.weight", b+"self_attn.k_proj.scales", b+"self_attn.k_proj.biases",
                b+"self_attn.v_proj.weight", b+"self_attn.v_proj.scales", b+"self_attn.v_proj.biases",
                b+"self_attn.o_proj.weight", b+"self_attn.o_proj.scales", b+"self_attn.o_proj.biases",
                b+"self_attn.q_norm.weight", b+"self_attn.k_norm.weight"
            };
            for(const auto &name:attn) {
                if(!has(name)) { error="required_tensor_missing="+name; return false; }
            }
            if(!check_affine(b+"self_attn.q_proj.weight",8192u,cfg.hidden_size)) return false;
            if(!check_affine(b+"self_attn.k_proj.weight",1024u,cfg.hidden_size)) return false;
            if(!check_affine(b+"self_attn.v_proj.weight",1024u,cfg.hidden_size)) return false;
            if(!check_affine(b+"self_attn.o_proj.weight",cfg.hidden_size,4096u)) return false;
            if(!check_vec(b+"self_attn.q_norm.weight",256u)) return false;
            if(!check_vec(b+"self_attn.k_norm.weight",256u)) return false;
        }
    }
    return true;
}

bool ornith15_executor_greedy_step(const std::string &model_path,
                                    const MlxSafetensorsInfo &info,
                                    const Ornith15TextConfig &cfg,
                                    uint32_t token_id,
                                    Ornith15DecoderStep &step,
                                    std::string &error) {
    if(!ornith15_executor_validate(info,cfg,error)) return false;
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


bool ornith15_executor_init_runtime(const Ornith15TextConfig &cfg,
                                     uint32_t max_attention_tokens,
                                     Ornith15LayerRuntime &runtime,
                                     std::string &error) {
    if (!ornith15_validate_config(cfg, error)) return false;
    if (!max_attention_tokens) { error = "max_attention_tokens_zero"; return false; }
    // Never preallocate the full 262K attention cache. The runtime is deliberately
    // bounded; a paged KV cache will extend context later without blowing RAM.
    const uint32_t bounded_attention_tokens = std::min(max_attention_tokens, 4096u);
    runtime.delta.clear();
    runtime.attention.clear();
    runtime.delta.resize(cfg.num_layers);
    runtime.attention.resize(cfg.num_layers);
    for (uint32_t i = 0; i < cfg.num_layers; ++i) {
        const auto plan = ornith15_make_layer_plan();
        if (plan[i].type == Ornith15LayerType::LinearAttention) {
            if (!ornith15_deltanet_init(runtime.delta[i], 32u * 128u, cfg.linear_conv_kernel)) {
                error = "deltanet_state_init_failed";
                return false;
            }
        } else {
            if (!ornith15_attention_init(runtime.attention[i], bounded_attention_tokens)) {
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

bool ornith15_executor_forward_token(const std::string &model_path,
                                       const MlxSafetensorsInfo &info,
                                       const Ornith15TextConfig &cfg,
                                       uint32_t token_id,
                                       uint32_t position,
                                       Ornith15LayerRuntime &runtime,
                                       Ornith15DecoderStep &step,
                                       std::string &error) {
    if (token_id >= cfg.vocab_size) { error="token_id_oob"; return false; }
    if (runtime.initialized_layers != cfg.num_layers) {
        if (!ornith15_executor_init_runtime(cfg, std::min<uint32_t>(cfg.context_length, 4096u), runtime, error))
            return false;
    }

    const std::string prefix="language_model.model.";
    const MlxTensorInfo *emb=find_tensor(info,prefix+"embed_tokens.weight");
    if(!emb || emb->shape.size()!=2 || emb->shape[0]!=cfg.vocab_size ||
       emb->shape[1]*8ull!=cfg.hidden_size) {
        error="embedding_shape_mismatch";
        return false;
    }

    std::vector<float> hidden;
    if(!mlx_read_affine4_row(model_path,info,emb->name,token_id,hidden,error)) return false;
    if(hidden.size()!=cfg.hidden_size){ error="embedding_decode_size"; return false; }

    const auto plan=ornith15_make_layer_plan();
    Ornith15ExecutorStats stats;
    for(uint32_t i=0;i<cfg.num_layers;i++) {
        const std::string lb="language_model.model.layers."+std::to_string(i)+".";
        const MlxTensorInfo *iw=tx(info,lb+"input_layernorm.weight");
        const MlxTensorInfo *pw=tx(info,lb+"post_attention_layernorm.weight");
        if(!iw || !pw) { error="layer_norm_weight_missing_"+std::to_string(i); return false; }

        std::vector<float> normw;
        if(!read_vec(model_path,*iw,cfg.hidden_size,normw,error)) return false;
        std::vector<float> residual=hidden;
        zero_centered_rms(hidden.data(),cfg.hidden_size,normw,cfg.rms_norm_eps);

        std::fill(runtime.work_b.begin(),runtime.work_b.end(),0.0f);
        bool ok=false;
        if (plan[i].type == Ornith15LayerType::LinearAttention) {
            ok=ornith15_executor_run_delta_layer(model_path,info,cfg,i,hidden.data(),
                                                 runtime.work_b.data(),runtime.delta[i],
                                                 stats,error);
        } else {
            ok=ornith15_executor_run_attention_layer(model_path,info,cfg,i,hidden.data(),
                                                      runtime.work_b.data(),runtime.attention[i],
                                                      position,stats,error);
        }
        if(!ok) return false;

        for(uint32_t d=0;d<cfg.hidden_size;d++) hidden[d]=residual[d]+runtime.work_b[d];

        residual=hidden;
        if(!read_vec(model_path,*pw,cfg.hidden_size,normw,error)) return false;
        zero_centered_rms(hidden.data(),cfg.hidden_size,normw,cfg.rms_norm_eps);
        std::fill(runtime.work_b.begin(),runtime.work_b.end(),0.0f);
        if(!ornith15_executor_apply_mlp(model_path,info,cfg,i,hidden.data(),
                                         runtime.work_b.data(),stats,error)) return false;
        for(uint32_t d=0;d<cfg.hidden_size;d++) hidden[d]=residual[d]+runtime.work_b[d];
        stats.layers_done=i+1;
    }

    const MlxTensorInfo *fw=tx(info,prefix+"norm.weight");
    if(!fw) { error="final_norm_weight_missing"; return false; }
    std::vector<float> finalw;
    if(!read_vec(model_path,*fw,cfg.hidden_size,finalw,error)) return false;
    zero_centered_rms(hidden.data(),cfg.hidden_size,finalw,cfg.rms_norm_eps);

    const std::string lm_name="language_model.lm_head.weight";
    step.logits.resize(cfg.vocab_size);
    Ornith15ProjectionStats ps;
    if(!ornith15_run_projection_token(model_path,info,lm_name,hidden.data(),cfg.hidden_size,
                                      step.logits.data(),cfg.vocab_size,ps)) {
        error=ps.status;
        return false;
    }
    stats.npu_calls+=ps.npu_calls;
    step.hidden=std::move(hidden);
    step.position=position;
    step.stats=stats;
    step.stats.status="OK ORNITH15_FULL_TOKEN_FORWARD";
    return true;
}

bool ornith15_executor_apply_mlp(const std::string &model_path,
                                  const MlxSafetensorsInfo &info,
                                  const Ornith15TextConfig &cfg,
                                  uint32_t layer_index,
                                  const float *hidden,
                                  float *out,
                                  Ornith15ExecutorStats &stats,
                                  std::string &error) {
    if (!hidden || !out || layer_index >= cfg.num_layers) { error = "mlp_args"; return false; }
    const std::string base = "language_model.model.layers." + std::to_string(layer_index) + ".mlp.";
    Ornith15MlpStats ms;
    if (!ornith15_run_mlp(model_path, hidden, out, cfg.hidden_size, cfg.intermediate_size, base, info, ms)) {
        error = ms.status.empty() ? "mlp_failed" : ms.status;
        return false;
    }
    stats.npu_calls += ms.projection_calls;
    return true;
}


bool ornith15_executor_run_attention_layer(const std::string &model_path,
                                           const MlxSafetensorsInfo &info,
                                           const Ornith15TextConfig &cfg,
                                           uint32_t layer_index,
                                           const float *hidden,
                                           float *out,
                                           Ornith15AttentionState &state,
                                           uint32_t position,
                                           Ornith15ExecutorStats &stats,
                                           std::string &error) {
    if(!hidden||!out||layer_index>=cfg.num_layers){error="attention_args";return false;}
    const std::string b="language_model.model.layers."+std::to_string(layer_index)+".self_attn.";
    std::vector<float> q(8192),k(1024),v(1024),qnorm(256),knorm(256),att(4096),gate(4096);
    Ornith15ProjectionStats ps;
    if(!ornith15_run_projection_token(model_path,info,b+"q_proj.weight",hidden,4096,q.data(),8192,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith15_run_projection_token(model_path,info,b+"k_proj.weight",hidden,4096,k.data(),1024,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith15_run_projection_token(model_path,info,b+"v_proj.weight",hidden,4096,v.data(),1024,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    const auto *qt=tx(info,b+"q_norm.weight"),*kt=tx(info,b+"k_norm.weight");
    if(!qt||!kt||!read_vec(model_path,*qt,256,qnorm,error)||!read_vec(model_path,*kt,256,knorm,error)) return false;
    if(!ornith15_attention_step(state,q.data(),k.data(),v.data(),16,4,256,position,10000000.0f,att,error,qnorm.data(),knorm.data())) return false;
    for(uint32_t i=0;i<4096;i++) gate[i]=q[4096+i];
    for(uint32_t i=0;i<4096;i++) att[i]*=1.0f/(1.0f+std::exp(-gate[i]));
    if(!ornith15_run_projection_token(model_path,info,b+"o_proj.weight",att.data(),4096,out,4096,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    return true;
}


bool ornith15_executor_run_delta_layer(const std::string &model_path,
                                       const MlxSafetensorsInfo &info,
                                       const Ornith15TextConfig &cfg,
                                       uint32_t layer_index,
                                       const float *hidden,
                                       float *out,
                                       Ornith15DeltaState &state,
                                       Ornith15ExecutorStats &stats,
                                       std::string &error) {
    if(!hidden||!out||layer_index>=cfg.num_layers){error="delta_args";return false;}
    const std::string b="language_model.model.layers."+std::to_string(layer_index)+".linear_attn.";
    std::vector<float> qkv(8192),z(4096),ba(64),q(2048),k(2048),v(4096),beta(32),a(32),decay(32),core(4096),normw(128);
    Ornith15ProjectionStats ps;
    if(!ornith15_run_projection_token(model_path,info,b+"in_proj_qkv.weight",hidden,4096,qkv.data(),8192,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith15_run_projection_token(model_path,info,b+"in_proj_z.weight",hidden,4096,z.data(),4096,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith15_run_projection_token(model_path,info,b+"in_proj_b.weight",hidden,4096,beta.data(),32,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith15_run_projection_token(model_path,info,b+"in_proj_a.weight",hidden,4096,a.data(),32,ps)){error=ps.status;return false;}
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
    std::vector<float> cw;
    if(!read_flat_tensor(model_path,*ct,(uint64_t)conv_elems,cw,error)) return false;
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

    if(!ornith15_deltanet_step(state,q.data(),k.data(),v.data(),beta.data(),decay.data(),16,32,128,128,core,error)) return false;
    const auto *nw=tx(info,b+"norm.weight");
    if(nw && nw->shape.size()==1 && nw->shape[0]==128) {
        if(!read_vec(model_path,*nw,128,normw,error)) return false;
    } else { error="delta_norm_weight_not_found"; return false; }
    for(uint32_t h=0;h<32;h++){
        float ss=0.0f; for(uint32_t d=0;d<128;d++){float x=core[h*128+d];ss+=x*x;}
        float inv=1.0f/std::sqrt(ss/128.0f+cfg.rms_norm_eps);
        for(uint32_t d=0;d<128;d++){float x=core[h*128+d]*inv*normw[d]; float g=z[h*128+d]; core[h*128+d]=x*(g/(1.0f+std::exp(-g)));}
    }
    if(!ornith15_run_projection_token(model_path,info,b+"out_proj.weight",core.data(),4096,out,4096,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    return true;
}

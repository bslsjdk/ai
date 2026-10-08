#include "ornith15_executor.h"
#include "ornith15_linear.h"
#include "mlx_safetensors.h"
#include "runtime_memory_budget.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace {
static uint16_t rd16(const unsigned char *p){ return (uint16_t)p[0] | ((uint16_t)p[1]<<8); }
static float half_to_f(uint16_t h){
    const uint32_t s=(h>>15)&1u, e=(h>>10)&31u, f=h&1023u;
    uint32_t out=0;
    if(e==0){
        if(f==0) {
            out=s<<31;
        } else {
            // Normalize the binary16 subnormal rather than silently flushing it
            // to zero. Static Q/K norms and recurrent parameters are small enough
            // that this distinction is worth preserving.
            uint32_t mant=f;
            int32_t exp=-14;
            while((mant&0x400u)==0){ mant<<=1; --exp; }
            mant&=0x3ffu;
            out=(s<<31)|((uint32_t)(exp+127)<<23)|(mant<<13);
        }
    } else if(e==31) {
        out=(s<<31)|0x7f800000u|(f<<13);
    } else {
        out=(s<<31)|((e-15+127)<<23)|(f<<13);
    }
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
    for(uint32_t i=0;i<n;i++) {
        // Qwen3.5RMSNorm is zero-centered: weight is initialized at zero
        // and applied as (1 + weight), not plain weight.
        const float gain = w.empty() ? 1.0f : (1.0f + w[i]);
        x[i]=x[i]*inv*gain;
    }
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
        if(!check_vec(b+"input_layernorm.weight",cfg.hidden_size)) return false;
        if(!check_vec(b+"post_attention_layernorm.weight",cfg.hidden_size)) return false;
        if(!check_affine(b+"mlp.gate_proj.weight",cfg.intermediate_size,cfg.hidden_size)) return false;
        if(!check_affine(b+"mlp.up_proj.weight",cfg.intermediate_size,cfg.hidden_size)) return false;
        if(!check_affine(b+"mlp.down_proj.weight",cfg.hidden_size,cfg.intermediate_size)) return false;
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


bool ornith15_executor_prepare_static_weights(const std::string &model_path,
                                              const MlxSafetensorsInfo &info,
                                              const Ornith15TextConfig &cfg,
                                              Ornith15LayerRuntime &runtime,
                                              std::string &error) {
    if (!ornith15_validate_config(cfg, error)) return false;
    runtime.static_weights.clear();
    runtime.static_weights.resize(cfg.num_layers);
    runtime.final_norm.clear();

    const auto plan = ornith15_make_layer_plan();
    for (uint32_t i = 0; i < cfg.num_layers; ++i) {
        const std::string b = "language_model.model.layers." + std::to_string(i) + ".";
        auto &w = runtime.static_weights[i];

        const auto *input_norm = tx(info, b + "input_layernorm.weight");
        const auto *post_norm = tx(info, b + "post_attention_layernorm.weight");
        if (!input_norm || !post_norm ||
            !read_vec(model_path, *input_norm, cfg.hidden_size, w.input_norm, error) ||
            !read_vec(model_path, *post_norm, cfg.hidden_size, w.post_norm, error)) {
            return false;
        }

        if (plan[i].type == Ornith15LayerType::FullAttention) {
            const auto *qn = tx(info, b + "self_attn.q_norm.weight");
            const auto *kn = tx(info, b + "self_attn.k_norm.weight");
            if (!qn || !kn ||
                !read_vec(model_path, *qn, cfg.head_dim, w.q_norm, error) ||
                !read_vec(model_path, *kn, cfg.head_dim, w.k_norm, error)) {
                return false;
            }
        } else {
            const auto *al = tx(info, b + "linear_attn.A_log");
            const auto *dt = tx(info, b + "linear_attn.dt_bias");
            const auto *nw = tx(info, b + "linear_attn.norm.weight");
            const auto *ct = tx(info, b + "linear_attn.conv1d.weight");
            if (!al || !dt || !nw || !ct ||
                !read_vec(model_path, *al, 32u, w.a_log, error) ||
                !read_vec(model_path, *dt, 32u, w.dt_bias, error) ||
                !read_vec(model_path, *nw, 128u, w.delta_norm, error)) {
                return false;
            }
            uint64_t conv_elems = 1;
            for (uint64_t d : ct->shape) {
                if (!d || conv_elems > UINT64_MAX / d) {
                    error = "static_conv_shape_overflow_" + std::to_string(i);
                    return false;
                }
                conv_elems *= d;
            }
            if (conv_elems != 8192ull * 4ull) {
                error = "static_conv_shape_" + std::to_string(i);
                return false;
            }
            if (!read_flat_tensor(model_path, *ct, conv_elems, w.conv1d, error)) return false;
        }
    }

    const auto *fw = tx(info, "language_model.model.norm.weight");
    if (!fw || !read_vec(model_path, *fw, cfg.hidden_size, runtime.final_norm, error))
        return false;
    return true;
}

uint64_t ornith15_effective_attention_tokens(const Ornith15TextConfig &cfg,
                                                uint64_t requested_tokens,
                                                std::string &diagnostic) {
    const auto plan = ornith15_make_layer_plan();
    uint32_t full_layers = 0;
    for (const auto &layer : plan)
        if (layer.type == Ornith15LayerType::FullAttention) ++full_layers;
    const uint64_t delta_bytes =
        (uint64_t)(cfg.num_layers - full_layers) * 32ull * 128ull * 128ull * sizeof(float);
    return ornith15_memory_effective_attention_tokens(
        requested_tokens, cfg.num_kv_heads, cfg.head_dim, full_layers, delta_bytes, diagnostic);
}

bool ornith15_executor_init_runtime(const Ornith15TextConfig &cfg,
                                     uint32_t max_attention_tokens,
                                     Ornith15LayerRuntime &runtime,
                                     std::string &error) {
    if (!ornith15_validate_config(cfg, error)) return false;
    if (!max_attention_tokens) { error = "max_attention_tokens_zero"; return false; }

    std::string plan_diag;
    const uint64_t safe_tokens = ornith15_effective_attention_tokens(
        cfg, max_attention_tokens, plan_diag);
    if (!safe_tokens) {
        error = "runtime_memory_plan=" + plan_diag;
        return false;
    }
    const uint32_t bounded_attention_tokens = (uint32_t)std::min<uint64_t>(
        max_attention_tokens, safe_tokens);

    const auto memory_plan = ornith15_make_layer_plan();
    uint32_t full_layers = 0;
    for (const auto &layer : memory_plan)
        if (layer.type == Ornith15LayerType::FullAttention) ++full_layers;

    const uint64_t kv_bytes =
        (uint64_t)bounded_attention_tokens * cfg.num_kv_heads * cfg.head_dim * 2ull * 2ull *
        full_layers;
    const uint64_t delta_bytes =
        (uint64_t)(cfg.num_layers - full_layers) * 32ull * 128ull * 128ull * sizeof(float);

    // This is a model-state budget only. The separate RSS guard reserves
    // additional space for QNN, temporary tensors, tokenizer and Java/native code.
    if (kv_bytes + delta_bytes > ORNITH15_MODEL_STATE_BUDGET_BYTES) {
        error = "runtime_state_budget_exceeded kv_bytes=" + std::to_string(kv_bytes) +
                " delta_bytes=" + std::to_string(delta_bytes) +
                " state_budget_bytes=" + std::to_string(ORNITH15_MODEL_STATE_BUDGET_BYTES);
        return false;
    }

    runtime.delta.clear();
    runtime.attention.clear();
    runtime.static_weights.clear();
    runtime.final_norm.clear();
    runtime.delta.resize(cfg.num_layers);
    runtime.attention.resize(cfg.num_layers);
    const auto plan = ornith15_make_layer_plan();
    for (uint32_t i = 0; i < cfg.num_layers; ++i) {
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

    const uint64_t work_bytes = (uint64_t)cfg.hidden_size * 3ull * sizeof(float);
    if (!ornith15_memory_headroom(work_bytes, "executor_work_vectors", error)) return false;
    runtime.work_a.resize(cfg.hidden_size);
    runtime.work_b.resize(cfg.hidden_size);
    runtime.work_c.resize(cfg.hidden_size);
    if (!ornith15_memory_within_limit(error)) return false;
    runtime.initialized_layers = cfg.num_layers;
    return true;
}

void ornith15_executor_reset_runtime(Ornith15LayerRuntime &runtime) {
    for (auto &s : runtime.delta) {
        std::fill(s.state.begin(), s.state.end(), 0.0f);
        std::fill(s.conv.begin(), s.conv.end(), 0.0f);
        s.tokens = 0;
    }
    for (auto &s : runtime.attention) {
        // Attention slots are valid only inside [first_position, position].
        // Starting a new replay merely resets the logical count; wiping up to
        // 2 GiB of FP16 KV would waste seconds of memory bandwidth and is not
        // required for correctness because old slots are never read.
        s.tokens = 0;
    }
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
        if (!ornith15_executor_init_runtime(cfg, std::min<uint32_t>(cfg.context_length, 65536u), runtime, error))
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
        (void)iw; (void)pw;
        if(i >= runtime.static_weights.size() ||
           runtime.static_weights[i].input_norm.size()!=cfg.hidden_size ||
           runtime.static_weights[i].post_norm.size()!=cfg.hidden_size) {
            error="layer_static_norm_missing_"+std::to_string(i); return false;
        }

        std::vector<float> residual=hidden;
        zero_centered_rms(hidden.data(),cfg.hidden_size,
                          runtime.static_weights[i].input_norm,cfg.rms_norm_eps);

        std::fill(runtime.work_b.begin(),runtime.work_b.end(),0.0f);
        if (i >= runtime.static_weights.size()) {
            error="static_weights_missing_"+std::to_string(i); return false;
        }
        bool ok=false;
        if (plan[i].type == Ornith15LayerType::LinearAttention) {
            ok=ornith15_executor_run_delta_layer(model_path,info,cfg,i,hidden.data(),
                                                 runtime.work_b.data(),runtime.delta[i],
                                                 runtime.static_weights[i],stats,error);
        } else {
            ok=ornith15_executor_run_attention_layer(model_path,info,cfg,i,hidden.data(),
                                                      runtime.work_b.data(),runtime.attention[i],
                                                      runtime.static_weights[i],position,stats,error);
        }
        if(!ok) return false;

        for(uint32_t d=0;d<cfg.hidden_size;d++) hidden[d]=residual[d]+runtime.work_b[d];

        (void)pw;
        residual=hidden;
        zero_centered_rms(hidden.data(),cfg.hidden_size,
                          runtime.static_weights[i].post_norm,cfg.rms_norm_eps);
        std::fill(runtime.work_b.begin(),runtime.work_b.end(),0.0f);
        if(!ornith15_executor_apply_mlp(model_path,info,cfg,i,hidden.data(),
                                         runtime.work_b.data(),stats,error)) return false;
        for(uint32_t d=0;d<cfg.hidden_size;d++) hidden[d]=residual[d]+runtime.work_b[d];
        stats.layers_done=i+1;
    }

    if(runtime.final_norm.size()!=cfg.hidden_size) { error="final_norm_static_missing"; return false; }
    zero_centered_rms(hidden.data(),cfg.hidden_size,runtime.final_norm,cfg.rms_norm_eps);

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
                                           const Ornith15LayerStaticWeights &static_weights,
                                           uint32_t position,
                                           Ornith15ExecutorStats &stats,
                                           std::string &error) {
    if(!hidden||!out||layer_index>=cfg.num_layers){error="attention_args";return false;}
    const std::string b="language_model.model.layers."+std::to_string(layer_index)+".self_attn.";
    std::vector<float> qg(8192),q(4096),k(1024),v(1024),att(4096),gate(4096);
    Ornith15ProjectionStats ps;
    if(!ornith15_run_projection_token(model_path,info,b+"q_proj.weight",hidden,4096,qg.data(),8192,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    // Qwen3.5 stores q_proj output interleaved per head: [Q_256, gate_256]
    // for each of the 16 heads. Deinterleave before q/k attention.
    for(uint32_t h=0;h<16;h++){
        std::copy(qg.begin()+(size_t)h*512,
                  qg.begin()+(size_t)h*512+256,
                  q.begin()+(size_t)h*256);
        std::copy(qg.begin()+(size_t)h*512+256,
                  qg.begin()+(size_t)h*512+512,
                  gate.begin()+(size_t)h*256);
    }
    if(!ornith15_run_projection_token(model_path,info,b+"k_proj.weight",hidden,4096,k.data(),1024,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(!ornith15_run_projection_token(model_path,info,b+"v_proj.weight",hidden,4096,v.data(),1024,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    if(static_weights.q_norm.size()!=256u || static_weights.k_norm.size()!=256u) {
        error="attention_static_norm_missing"; return false;
    }
    if(!ornith15_attention_step(state,q.data(),k.data(),v.data(),16,4,256,position,
                                 10000000.0f,att,error,
                                 static_weights.q_norm.data(),static_weights.k_norm.data())) return false;
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
                                       const Ornith15LayerStaticWeights &static_weights,
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

    if(static_weights.a_log.size()!=32u || static_weights.dt_bias.size()!=32u ||
       static_weights.delta_norm.size()!=128u || static_weights.conv1d.size()!=8192u*4u) {
        error="delta_static_weights_missing"; return false;
    }
    std::vector<float> decay(32);
    const std::vector<float> &decayBase=static_weights.a_log;
    const std::vector<float> &dtv=static_weights.dt_bias;
    for(uint32_t i=0;i<32;i++){
        beta[i]=1.0f/(1.0f+std::exp(-beta[i]));
        const float x=a[i]+dtv[i];
        // Stable softplus without the previous hard cap at 20.
        const float sp=x>20.0f ? x : std::log1p(std::exp(x));
        decay[i]=-std::exp(decayBase[i])*sp;
    }

    const std::vector<float> &cw=static_weights.conv1d;
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
    const std::vector<float> &normw=static_weights.delta_norm;
    for(uint32_t h=0;h<32;h++){
        float ss=0.0f; for(uint32_t d=0;d<128;d++){float x=core[h*128+d];ss+=x*x;}
        float inv=1.0f/std::sqrt(ss/128.0f+cfg.rms_norm_eps);
        for(uint32_t d=0;d<128;d++){float x=core[h*128+d]*inv*normw[d]; float g=z[h*128+d]; const float sig=1.0f/(1.0f+std::exp(-g)); core[h*128+d]=x*(g*sig);}
    }
    if(!ornith15_run_projection_token(model_path,info,b+"out_proj.weight",core.data(),4096,out,4096,ps)){error=ps.status;return false;}
    stats.npu_calls+=ps.npu_calls;
    return true;
}

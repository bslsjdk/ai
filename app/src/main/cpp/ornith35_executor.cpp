#include "ornith35_executor.h"
#include "mlx_safetensors.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
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
    step.position++;
    step.stats.layers_done=0;
    step.stats.status="EMBEDDING_READY_EXECUTOR_NEXT";
    return true;
}

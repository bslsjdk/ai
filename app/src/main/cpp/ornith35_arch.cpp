#include "ornith35_arch.h"
#include <utility>

std::vector<Ornith35LayerPlan> ornith35_make_layer_plan() {
    std::vector<Ornith35LayerPlan> plan;
    plan.reserve(32);
    for (uint32_t i = 0; i < 32; ++i) {
        const bool full = ((i + 1u) % 4u) == 0u;
        Ornith35LayerPlan p;
        p.index = i;
        p.type = full ? Ornith35LayerType::FullAttention
                      : Ornith35LayerType::LinearAttention;
        p.prefix = "language_model.model.layers." + std::to_string(i) + ".";
        plan.push_back(std::move(p));
    }
    return plan;
}

const char * ornith35_layer_type_name(Ornith35LayerType type) {
    return type == Ornith35LayerType::FullAttention
        ? "full_attention" : "linear_attention";
}

bool ornith35_validate_config(const Ornith35TextConfig & c, std::string & error) {
    if (c.hidden_size != 4096 || c.intermediate_size != 12288 ||
        c.num_layers != 32 || c.num_attention_heads != 16 ||
        c.num_kv_heads != 4 || c.head_dim != 256 ||
        c.linear_key_heads != 16 || c.linear_value_heads != 32 ||
        c.linear_key_head_dim != 128 || c.linear_value_head_dim != 128 ||
        c.linear_conv_kernel != 4 || c.vocab_size != 248320 ||
        c.context_length != 262144) {
        error = "unsupported_ornith35_text_config";
        return false;
    }
    if (c.rms_norm_eps <= 0.0f) {
        error = "invalid_rms_norm_eps";
        return false;
    }
    return true;
}

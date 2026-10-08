#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class Ornith35LayerType : uint8_t {
    LinearAttention = 0,
    FullAttention = 1,
};

struct Ornith35TextConfig {
    uint32_t hidden_size = 4096;
    uint32_t intermediate_size = 12288;
    uint32_t num_layers = 32;
    uint32_t num_attention_heads = 16;
    uint32_t num_kv_heads = 4;
    uint32_t head_dim = 256;
    uint32_t linear_key_heads = 16;
    uint32_t linear_value_heads = 32;
    uint32_t linear_key_head_dim = 128;
    uint32_t linear_value_head_dim = 128;
    uint32_t linear_conv_kernel = 4;
    uint32_t vocab_size = 248320;
    uint32_t context_length = 262144;
    float rms_norm_eps = 1.0e-6f;
};

struct Ornith35LayerPlan {
    uint32_t index = 0;
    Ornith35LayerType type = Ornith35LayerType::LinearAttention;
    std::string prefix;
};

std::vector<Ornith35LayerPlan> ornith35_make_layer_plan();
const char * ornith35_layer_type_name(Ornith35LayerType type);
bool ornith35_validate_config(const Ornith35TextConfig & cfg, std::string & error);

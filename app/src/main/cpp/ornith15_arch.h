#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class Ornith15LayerType { LinearAttention, FullAttention };

struct Ornith15LayerPlan {
    uint32_t index = 0;
    Ornith15LayerType type = Ornith15LayerType::LinearAttention;
    std::string prefix;
};

struct Ornith15TextConfig {
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
    uint32_t rope_theta = 10000000;
    uint32_t rotary_dim = 64;
    bool attention_output_gate = true;
};

std::vector<Ornith15LayerPlan> ornith15_make_layer_plan();
const char *ornith15_layer_type_name(Ornith15LayerType type);
bool ornith15_validate_config(const Ornith15TextConfig &cfg, std::string &error);

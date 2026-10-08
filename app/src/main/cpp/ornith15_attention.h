#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct Ornith15AttentionState {
    uint32_t q_heads = 16;
    uint32_t kv_heads = 4;
    uint32_t head_dim = 256;
    uint64_t tokens = 0;
    std::vector<float> keys;
    std::vector<float> values;
};

bool ornith15_attention_init(Ornith15AttentionState &s, uint32_t max_tokens);
bool ornith15_attention_step(
    Ornith15AttentionState &s,
    const float *q, const float *k, const float *v,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t position,
    float rope_theta,
    std::vector<float> &out,
    std::string &error,
    const float *q_norm_weight = nullptr,
    const float *k_norm_weight = nullptr);

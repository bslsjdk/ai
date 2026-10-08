#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct Ornith15AttentionState {
    uint32_t q_heads = 16;
    uint32_t kv_heads = 4;
    uint32_t head_dim = 256;
    uint64_t tokens = 0;
    // The 64K K/V range is reserved virtually, while physical pages are
    // faulted in only as tokens are actually written.
    uint32_t window_tokens = 0;
    uint64_t mapped_elems = 0;
    uint16_t *keys = nullptr;
    uint16_t *values = nullptr;

    Ornith15AttentionState() = default;
    ~Ornith15AttentionState();

    Ornith15AttentionState(const Ornith15AttentionState&) = delete;
    Ornith15AttentionState& operator=(const Ornith15AttentionState&) = delete;
    Ornith15AttentionState(Ornith15AttentionState&& other) noexcept;
    Ornith15AttentionState& operator=(Ornith15AttentionState&& other) noexcept;
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

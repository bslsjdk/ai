#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct Ornith15DeltaState {
    uint32_t key_heads = 16;
    uint32_t value_heads = 32;
    uint32_t key_dim = 128;
    uint32_t value_dim = 128;
    // One recurrent matrix per value head: [value_heads, key_dim, value_dim].
    // This is the exact bounded-memory state used by the recurrent delta rule.
    std::vector<float> state;
    // Causal-convolution history is kept separately by the projection/runtime.
    std::vector<float> conv;
    uint64_t tokens = 0;
};

bool ornith15_deltanet_init(Ornith15DeltaState &s, uint32_t conv_channels = 0, uint32_t conv_kernel = 4);

bool ornith15_deltanet_step(
    Ornith15DeltaState &s,
    const float *q, const float *k, const float *v,
    const float *beta, const float *decay_log,
    uint32_t heads_qk, uint32_t heads_v,
    uint32_t key_dim, uint32_t value_dim,
    std::vector<float> &out,
    std::string &error);

bool ornith15_deltanet_conv_step(Ornith15DeltaState &s,
                                  const float *input,
                                  uint32_t channels,
                                  uint32_t kernel,
                                  std::vector<float> &output,
                                  std::string &error);

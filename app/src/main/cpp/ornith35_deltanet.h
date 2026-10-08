#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct Ornith35DeltaState {
    uint32_t key_heads = 16;
    uint32_t value_heads = 32;
    uint32_t key_dim = 128;
    uint32_t value_dim = 128;
    std::vector<float> state;
    std::vector<float> conv;
    uint64_t tokens = 0;
};

bool ornith35_deltanet_init(Ornith35DeltaState &s);
bool ornith35_deltanet_step(Ornith35DeltaState &s,
                            const float *q, const float *k,
                            const float *v, const float *gate,
                            uint32_t heads_qk, uint32_t heads_v,
                            uint32_t key_dim, uint32_t value_dim,
                            std::vector<float> &out,
                            std::string &error);

#pragma once
#include <cstdint>
#include <string>
#include "ornith35_arch.h"
#include "mlx_safetensors.h"

struct Ornith35ProjectionStats {
    uint64_t tiles = 0;
    uint64_t rows = 0;
    uint64_t cols = 0;
    uint64_t npu_calls = 0;
    std::string status;
};

bool ornith35_run_projection_tile(
    const std::string & model_path,
    const MlxSafetensorsInfo & info,
    const std::string & weight_name,
    const float * input,
    uint32_t m,
    uint32_t k,
    float * output,
    uint32_t n,
    Ornith35ProjectionStats & stats);

/*
 * Decode-time projection. MCNPU HTP uses bucketed M=32, so a single-token
 * vector is zero-padded to one 32-row bucket and only row zero is returned.
 */
bool ornith35_run_projection_token(
    const std::string & model_path,
    const MlxSafetensorsInfo & info,
    const std::string & weight_name,
    const float * input,
    uint32_t k,
    float * output,
    uint32_t n,
    Ornith35ProjectionStats & stats);

struct Ornith35MlpStats {
    uint32_t projection_calls = 0;
    bool npu = false;
    std::string status;
};

bool ornith35_run_mlp(const std::string &model_path, const float *hidden, float *out, uint32_t hidden_size,
                      uint32_t intermediate_size, const std::string &layer_prefix,
                      const MlxSafetensorsInfo &model, Ornith35MlpStats &stats);

#pragma once
#include <cstdint>
#include <string>
#include "ornith35_arch.h"

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

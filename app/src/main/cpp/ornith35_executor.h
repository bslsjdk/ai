#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "mlx_safetensors.h"
#include "ornith35_arch.h"

struct Ornith35ExecutorStats {
    uint32_t layers_done = 0;
    uint32_t npu_calls = 0;
    std::string status;
};

bool ornith35_executor_validate(const MlxSafetensorsInfo &info,
                                 const Ornith35TextConfig &cfg,
                                 std::string &error);

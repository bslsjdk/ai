#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "mlx_safetensors.h"
#include "ornith35_arch.h"
#include "ornith35_attention.h"
#include "ornith35_deltanet.h"

struct Ornith35ExecutorStats {
    uint32_t layers_done = 0;
    uint32_t npu_calls = 0;
    std::string status;
};

bool ornith35_executor_validate(const MlxSafetensorsInfo &info,
                                 const Ornith35TextConfig &cfg,
                                 std::string &error);

struct Ornith35DecoderStep {
    uint32_t position = 0;
    std::vector<float> hidden;
    std::vector<float> logits;
    Ornith35ExecutorStats stats;
};

bool ornith35_executor_greedy_step(const std::string &model_path,
                                    const MlxSafetensorsInfo &info,
                                    const Ornith35TextConfig &cfg,
                                    uint32_t token_id,
                                    Ornith35DecoderStep &step,
                                    std::string &error);

struct Ornith35LayerRuntime;

bool ornith35_executor_forward_token(const std::string &model_path,
                                       const MlxSafetensorsInfo &info,
                                       const Ornith35TextConfig &cfg,
                                       uint32_t token_id,
                                       uint32_t position,
                                       Ornith35LayerRuntime &runtime,
                                       Ornith35DecoderStep &step,
                                       std::string &error);

struct Ornith35LayerRuntime {
    std::vector<Ornith35DeltaState> delta;
    std::vector<Ornith35AttentionState> attention;
    std::vector<float> work_a;
    std::vector<float> work_b;
    std::vector<float> work_c;
    uint32_t initialized_layers = 0;
};

bool ornith35_executor_init_runtime(const Ornith35TextConfig &cfg,
                                     uint32_t max_attention_tokens,
                                     Ornith35LayerRuntime &runtime,
                                     std::string &error);

bool ornith35_executor_apply_mlp(const std::string &model_path,
                                  const MlxSafetensorsInfo &info,
                                  const Ornith35TextConfig &cfg,
                                  uint32_t layer_index,
                                  const float *hidden,
                                  float *out,
                                  Ornith35ExecutorStats &stats,
                                  std::string &error);

bool ornith35_executor_run_attention_layer(const std::string &model_path,
                                           const MlxSafetensorsInfo &info,
                                           const Ornith35TextConfig &cfg,
                                           uint32_t layer_index,
                                           const float *hidden,
                                           float *out,
                                           Ornith35AttentionState &state,
                                           uint32_t position,
                                           Ornith35ExecutorStats &stats,
                                           std::string &error);

bool ornith35_executor_run_delta_layer(const std::string &model_path,
                                       const MlxSafetensorsInfo &info,
                                       const Ornith35TextConfig &cfg,
                                       uint32_t layer_index,
                                       const float *hidden,
                                       float *out,
                                       Ornith35DeltaState &state,
                                       Ornith35ExecutorStats &stats,
                                       std::string &error);

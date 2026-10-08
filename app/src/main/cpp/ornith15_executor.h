#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "mlx_safetensors.h"
#include "ornith15_arch.h"
#include "ornith15_attention.h"
#include "ornith15_deltanet.h"

struct Ornith15ExecutorStats {
    uint32_t layers_done = 0;
    uint32_t npu_calls = 0;
    std::string status;
};

bool ornith15_executor_validate(const MlxSafetensorsInfo &info,
                                 const Ornith15TextConfig &cfg,
                                 std::string &error);

struct Ornith15DecoderStep {
    uint32_t position = 0;
    std::vector<float> hidden;
    std::vector<float> logits;
    Ornith15ExecutorStats stats;
};

struct Ornith15LayerStaticWeights {
    std::vector<float> input_norm;
    std::vector<float> post_norm;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
    std::vector<float> a_log;
    std::vector<float> dt_bias;
    std::vector<float> delta_norm;
    std::vector<float> conv1d;
};

bool ornith15_executor_greedy_step(const std::string &model_path,
                                    const MlxSafetensorsInfo &info,
                                    const Ornith15TextConfig &cfg,
                                    uint32_t token_id,
                                    Ornith15DecoderStep &step,
                                    std::string &error);

struct Ornith15LayerRuntime;

bool ornith15_executor_forward_token(const std::string &model_path,
                                       const MlxSafetensorsInfo &info,
                                       const Ornith15TextConfig &cfg,
                                       uint32_t token_id,
                                       uint32_t position,
                                       Ornith15LayerRuntime &runtime,
                                       Ornith15DecoderStep &step,
                                       std::string &error);

struct Ornith15LayerRuntime {
    std::vector<Ornith15DeltaState> delta;
    std::vector<Ornith15AttentionState> attention;
    std::vector<Ornith15LayerStaticWeights> static_weights;
    std::vector<float> final_norm;
    std::vector<float> work_a;
    std::vector<float> work_b;
    std::vector<float> work_c;
    uint32_t initialized_layers = 0;
};

bool ornith15_executor_prepare_static_weights(const std::string &model_path,
                                              const MlxSafetensorsInfo &info,
                                              const Ornith15TextConfig &cfg,
                                              Ornith15LayerRuntime &runtime,
                                              std::string &error);

uint64_t ornith15_effective_attention_tokens(const Ornith15TextConfig &cfg,
                                                uint64_t requested_tokens,
                                                std::string &diagnostic);

bool ornith15_executor_init_runtime(const Ornith15TextConfig &cfg,
                                     uint32_t max_attention_tokens,
                                     Ornith15LayerRuntime &runtime,
                                     std::string &error);

// Clear recurrent/KV state before replaying a conversation prompt. A prompt replay
// must start from a clean model state; otherwise previous turns are counted twice.
void ornith15_executor_reset_runtime(Ornith15LayerRuntime &runtime);

bool ornith15_executor_apply_mlp(const std::string &model_path,
                                  const MlxSafetensorsInfo &info,
                                  const Ornith15TextConfig &cfg,
                                  uint32_t layer_index,
                                  const float *hidden,
                                  float *out,
                                  Ornith15ExecutorStats &stats,
                                  std::string &error);

bool ornith15_executor_run_attention_layer(const std::string &model_path,
                                           const MlxSafetensorsInfo &info,
                                           const Ornith15TextConfig &cfg,
                                           uint32_t layer_index,
                                           const float *hidden,
                                           float *out,
                                           Ornith15AttentionState &state,
                                           const Ornith15LayerStaticWeights &static_weights,
                                           uint32_t position,
                                           Ornith15ExecutorStats &stats,
                                           std::string &error);

bool ornith15_executor_run_delta_layer(const std::string &model_path,
                                       const MlxSafetensorsInfo &info,
                                       const Ornith15TextConfig &cfg,
                                       uint32_t layer_index,
                                       const float *hidden,
                                       float *out,
                                       Ornith15DeltaState &state,
                                       const Ornith15LayerStaticWeights &static_weights,
                                       Ornith15ExecutorStats &stats,
                                       std::string &error);

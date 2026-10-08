#include "ornith35_executor.h"
#include "ornith35_linear.h"
#include <string>

bool ornith35_executor_validate(const MlxSafetensorsInfo &info,
                                 const Ornith35TextConfig &cfg,
                                 std::string &error) {
    if (!ornith35_validate_config(cfg, error)) return false;
    if (!info.tensor_count || info.tensors.empty()) {
        error = "empty_safetensors";
        return false;
    }
    const auto plan = ornith35_make_layer_plan();
    if (plan.size() != cfg.num_layers) {
        error = "layer_plan_size_mismatch";
        return false;
    }
    uint32_t affine = 0;
    for (const auto &t : info.tensors) {
        if (t.dtype == "U32" && t.shape.size() == 2 &&
            t.name.size() >= 7 &&
            t.name.compare(t.name.size() - 7, 7, ".weight") == 0) {
            ++affine;
        }
    }
    if (!affine) {
        error = "no_affine4_weights";
        return false;
    }
    return true;
}

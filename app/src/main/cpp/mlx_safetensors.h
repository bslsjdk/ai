#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct MlxTensorInfo {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
};

struct MlxSafetensorsInfo {
    uint64_t file_bytes = 0;
    uint64_t header_bytes = 0;
    uint64_t tensor_count = 0;
    uint64_t quantized_tensor_count = 0;
    uint64_t total_tensor_bytes = 0;
    std::string quantization_summary;
    std::vector<MlxTensorInfo> tensors;
};

bool mlx_safetensors_probe(const std::string & path, MlxSafetensorsInfo & out, std::string & error);

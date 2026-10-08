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

struct MlxQuantInfo {
    bool valid = false;
    uint32_t bits = 0;
    uint32_t group_size = 0;
    uint64_t quantized_weight_count = 0;
    uint64_t scale_count = 0;
    uint64_t bias_count = 0;
};
bool mlx_infer_affine4(const MlxSafetensorsInfo & info, MlxQuantInfo & out, std::string & error);
bool mlx_read_tensor_range(const std::string & path, const MlxTensorInfo & tensor,
                           uint64_t relative_offset, void * dst, size_t bytes, std::string & error);

bool mlx_decode_affine4_tile(const uint32_t * packed, size_t packed_words,
                             const float * scales, const float * biases,
                             size_t rows, size_t cols, size_t group_size,
                             float * out, size_t out_capacity);

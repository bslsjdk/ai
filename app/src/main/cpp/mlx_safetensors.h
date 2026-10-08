#pragma once
#include <cstdint>
#include <cstddef>
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

// Reads a tiny real MLX affine4 weight tile from disk and executes one 32x64x32
// multiply through the in-process MCNPU bridge. This is a diagnostic proof that
// real model bytes, not synthetic weights, reached QNN HTP.
struct MlxAffine4Tile {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint32_t> packed_weight;
    std::vector<float> scales;
    std::vector<float> biases;
    std::string tensor_name;
};

bool mlx_read_affine4_tile(const std::string & path,
                           const MlxSafetensorsInfo & info,
                           const std::string & tensor_suffix,
                           uint32_t row0, uint32_t rows,
                           uint32_t col0, uint32_t cols,
                           MlxAffine4Tile & out,
                           std::string & error);

std::string mlx_affine4_npu_probe(const std::string & path,
                                  const MlxSafetensorsInfo & info);

bool mlx_decode_affine4_tile(const uint32_t * packed, size_t packed_words,
                             const float * scales, const float * biases,
                             size_t rows, size_t cols, size_t group_size,
                             float * out, size_t out_capacity);

struct MlxNpuTileResult {
    bool ok = false;
    uint32_t m = 0, k = 0, n = 0;
    float scale = 0.0f;
    float max_abs_error = 0.0f;
    float max_relative_error = 0.0f;
    std::string status;
};

MlxNpuTileResult mlx_affine4_npu_matmul_tile(
        const float * activation,
        const uint32_t * packed_weight,
        const float * scales,
        const float * biases,
        uint32_t m, uint32_t k, uint32_t n,
        uint32_t group_size);

bool mlx_read_affine4_row(const std::string &path, const MlxSafetensorsInfo &info,
                          const std::string &weight_name, uint32_t row,
                          std::vector<float> &out, std::string &error);

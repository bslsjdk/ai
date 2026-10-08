#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

struct Ornith15Tokenizer {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    int32_t bos = 0;
    int32_t eos = 0;
    bool loaded = false;
    std::unordered_map<std::string, int32_t> bytes_to_id;
    std::vector<std::string> special_text;
    std::vector<int32_t> special_ids;
};

// Loads the compact sidecar emitted by the app's model-preparation step.
// It deliberately does not pretend that raw MLX weights contain a tokenizer.
bool ornith15_tokenizer_load(const std::string &path, Ornith15Tokenizer &out, std::string &error);
bool ornith15_tokenizer_encode(const Ornith15Tokenizer &tok, const std::string &text,
                               std::vector<int32_t> &ids, std::string &error);

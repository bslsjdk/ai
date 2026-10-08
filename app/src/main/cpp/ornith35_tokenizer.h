#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct Ornith35Tokenizer {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    int32_t bos = 0;
    int32_t eos = 0;
    bool loaded = false;
};

bool ornith35_tokenizer_load(const std::string &path, Ornith35Tokenizer &out, std::string &error);
bool ornith35_tokenizer_encode(const Ornith35Tokenizer &tok, const std::string &text,
                               std::vector<int32_t> &ids, std::string &error);

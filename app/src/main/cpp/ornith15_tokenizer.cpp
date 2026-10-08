#include "ornith15_tokenizer.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {
static bool u32(std::ifstream &f, uint32_t &v) {
    unsigned char b[4];
    if (!f.read(reinterpret_cast<char *>(b), 4)) return false;
    v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
        ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return true;
}
static bool i32(std::ifstream &f, int32_t &v) {
    uint32_t u = 0;
    if (!u32(f, u)) return false;
    std::memcpy(&v, &u, sizeof(v));
    return true;
}
static bool str32(std::ifstream &f, std::string &s) {
    uint32_t n = 0;
    if (!u32(f, n) || n > 1u << 20) return false;
    s.resize(n);
    return n == 0 || (bool)f.read(s.data(), n);
}
static void rebuild_indexes(Ornith15Tokenizer &out) {
    out.bytes_to_id.clear();
    out.bytes_to_id.reserve(out.tokens.size() * 2u);
    for (size_t i = 0; i < out.tokens.size(); ++i) {
        if (!out.tokens[i].empty())
            out.bytes_to_id.emplace(out.tokens[i], (int32_t)i);
    }
}
static bool encode_segment(const Ornith15Tokenizer &tok, const std::string &text,
                           std::vector<int32_t> &ids, std::string &error) {
    if (text.empty()) return true;
    std::vector<int32_t> cur;
    std::vector<std::string> pieces;
    cur.reserve(text.size());
    pieces.reserve(text.size());
    for (unsigned char c : text) {
        std::string one(1, (char)c);
        auto it = tok.bytes_to_id.find(one);
        if (it == tok.bytes_to_id.end()) {
            error = "byte_token_not_found";
            return false;
        }
        cur.push_back(it->second);
        pieces.push_back(std::move(one));
    }

    // BPE merge loop. The model prompt is small compared with the 9B forward
    // pass, so this deliberately simple implementation keeps RAM bounded.
    while (cur.size() > 1) {
        size_t best = std::numeric_limits<size_t>::max();
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        for (size_t i = 0; i + 1 < cur.size(); ++i) {
            const std::string merged = pieces[i] + pieces[i + 1];
            auto it = tok.bytes_to_id.find(merged);
            if (it == tok.bytes_to_id.end()) continue;
            const int32_t rank = it->second >= 0 &&
                                 (size_t)it->second < tok.scores.size()
                                 ? (int32_t)tok.scores[(size_t)it->second]
                                 : -1;
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best = i;
            }
        }
        if (best == std::numeric_limits<size_t>::max()) break;
        const std::string merged = pieces[best] + pieces[best + 1];
        const int32_t merged_id =
            tok.bytes_to_id.find(merged)->second;
        cur[best] = merged_id;
        pieces[best] = merged;
        cur.erase(cur.begin() + (ptrdiff_t)best + 1);
        pieces.erase(pieces.begin() + (ptrdiff_t)best + 1);
    }
    ids.insert(ids.end(), cur.begin(), cur.end());
    return true;
}
}

bool ornith15_tokenizer_load(const std::string &path,
                             Ornith15Tokenizer &out,
                             std::string &error) {
    out = Ornith15Tokenizer{};
    std::ifstream f(path, std::ios::binary);
    if (!f) { error = "tokenizer_open_failed"; return false; }

    char m[4];
    if (!f.read(m, 4)) { error = "tokenizer_magic_read_failed"; return false; }

    if (std::string(m, 4) == "OTK2") {
        uint32_t ver = 0, n = 0, special_count = 0;
        int32_t bos = -1, eos = -1;
        if (!u32(f, ver) || ver != 2 || !u32(f, n) ||
            n > 300000 || !i32(f, bos) || !i32(f, eos) ||
            !u32(f, special_count) || special_count > 1024) {
            error = "bad_tokenizer_header";
            return false;
        }
        out.tokens.resize(n);
        out.scores.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            int32_t rank = -1;
            if (!str32(f, out.tokens[i]) || !i32(f, rank)) {
                // Scores are only integer merge ranks in OTK2. Read the bits
                // explicitly below on the next pass is not possible, so reject
                // malformed records cleanly.
                error = "bad_tokenizer_entry";
                return false;
            }
            out.scores[i] = (float)rank;
        }
        // OTK2 stores integer merge ranks directly.
        out.bos = bos;
        out.eos = eos;
        out.special_text.resize(special_count);
        out.special_ids.resize(special_count);
        for (uint32_t i = 0; i < special_count; ++i) {
            uint32_t sid = 0;
            if (!str32(f, out.special_text[i]) || !u32(f, sid) || sid >= n) {
                error = "bad_tokenizer_special";
                return false;
            }
            out.special_ids[i] = (int32_t)sid;
        }
        rebuild_indexes(out);
        out.loaded = true;
        return true;
    }

    if (std::string(m, 4) != "OTOK") {
        error = "bad_tokenizer_magic";
        return false;
    }
    uint32_t ver = 0, n = 0;
    if (!u32(f, ver) || ver != 1 || !u32(f, n) || n > 300000) {
        error = "bad_tokenizer_header";
        return false;
    }
    out.tokens.resize(n);
    out.scores.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (!str32(f, out.tokens[i])) { error = "bad_tokenizer_token"; return false; }
        uint32_t bits = 0;
        if (!u32(f, bits)) { error = "bad_tokenizer_score"; return false; }
        std::memcpy(&out.scores[i], &bits, sizeof(bits));
    }
    rebuild_indexes(out);
    out.loaded = true;
    return true;
}

bool ornith15_tokenizer_encode(const Ornith15Tokenizer &tok,
                               const std::string &text,
                               std::vector<int32_t> &ids,
                               std::string &error) {
    if (!tok.loaded) { error = "tokenizer_not_loaded"; return false; }
    ids.clear();

    size_t segment_start = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t matched = 0;
        int32_t special_id = -1;
        for (size_t i = 0; i < tok.special_text.size(); ++i) {
            const std::string &sp = tok.special_text[i];
            if (sp.empty() || pos + sp.size() > text.size()) continue;
            if (text.compare(pos, sp.size(), sp) == 0 && sp.size() > matched) {
                matched = sp.size();
                special_id = tok.special_ids[i];
            }
        }
        if (matched == 0) {
            ++pos;
            continue;
        }

        if (pos > segment_start &&
            !encode_segment(tok, text.substr(segment_start, pos - segment_start), ids, error))
            return false;
        ids.push_back(special_id);
        pos += matched;
        segment_start = pos;
    }

    if (segment_start < text.size() &&
        !encode_segment(tok, text.substr(segment_start), ids, error))
        return false;
    return true;
}

std::string ornith15_tokenizer_decode(const Ornith15Tokenizer &tok,
                                      const std::vector<int32_t> &ids) {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || (size_t)id >= tok.tokens.size()) continue;
        out += tok.tokens[(size_t)id];
    }
    return out;
}

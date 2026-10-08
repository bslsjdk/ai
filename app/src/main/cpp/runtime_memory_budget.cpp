#include "runtime_memory_budget.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

namespace {
struct ProcMemory {
    uint64_t rss = 0;
    uint64_t hwm = 0;
    uint64_t peak = 0;
};

static ProcMemory read_proc_memory() {
    ProcMemory out;
    std::ifstream in("/proc/self/status");
    if (!in) return out;
    std::string line;
    while (std::getline(in, line)) {
        const char *field = nullptr;
        uint64_t *dst = nullptr;
        if (line.rfind("VmRSS:", 0) == 0) { field = "VmRSS:"; dst = &out.rss; }
        else if (line.rfind("VmHWM:", 0) == 0) { field = "VmHWM:"; dst = &out.hwm; }
        else if (line.rfind("VmPeak:", 0) == 0) { field = "VmPeak:"; dst = &out.peak; }
        if (!field || !dst) continue;
        unsigned long long kb = 0;
        if (std::sscanf(line.c_str(), "%*[^:]: %llu kB", &kb) == 1)
            *dst = (uint64_t)kb * 1024ull;
    }
    return out;
}

static uint64_t read_rss_bytes() {
    return read_proc_memory().rss;
}

static uint64_t ceil_div(uint64_t a, uint64_t b) {
    return b ? (a + b - 1u) / b : std::numeric_limits<uint64_t>::max();
}
}

uint64_t ornith15_process_rss_bytes() {
    return read_rss_bytes();
}

uint64_t ornith15_memory_effective_attention_tokens(
    uint64_t requested_tokens,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t full_attention_layers,
    uint64_t fixed_state_bytes,
    std::string &diagnostic) {

    constexpr uint64_t MAX_CONTEXT = 65536ull;
    constexpr uint64_t TOKEN_GRANULARITY = 4096ull;
    const uint64_t requested = std::min<uint64_t>(requested_tokens, MAX_CONTEXT);

    if (!requested || !kv_heads || !head_dim || !full_attention_layers) {
        diagnostic = "invalid_memory_plan_inputs";
        return 0;
    }

    const uint64_t rss = read_rss_bytes();
    const uint64_t state_cap = ORNITH15_RSS_HARD_LIMIT_BYTES > ORNITH15_RSS_SAFETY_BYTES + rss
        ? ORNITH15_RSS_HARD_LIMIT_BYTES - ORNITH15_RSS_SAFETY_BYTES - rss
        : 0;

    if (state_cap <= ORNITH15_NON_STATE_RESERVE_BYTES + fixed_state_bytes) {
        diagnostic = "no_attention_budget rss_bytes=" + std::to_string(rss);
        return 0;
    }

    const uint64_t kv_budget = state_cap - ORNITH15_NON_STATE_RESERVE_BYTES - fixed_state_bytes;
    const uint64_t bytes_per_token =
        (uint64_t)kv_heads * head_dim * 2ull * 2ull * full_attention_layers;
    if (!bytes_per_token) {
        diagnostic = "attention_bytes_per_token_zero";
        return 0;
    }

    uint64_t max_tokens = std::min<uint64_t>(requested, kv_budget / bytes_per_token);
    max_tokens = (max_tokens / TOKEN_GRANULARITY) * TOKEN_GRANULARITY;

    if (max_tokens < TOKEN_GRANULARITY) {
        diagnostic = "attention_budget_too_small rss_bytes=" + std::to_string(rss) +
                     " fixed_state_bytes=" + std::to_string(fixed_state_bytes) +
                     " bytes_per_token=" + std::to_string(bytes_per_token);
        return 0;
    }

    diagnostic = "rss_bytes=" + std::to_string(rss) +
                 " requested=" + std::to_string(requested) +
                 " effective=" + std::to_string(max_tokens) +
                 " kv_budget_bytes=" + std::to_string(kv_budget) +
                 " non_state_reserve_bytes=" + std::to_string(ORNITH15_NON_STATE_RESERVE_BYTES) +
                 " safety_bytes=" + std::to_string(ORNITH15_RSS_SAFETY_BYTES);
    return max_tokens;
}

bool ornith15_memory_headroom(uint64_t planned_growth_bytes,
                              const char *tag,
                              std::string &error) {
    const uint64_t rss = read_rss_bytes();
    if (!rss) return true; // /proc can be unavailable on unusual test hosts.
    if (planned_growth_bytes > ORNITH15_RSS_HARD_LIMIT_BYTES) {
        error = "memory_plan_too_large tag=" + std::string(tag ? tag : "unknown");
        return false;
    }
    if (rss > ORNITH15_RSS_HARD_LIMIT_BYTES - planned_growth_bytes) {
        error = "memory_headroom_exceeded tag=" + std::string(tag ? tag : "unknown") +
                " rss_bytes=" + std::to_string(rss) +
                " planned_growth_bytes=" + std::to_string(planned_growth_bytes) +
                " hard_limit_bytes=" + std::to_string(ORNITH15_RSS_HARD_LIMIT_BYTES);
        return false;
    }
    return true;
}

bool ornith15_memory_within_limit(std::string &error) {
    const uint64_t rss = read_rss_bytes();
    if (!rss) return true;
    if (rss >= ORNITH15_RSS_HARD_LIMIT_BYTES) {
        error = "rss_hard_limit_reached rss_bytes=" + std::to_string(rss) +
                " hard_limit_bytes=" + std::to_string(ORNITH15_RSS_HARD_LIMIT_BYTES);
        return false;
    }
    return true;
}

std::string ornith15_memory_status() {
    const ProcMemory mem = read_proc_memory();
    return "rss_bytes=" + std::to_string(mem.rss) +
           " hwm_bytes=" + std::to_string(mem.hwm) +
           " peak_virtual_bytes=" + std::to_string(mem.peak) +
           " rss_limit_bytes=" + std::to_string(ORNITH15_RSS_HARD_LIMIT_BYTES) +
           " safety_bytes=" + std::to_string(ORNITH15_RSS_SAFETY_BYTES) +
           " non_state_reserve_bytes=" + std::to_string(ORNITH15_NON_STATE_RESERVE_BYTES) +
           " model_state_budget_bytes=" + std::to_string(ORNITH15_MODEL_STATE_BUDGET_BYTES);
}

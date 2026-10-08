#pragma once
#include <cstdint>
#include <string>

// This guard is deliberately below the user's 4 GiB RSS ceiling.  It is a
// process-level safety boundary, not a claim that the Android OS exposes a
// hard per-process RSS limit.
constexpr uint64_t ORNITH15_RSS_HARD_LIMIT_BYTES = 3584ull << 20; // 3.5 GiB
constexpr uint64_t ORNITH15_RSS_SAFETY_BYTES = 256ull << 20;      // 256 MiB
constexpr uint64_t ORNITH15_NON_STATE_RESERVE_BYTES = 640ull << 20;
constexpr uint64_t ORNITH15_MODEL_STATE_BUDGET_BYTES = 2304ull << 20;

uint64_t ornith15_process_rss_bytes();
uint64_t ornith15_effective_attention_tokens(
    uint64_t requested_tokens,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t full_attention_layers,
    uint64_t fixed_state_bytes,
    std::string &diagnostic);

bool ornith15_memory_headroom(uint64_t planned_growth_bytes,
                              const char *tag,
                              std::string &error);
bool ornith15_memory_within_limit(std::string &error);
std::string ornith15_memory_status();

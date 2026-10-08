#pragma once
#include <cstdint>
#include <string>

// In-process MCNPU/QNN backend bridge.
// This is deliberately not an IPC API: Ornith and MCNPU live in the same APK
// and the call stays inside the same native process.
bool mcnpu_backend_ready();
std::string mcnpu_backend_status();
std::string mcnpu_backend_matmul_int8(
        const int8_t* a, const int8_t* b, int8_t* c,
        uint32_t m, uint32_t k, uint32_t n, float& scaleC);

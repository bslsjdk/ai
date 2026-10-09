#include "ornith15_deltanet.h"
#include <algorithm>
#include <cmath>

namespace {
static inline float l2_normalize(float *x, uint32_t n) {
    float ss = 1.0e-6f;
    for (uint32_t i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss);
    for (uint32_t i = 0; i < n; ++i) x[i] *= inv;
    return inv;
}
}

bool ornith15_deltanet_init(Ornith15DeltaState &s, uint32_t conv_channels, uint32_t conv_kernel) {
    if (!s.key_heads || !s.value_heads || !s.key_dim || !s.value_dim ||
        (s.value_heads % s.key_heads) != 0) return false;
    s.state.assign((size_t)s.value_heads * s.key_dim * s.value_dim, 0.0f);
    if (conv_channels && conv_kernel) s.conv.assign((size_t)conv_channels * conv_kernel, 0.0f); else s.conv.clear();
    s.tokens = 0;
    return true;
}

bool ornith15_deltanet_step(
    Ornith15DeltaState &s,
    const float *q, const float *k, const float *v,
    const float *beta, const float *decay_log,
    uint32_t hq, uint32_t hv,
    uint32_t kd, uint32_t vd,
    std::vector<float> &out,
    std::string &error) {

    if (!q || !k || !v || !beta || !decay_log ||
        hq != s.key_heads || hv != s.value_heads ||
        kd != s.key_dim || vd != s.value_dim ||
        !hq || !hv || !kd || !vd || (hv % hq) != 0) {
        error = "deltanet_shape_mismatch";
        return false;
    }
    if (s.state.size() != (size_t)hv * kd * vd && !ornith15_deltanet_init(s)) {
        error = "deltanet_state_init_failed";
        return false;
    }

    out.assign((size_t)hv * vd, 0.0f);
    const uint32_t repeat = hv / hq;

    // Qwen3.5's recurrent path uses L2-normalized Q/K, then a gated decay
    // and delta-rule write. The official recurrent equations are:
    //   S <- exp(g) S
    //   delta <- (v - k^T S) * beta
    //   S <- S + k outer delta
    //   y <- q^T S
    std::vector<float> qn(kd), kn(kd);

    for (uint32_t h = 0; h < hv; ++h) {
        const uint32_t hk = h / repeat;
        std::copy(q + (size_t)hk * kd, q + (size_t)(hk + 1) * kd, qn.begin());
        std::copy(k + (size_t)hk * kd, k + (size_t)(hk + 1) * kd, kn.begin());
        l2_normalize(qn.data(), kd);
        l2_normalize(kn.data(), kd);
        // Qwen3.5 scales normalized Q by 1/sqrt(head_k_dim) before
        // the recurrent read. K is left unit-normalized.
        const float q_scale = 1.0f / std::sqrt((float)kd);
        for (uint32_t d = 0; d < kd; ++d) qn[d] *= q_scale;

        float *S = s.state.data() + (size_t)h * kd * vd;
        const float decay = std::exp(decay_log[h]);
        for (size_t i = 0, total = (size_t)kd * vd; i < total; ++i)
            S[i] *= decay;

        std::vector<float> predicted(vd, 0.0f);
        for (uint32_t d = 0; d < kd; ++d)
            for (uint32_t e = 0; e < vd; ++e)
                predicted[e] += kn[d] * S[(size_t)d * vd + e];

        const float b = std::max(0.0f, std::min(1.0f, beta[h]));
        for (uint32_t e = 0; e < vd; ++e) {
            const float delta = (v[(size_t)h * vd + e] - predicted[e]) * b;
            for (uint32_t d = 0; d < kd; ++d)
                S[(size_t)d * vd + e] += kn[d] * delta;
        }

        for (uint32_t d = 0; d < kd; ++d)
            for (uint32_t e = 0; e < vd; ++e)
                out[(size_t)h * vd + e] += qn[d] * S[(size_t)d * vd + e];
    }

    ++s.tokens;
    return true;
}


bool ornith15_deltanet_conv_step(Ornith15DeltaState &s,
                                  const float *input,
                                  uint32_t channels,
                                  uint32_t kernel,
                                  std::vector<float> &output,
                                  std::string &error) {
    if (!input || !channels || !kernel) { error = "conv_args"; return false; }
    if (s.conv.size() != (size_t)channels * kernel)
        s.conv.assign((size_t)channels * kernel, 0.0f);
    output.resize(channels);
    for (uint32_t ch = 0; ch < channels; ++ch) {
        float acc = 0.0f;
        // The MLX checkpoint stores depthwise conv weights separately; this
        // state helper intentionally provides the causal history boundary.
        // Weight application is performed by the executor once the conv
        // tensor is streamed, so no full kernel tensor is resident here.
        (void)acc;
        output[ch] = input[ch];
    }
    if (kernel > 1) {
        for (uint32_t ch = 0; ch < channels; ++ch) {
            float *hist = s.conv.data() + (size_t)ch * kernel;
            // Store causal history oldest-to-newest, matching Conv1d weight order.
            for (uint32_t j = 0; j + 1 < kernel; ++j) hist[j] = hist[j + 1];
            hist[kernel - 1] = input[ch];
        }
    }
    return true;
}

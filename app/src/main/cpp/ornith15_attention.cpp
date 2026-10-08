#include "ornith15_attention.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
static uint16_t float_to_half(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t mant = bits & 0x007fffffu;
    const int32_t exp = (int32_t)((bits >> 23) & 0xffu);

    if (exp == 0xff) {
        if (mant) return (uint16_t)(sign | 0x7e00u);
        return (uint16_t)(sign | 0x7c00u);
    }

    int32_t half_exp = exp - 127 + 15;
    if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return (uint16_t)sign;
        uint32_t m = mant | 0x00800000u;
        const int32_t shift = 14 - half_exp;
        uint32_t hm = m >> shift;
        const uint32_t round_bit = (m >> (shift - 1)) & 1u;
        const uint32_t sticky = m & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (hm & 1u))) ++hm;
        return (uint16_t)(sign | hm);
    }

    uint32_t hm = mant >> 13;
    const uint32_t round_bit = (mant >> 12) & 1u;
    const uint32_t sticky = mant & 0xfffu;
    if (round_bit && (sticky || (hm & 1u))) {
        ++hm;
        if (hm == 0x400u) {
            hm = 0;
            ++half_exp;
            if (half_exp >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | hm);
}

static float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t m = mant;
            int32_t e = -14;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03ffu;
            bits = sign | (uint32_t)(e + 127) << 23 | (m << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static void rope(float *x, uint32_t dim, uint32_t rotary_dim, uint32_t pos, float theta) {
    const uint32_t rd = std::min(dim, rotary_dim);
    if(rd < 2u) return;
    const uint32_t half = rd / 2u;
    // Qwen3.5 applies rotate_half to the rotary slice, pairing the first
    // half with the second half rather than adjacent dimensions.
    for(uint32_t i=0;i<half;i++) {
        const float inv = std::pow(theta, -(float)i/(float)half);
        const float a = (float)pos*inv;
        const float c = std::cos(a), sn = std::sin(a);
        const float x0=x[i], x1=x[i+half];
        x[i]=x0*c-x1*sn;
        x[i+half]=x0*sn+x1*c;
    }
}

static void rms(float *x, uint32_t n, const float *weight=nullptr, float eps=1e-6f) {
    float ss=0.0f;
    for(uint32_t i=0;i<n;i++) ss += x[i]*x[i];
    const float inv=1.0f/std::sqrt(ss/(float)n+eps);
    for(uint32_t i=0;i<n;i++) {
        // Qwen3.5 q/k RMSNorm uses the same zero-centered gain:
        // normalized(x) * (1 + weight).
        const float gain = weight ? (1.0f + weight[i]) : 1.0f;
        x[i]=x[i]*inv*gain;
    }
}
}

bool ornith15_attention_init(Ornith15AttentionState &s, uint32_t max_tokens) {
    constexpr uint32_t MAX_RESIDENT_TOKENS = 65536u;
    if (!max_tokens || max_tokens > MAX_RESIDENT_TOKENS ||
        s.q_heads != 16 || s.kv_heads != 4 || s.head_dim != 256)
        return false;

    const uint64_t elems64 = (uint64_t)max_tokens * s.kv_heads * s.head_dim;
    if (elems64 > (uint64_t)std::numeric_limits<size_t>::max())
        return false;
    const size_t elems = (size_t)elems64;
    s.keys.assign(elems, 0);
    s.values.assign(elems, 0);
    s.tokens=0;
    s.window_tokens=max_tokens;
    return true;
}

bool ornith15_attention_step(
    Ornith15AttentionState &s,
    const float *q, const float *k, const float *v,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t position, float rope_theta,
    std::vector<float> &out, std::string &error,
    const float *q_norm_weight, const float *k_norm_weight) {

    const uint32_t capacity = s.window_tokens;
    const uint64_t expected = (uint64_t)capacity * kv_heads * head_dim;
    if(!q||!k||!v||q_heads!=s.q_heads||kv_heads!=s.kv_heads||
       head_dim!=s.head_dim||(q_heads%kv_heads)!=0||!capacity ||
       expected > (uint64_t)std::numeric_limits<size_t>::max() ||
       s.keys.size() != (size_t)expected || s.values.size() != (size_t)expected) {
        error="full_attention_shape_mismatch";
        return false;
    }

    // Keep the newest resident window. KV is stored as FP16 to make 64K
    // feasible under the phone's strict <4 GiB process-memory budget.
    const uint32_t slot = position % capacity;
    const uint32_t count = std::min(position + 1u, capacity);
    const uint32_t first_position = position + 1u - count;

    std::vector<float> qr(head_dim), kr(head_dim);
    const uint32_t group=q_heads/kv_heads;
    for(uint32_t h=0;h<kv_heads;h++) {
        std::copy(k+(size_t)h*head_dim,k+(size_t)(h+1)*head_dim,kr.begin());
        rms(kr.data(),head_dim,k_norm_weight);
        rope(kr.data(),head_dim,64u,position,rope_theta);
        uint16_t *kdst=s.keys.data()+(size_t)slot*kv_heads*head_dim+h*head_dim;
        uint16_t *vdst=s.values.data()+(size_t)slot*kv_heads*head_dim+h*head_dim;
        for(uint32_t d=0;d<head_dim;d++) {
            kdst[d]=float_to_half(kr[d]);
            vdst[d]=float_to_half(v[(size_t)h*head_dim+d]);
        }
    }

    out.assign((size_t)q_heads*head_dim,0.0f);
    std::vector<float> scores(count);
    const float inv_sqrt_head = 1.0f/std::sqrt((float)head_dim);
    for(uint32_t h=0;h<q_heads;h++) {
        std::copy(q+(size_t)h*head_dim,q+(size_t)(h+1)*head_dim,qr.begin());
        rms(qr.data(),head_dim,q_norm_weight);
        rope(qr.data(),head_dim,64u,position,rope_theta);
        const uint32_t kh=h/group;
        float mx=-INFINITY;
        for(uint32_t t=0;t<count;t++) {
            const uint32_t logical_position = first_position + t;
            const uint32_t cache_slot = logical_position % capacity;
            const uint16_t *kt=s.keys.data()+(size_t)cache_slot*kv_heads*head_dim+kh*head_dim;
            float dot=0.0f;
            for(uint32_t d=0;d<head_dim;d++) dot+=qr[d]*half_to_float(kt[d]);
            scores[t]=dot*inv_sqrt_head;
            mx=std::max(mx,scores[t]);
        }
        float denom=0.0f;
        for(uint32_t t=0;t<count;t++) {
            scores[t]=std::exp(scores[t]-mx);
            denom+=scores[t];
        }
        const float inv=1.0f/std::max(denom,1e-20f);
        for(uint32_t t=0;t<count;t++) {
            const uint32_t logical_position = first_position + t;
            const uint32_t cache_slot = logical_position % capacity;
            const uint16_t *vt=s.values.data()+(size_t)cache_slot*kv_heads*head_dim+kh*head_dim;
            const float scale=scores[t]*inv;
            for(uint32_t d=0;d<head_dim;d++)
                out[(size_t)h*head_dim+d]+=scale*half_to_float(vt[d]);
        }
    }
    s.tokens=std::max<uint64_t>(s.tokens,(uint64_t)position+1);
    return true;
}

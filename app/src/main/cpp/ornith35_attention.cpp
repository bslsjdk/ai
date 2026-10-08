#include "ornith35_attention.h"
#include <algorithm>
#include <cmath>

namespace {
static void rope(float *x, uint32_t dim, uint32_t pos, float theta) {
    for (uint32_t i=0; i+1<dim; i+=2) {
        const float inv = std::pow(theta, -(float)i / (float)dim);
        const float a = (float)pos * inv;
        const float c = std::cos(a), s = std::sin(a);
        const float x0=x[i], x1=x[i+1];
        x[i]=x0*c-x1*s;
        x[i+1]=x0*s+x1*c;
    }
}
static void rms(float *x, uint32_t n, const float *weight=nullptr, float eps=1e-6f) {
    float ss=0.0f;
    for(uint32_t i=0;i<n;i++) ss += x[i]*x[i];
    const float inv=1.0f/std::sqrt(ss/(float)n+eps);
    for(uint32_t i=0;i<n;i++) x[i]=x[i]*inv*(weight?weight[i]:1.0f);
}
}

bool ornith35_attention_init(Ornith35AttentionState &s, uint32_t max_tokens) {
    if (!max_tokens || s.q_heads != 16 || s.kv_heads != 4 || s.head_dim != 256)
        return false;
    s.keys.assign((size_t)max_tokens*s.kv_heads*s.head_dim,0.0f);
    s.values.assign((size_t)max_tokens*s.kv_heads*s.head_dim,0.0f);
    s.tokens=0;
    return true;
}

bool ornith35_attention_step(
    Ornith35AttentionState &s,
    const float *q, const float *k, const float *v,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t position, float rope_theta,
    std::vector<float> &out, std::string &error,
    const float *q_norm_weight, const float *k_norm_weight) {

    if(!q||!k||!v||q_heads!=s.q_heads||kv_heads!=s.kv_heads||
       head_dim!=s.head_dim||(q_heads%kv_heads)!=0||
       position >= s.keys.size()/(size_t)(kv_heads*head_dim)) {
        error="full_attention_shape_mismatch";
        return false;
    }

    std::vector<float> qr(head_dim), kr(head_dim);
    const uint32_t group=q_heads/kv_heads;
    for(uint32_t h=0;h<kv_heads;h++) {
        std::copy(k+(size_t)h*head_dim,k+(size_t)(h+1)*head_dim,kr.begin());
        rms(kr.data(),head_dim,k_norm_weight);
        rope(kr.data(),head_dim,position,rope_theta);
        std::copy(kr.begin(),kr.end(),
                  s.keys.begin()+(size_t)position*kv_heads*head_dim+h*head_dim);
        std::copy(v+(size_t)h*head_dim,v+(size_t)(h+1)*head_dim,
                  s.values.begin()+(size_t)position*kv_heads*head_dim+h*head_dim);
    }

    out.assign((size_t)q_heads*head_dim,0.0f);
    const uint32_t count=position+1;
    std::vector<float> scores(count);
    for(uint32_t h=0;h<q_heads;h++) {
        std::copy(q+(size_t)h*head_dim,q+(size_t)(h+1)*head_dim,qr.begin());
        rms(qr.data(),head_dim,q_norm_weight);
        rope(qr.data(),head_dim,position,rope_theta);
        const uint32_t kh=h/group;
        float mx=-INFINITY;
        for(uint32_t t=0;t<count;t++) {
            const float *kt=s.keys.data()+(size_t)t*kv_heads*head_dim+kh*head_dim;
            float dot=0.0f;
            for(uint32_t d=0;d<head_dim;d++) dot+=qr[d]*kt[d];
            scores[t]=dot/std::sqrt((float)head_dim);
            mx=std::max(mx,scores[t]);
        }
        float denom=0.0f;
        for(uint32_t t=0;t<count;t++) {
            scores[t]=std::exp(scores[t]-mx);
            denom+=scores[t];
        }
        const float inv=1.0f/std::max(denom,1e-20f);
        for(uint32_t t=0;t<count;t++) {
            const float *vt=s.values.data()+(size_t)t*kv_heads*head_dim+kh*head_dim;
            for(uint32_t d=0;d<head_dim;d++)
                out[(size_t)h*head_dim+d]+=scores[t]*inv*vt[d];
        }
    }
    s.tokens=std::max<uint64_t>(s.tokens,(uint64_t)position+1);
    return true;
}

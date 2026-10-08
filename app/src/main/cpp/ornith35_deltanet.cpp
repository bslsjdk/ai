#include "ornith35_deltanet.h"
#include <algorithm>
#include <cmath>

bool ornith35_deltanet_init(Ornith35DeltaState &s) {
    s.state.assign((size_t)s.value_heads*s.key_dim*s.value_dim,0.0f);
    s.conv.clear();
    s.tokens=0;
    return true;
}

bool ornith35_deltanet_step(Ornith35DeltaState &s,
                            const float *q,const float *k,const float *v,
                            const float *gate,uint32_t hq,uint32_t hv,
                            uint32_t kd,uint32_t vd,std::vector<float> &out,
                            std::string &error) {
    if(!q||!k||!v||hq!=s.key_heads||hv!=s.value_heads||
       kd!=s.key_dim||vd!=s.value_dim) {
        error="deltanet_shape_mismatch"; return false;
    }
    if(!s.state.size()) ornith35_deltanet_init(s);
    out.assign((size_t)hv*vd,0.0f);
    const uint32_t ratio=hv/hq;
    for(uint32_t h=0;h<hv;h++) {
        const uint32_t kh=h/ratio;
        const float g=gate ? std::max(0.0f,std::min(1.0f,gate[h])) : 1.0f;
        float *S=s.state.data()+(size_t)h*kd*vd;
        const float *qq=q+(size_t)kh*kd;
        const float *kk=k+(size_t)kh*kd;
        const float *vv=v+(size_t)h*vd;
        float denom=1e-4f;
        for(uint32_t d=0;d<kd;d++) denom+=std::fabs(kk[d]);
        for(uint32_t d=0;d<kd;d++) {
            const float qv=qq[d];
            const float kv=kk[d];
            for(uint32_t e=0;e<vd;e++) {
                const size_t idx=(size_t)d*vd+e;
                S[idx]=g*S[idx]+kv*vv[e];
                out[(size_t)h*vd+e]+=qv*S[idx]/denom;
            }
        }
    }
    s.tokens++;
    return true;
}

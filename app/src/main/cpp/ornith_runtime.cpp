#include <jni.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <utility>
#include "mcnpu_backend.h"
#include "mlx_safetensors.h"
#include "ornith15_executor.h"
#include "ornith15_tokenizer.h"
#include "runtime_memory_budget.h"
#if MCNPU_HAS_LLAMA
extern "C" ggml_backend_reg_t mcnpu_ggml_backend_reg(void);
#endif
#if MCNPU_HAS_LLAMA
#include "llama.h"
#endif

namespace {

struct RuntimeState {
    std::string path;
    uint64_t fileBytes = 0;
    uint64_t context = 0;
    uint64_t blocks = 0;
    uint64_t hidden = 0;
    uint64_t vocab = 0;
    std::string arch;
    bool loaded = false;
    bool mlx_loaded = false;
    MlxSafetensorsInfo mlx_info;
    Ornith15TextConfig mlx_cfg;
    Ornith15LayerRuntime mlx_runtime;
    Ornith15Tokenizer tokenizer;
    uint64_t last_prompt_tokens = 0;
    uint64_t last_generated_tokens = 0;
    uint64_t last_tokenize_us = 0;
    uint64_t last_prefill_us = 0;
    uint64_t last_first_token_us = 0;
    uint64_t last_decode_us = 0;
    std::string last_generation_memory;
#if MCNPU_HAS_LLAMA
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
#endif
} g;

static bool readU32(std::ifstream &f, uint32_t &v) {
    unsigned char b[4];
    if (!f.read(reinterpret_cast<char *>(b), 4)) return false;
    v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);
    return true;
}
static bool readU64(std::ifstream &f, uint64_t &v) {
    unsigned char b[8];
    if (!f.read(reinterpret_cast<char *>(b), 8)) return false;
    v=0;
    for(int i=0;i<8;i++) v|=((uint64_t)b[i])<<(8*i);
    return true;
}
static bool skipString(std::ifstream &f, std::string *out=nullptr) {
    uint64_t n;
    if(!readU64(f,n) || n>(1u<<20)) return false;
    std::string s((size_t)n, '\0');
    if(n && !f.read(s.data(),(std::streamsize)n)) return false;
    if(out) *out=std::move(s);
    return true;
}
static bool skipValue(std::ifstream &f, uint32_t type, int depth=0) {
    if(depth>4) return false;
    uint64_t n;
    switch(type) {
        case 0: case 1: case 7: return f.ignore(1).good();
        case 2: case 3: return f.ignore(2).good();
        case 4: case 5: case 6: return f.ignore(4).good();
        case 8: return skipString(f);
        case 9: {
            uint32_t elem; uint64_t count;
            return readU32(f,elem) && readU64(f,count) && count<=1000000 &&
                   [&]() { for(uint64_t i=0;i<count;i++) if(!skipValue(f,elem,depth+1)) return false; return true; }();
        }
        case 10: case 11: case 12: return f.ignore(8).good();
        default: return false;
    }
}

static int32_t sample_mlx_logits(const std::vector<float> &logits, std::mt19937 &rng) {
    if (logits.empty()) return -1;
    constexpr size_t TOP_K = 20;
    std::vector<int32_t> top;
    top.reserve(TOP_K);
    for (int32_t id = 0; id < (int32_t)logits.size(); ++id) {
        if (!std::isfinite(logits[(size_t)id])) continue;
        auto it = top.begin();
        while (it != top.end() && logits[(size_t)*it] >= logits[(size_t)id]) ++it;
        top.insert(it, id);
        if (top.size() > TOP_K) top.pop_back();
    }
    if (top.empty()) return -1;
    constexpr float temperature = 0.6f;
    float mx = -INFINITY;
    for (int32_t id : top) mx = std::max(mx, logits[(size_t)id] / temperature);
    std::vector<float> p(top.size());
    float sum = 0.0f;
    for (size_t i = 0; i < top.size(); ++i) {
        p[i] = std::exp((logits[(size_t)top[i]] / temperature) - mx);
        sum += p[i];
    }
    if (sum <= 0.0f || !std::isfinite(sum)) return top.front();
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float pick = dist(rng);
    for (size_t i = 0; i < p.size(); ++i) {
        pick -= p[i];
        if (pick <= 0.0f) return top[i];
    }
    return top.back();
}

static std::string generateMlxModel(const std::string &prompt, int maxTokens) {
    if (!g.loaded || !g.mlx_loaded) return "ERR ORNITH15_RUNTIME not_loaded";
    if (prompt.empty()) return "ERR ORNITH15_RUNTIME empty_prompt";
    if (maxTokens <= 0 || maxTokens > 1024) return "ERR ORNITH15_RUNTIME bad_max_tokens";
    if (!g.tokenizer.loaded) return "ERR ORNITH15_RUNTIME tokenizer_not_loaded";

    g.last_prompt_tokens = 0;
    g.last_generated_tokens = 0;
    g.last_tokenize_us = 0;
    g.last_prefill_us = 0;
    g.last_first_token_us = 0;
    g.last_decode_us = 0;
    g.last_generation_memory.clear();
    mlx_reset_affine4_io_stats();

    const auto tokenize0 = std::chrono::steady_clock::now();
    const std::string chat =
        "<|im_start|>user\\n" + prompt + "<|im_end|>\\n"
        "<|im_start|>assistant\\n<think>\\n";
    std::vector<int32_t> ids;
    std::string error;
    if (!ornith15_tokenizer_encode(g.tokenizer, chat, ids, error) || ids.empty())
        return "ERR ORNITH15_RUNTIME tokenize=" + error;
    g.last_tokenize_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - tokenize0).count();
    g.last_prompt_tokens = (uint64_t)ids.size();

    // The model supports 262K positions, but the first phone target is a real
    // 64K resident attention window. The recurrent layers still carry state
    // across the whole replay; only full-attention K/V is windowed.
    if (ids.size() > g.context)
        return "ERR ORNITH15_RUNTIME prompt_context_exceeded=" + std::to_string(g.context);

    // Runtime state is allocated once when the model is loaded. Reuse the
    // 64K KV/recurrent buffers across turns; rebuilding them for every message
    // would add multi-gigabyte allocation churn before the first token.
    if (g.mlx_runtime.initialized_layers != g.mlx_cfg.num_layers) {
        if (!ornith15_executor_init_runtime(g.mlx_cfg,
                                            (uint32_t)g.context,
                                            g.mlx_runtime, error))
            return "ERR ORNITH15_RUNTIME runtime_init=" + error;
    }

    // Each generation replays the complete working prompt. Reset recurrent and
    // KV state first, otherwise the previous request would be silently replayed twice.
    ornith15_executor_reset_runtime(g.mlx_runtime);

    Ornith15DecoderStep step;
    const auto prefill0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!ornith15_executor_forward_token(g.path, g.mlx_info, g.mlx_cfg,
                                             (uint32_t)ids[i], (uint32_t)i,
                                             g.mlx_runtime, step, error))
            return "ERR ORNITH15_RUNTIME forward=" + error;
    }
    g.last_prefill_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - prefill0).count();

    std::mt19937 rng(0x4f524e49u);
    std::vector<int32_t> generated;
    generated.reserve((size_t)maxTokens);
    const auto decode0 = std::chrono::steady_clock::now();
    for (int i = 0; i < maxTokens; ++i) {
        const int32_t next = sample_mlx_logits(step.logits, rng);
        if (next < 0) return "ERR ORNITH15_RUNTIME sample_failed";
        if (next == g.tokenizer.eos) break;
        generated.push_back(next);
        if (generated.size() == 1) {
            g.last_first_token_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - tokenize0).count();
        }
        // No next-token logits are needed after the final requested token.
        // Avoid one complete extra 32-layer forward pass and keep decode timing
        // aligned with the work actually needed to produce generated tokens.
        if (i + 1 >= maxTokens) break;
        const uint32_t pos = (uint32_t)ids.size() + (uint32_t)i;
        if (!ornith15_executor_forward_token(g.path, g.mlx_info, g.mlx_cfg,
                                             (uint32_t)next, pos,
                                             g.mlx_runtime, step, error))
            return "ERR ORNITH15_RUNTIME forward=" + error;
    }
    g.last_decode_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - decode0).count();
    g.last_generated_tokens = (uint64_t)generated.size();
    g.last_generation_memory = ornith15_memory_status();
    const std::string out = ornith15_tokenizer_decode(g.tokenizer, generated);
    return "OK ORNITH15_GENERATE/1 text=" + out;
}
static std::string loadMlxModel(const std::string &path, uint64_t requested) {
    // A reload must be transactional. Never leave the previous model marked
    // loaded if validation or tokenizer preparation for the new file fails.
    g.loaded = false;
    g.mlx_loaded = false;
    g.mlx_info = MlxSafetensorsInfo{};
    g.mlx_cfg = Ornith15TextConfig{};
    g.mlx_runtime = Ornith15LayerRuntime{};
    g.tokenizer = Ornith15Tokenizer{};
    if (!mcnpu_backend_ready())
        return "ERR ORNITH15_MLX npu_not_ready "+mcnpu_backend_status();
    MlxSafetensorsInfo info;
    std::string error;
    if (!mlx_safetensors_probe(path, info, error))
        return "ERR ORNITH15_MLX " + error;
    // The single-model policy is enforced by the complete tensor-layout
    // validator below, not by a brittle byte-for-byte file-size assumption.
    if (info.file_bytes < (4ull << 30))
        return "ERR ORNITH15_MLX model_file_too_small=" + std::to_string(info.file_bytes);
    MlxQuantInfo q;
    if (!mlx_infer_affine4(info, q, error))
        return "ERR ORNITH15_MLX " + error;
    Ornith15TextConfig cfg;
    if (!ornith15_executor_validate(info, cfg, error))
        return "ERR ORNITH15_MLX executor_preflight=" + error;
    if (requested == 0 || requested > cfg.context_length)
        return "ERR ORNITH15_RUNTIME requested_context=" + std::to_string(requested) + " native=262144";
    constexpr uint64_t MAX_FIRST_STAGE_CONTEXT = 65536ull;
    if (requested > MAX_FIRST_STAGE_CONTEXT)
        return "ERR ORNITH15_RUNTIME requested_context=" + std::to_string(requested) + " first_stage_max=65536";

#if MCNPU_HAS_LLAMA
    if (g.sampler) { llama_sampler_free(g.sampler); g.sampler=nullptr; }
    if (g.ctx) { llama_free(g.ctx); g.ctx=nullptr; }
    if (g.model) { llama_model_free(g.model); g.model=nullptr; }
#endif
    std::string tokPath = path + ".tokenizer";
    Ornith15Tokenizer tok;
    if (!ornith15_tokenizer_load(tokPath, tok, error))
        return "ERR ORNITH15_MLX tokenizer=" + error;
    Ornith15LayerRuntime layerRuntime;
    if (!ornith15_executor_init_runtime(cfg,
                                        (uint32_t)requested,
                                        layerRuntime, error))
        return "ERR ORNITH15_MLX runtime_init=" + error;

    g.path = path;
    g.context = requested;
    g.fileBytes = info.file_bytes;
    g.arch = "qwen3_5";
    g.blocks = cfg.num_layers;
    g.hidden = cfg.hidden_size;
    g.vocab = cfg.vocab_size;
    g.mlx_info = std::move(info);
    g.mlx_cfg = cfg;
    g.mlx_runtime = std::move(layerRuntime);
    g.tokenizer = std::move(tok);
    g.mlx_loaded = true;
    g.loaded = true;
    const std::string npuProbe = mlx_affine4_npu_probe(g.path, g.mlx_info);
    if (npuProbe.rfind("OK MLX_NPU_PROBE/1", 0) != 0) {
        g.mlx_loaded = false;
        g.loaded = false;
        g.mlx_runtime = Ornith15LayerRuntime{};
        g.tokenizer = Ornith15Tokenizer{};
        return "ERR ORNITH15_MLX npu_probe=" + npuProbe;
    }
    return "OK ORNITH15_RUNTIME/1 format=MLX_SAFE_TENSORS_4BIT"
           " arch=qwen3_5 layers=32 hidden=4096 vocab=248320"
           " context=" + std::to_string(requested) +
           " attention_window=" + std::to_string(requested) +
           " kv_storage=fp16"
           " file_bytes=" + std::to_string(g.fileBytes) +
           " header_bytes=" + std::to_string(g.mlx_info.header_bytes) +
           " tensors=" + std::to_string(g.mlx_info.tensor_count) +
           " affine4=bits4_group64 weight_triplets=" + std::to_string(q.quantized_weight_count) +
           " npu=" + mcnpu_backend_status() +
           " weight_npu_probe=" + npuProbe +
           " inference=MLX4BIT_NPU_EXECUTOR";
}
static std::string loadModel(const std::string &path, uint64_t requested) {
    if (path.size() >= 11 && path.compare(path.size()-11, 11, ".safetensors") == 0)
        return loadMlxModel(path, requested);
#if MCNPU_HAS_LLAMA
#if MCNPU_HAS_LLAMA
    if (g.loaded) {
        if (g.sampler) { llama_sampler_free(g.sampler); g.sampler=nullptr; }
        if (g.ctx) { llama_free(g.ctx); g.ctx=nullptr; }
        if (g.model) { llama_model_free(g.model); g.model=nullptr; }
    }
#endif
    g.loaded=false;
    g.mlx_loaded = false;
    g.mlx_info = MlxSafetensorsInfo{};
    g.mlx_cfg = Ornith15TextConfig{};
    g.mlx_runtime = Ornith15LayerRuntime{};
    g.tokenizer = Ornith15Tokenizer{};
    std::ifstream f(path, std::ios::binary|std::ios::ate);
    if(!f) return "ERR ORNITH15_RUNTIME open_failed";
    const std::streamoff end=f.tellg();
    if(end<24) return "ERR ORNITH15_RUNTIME file_too_small";
    g.fileBytes=(uint64_t)end;
    f.seekg(0);

    char magic[4];
    if(!f.read(magic,4) || std::memcmp(magic,"GGUF",4)!=0)
        return "ERR ORNITH15_RUNTIME bad_magic";
    uint32_t version; uint64_t tensors,kvs;
    if(!readU32(f,version) || (version!=2 && version!=3) ||
       !readU64(f,tensors) || !readU64(f,kvs) || kvs>100000)
        return "ERR ORNITH15_RUNTIME bad_header";

    g.arch=""; g.blocks=g.hidden=g.vocab=0;
    for(uint64_t i=0;i<kvs;i++) {
        std::string key; uint32_t type;
        if(!skipString(f,&key) || !readU32(f,type))
            return "ERR ORNITH15_RUNTIME bad_kv";
        if(type==8) {
            std::string value;
            if(!skipString(f,&value)) return "ERR ORNITH15_RUNTIME bad_string";
            if(key=="general.architecture") g.arch=value;
        } else if(type==4 || type==5 || type==10 || type==11) {
            uint64_t value=0;
            if(type==4 || type==5) { uint32_t x; if(!readU32(f,x)) return "ERR ORNITH15_RUNTIME bad_u32"; value=x; }
            else if(!readU64(f,value)) return "ERR ORNITH15_RUNTIME bad_u64";
            if(key=="qwen35.block_count") g.blocks=value;
            if(key=="qwen35.embedding_length") g.hidden=value;
            if(key=="qwen35.vocab_size") g.vocab=value;
            if(key=="qwen35.context_length") g.context=value;
        } else {
            if(!skipValue(f,type)) return "ERR ORNITH15_RUNTIME bad_value";
        }
    }

    if(g.arch!="qwen35") return "ERR ORNITH15_RUNTIME arch="+(g.arch.empty()?"unknown":g.arch);
    if(g.blocks==0) return "ERR ORNITH15_RUNTIME missing_block_count";
    if(g.context==0) return "ERR ORNITH15_RUNTIME missing_context";
    if(requested>g.context) return "ERR ORNITH15_RUNTIME requested_context="+std::to_string(requested)+" native="+std::to_string(g.context);

#if MCNPU_HAS_LLAMA
    llama_backend_init();
    auto * mcnpu_reg = mcnpu_ggml_backend_reg();
    if (mcnpu_reg) {
        ggml_backend_register(mcnpu_reg);
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g.model = llama_model_load_from_file(path.c_str(), mp);
    if (!g.model) return "ERR ORNITH15_RUNTIME llama_model_load_failed";
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) requested;
    cp.n_batch = (uint32_t) std::min<uint64_t>(requested, 512);
    cp.n_ubatch = std::min<uint32_t>(cp.n_batch, 512);
    cp.n_seq_max = 1;
    g.ctx = llama_init_from_model(g.model, cp);
    if (!g.ctx) {
        llama_model_free(g.model); g.model=nullptr;
        return "ERR ORNITH15_RUNTIME llama_context_failed";
    }
    auto sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    g.sampler = llama_sampler_chain_init(sp);
    if (!g.sampler) {
        llama_free(g.ctx); g.ctx=nullptr;
        llama_model_free(g.model); g.model=nullptr;
        return "ERR ORNITH15_RUNTIME llama_sampler_failed";
    }
    llama_sampler_chain_add(g.sampler, llama_sampler_init_greedy());
#endif

    g.path=path; g.context=requested; g.loaded=true;
    return "OK ORNITH15_RUNTIME/1 arch="+g.arch+
           " layers="+std::to_string(g.blocks)+
           " hidden="+std::to_string(g.hidden)+
           " vocab="+std::to_string(g.vocab)+
           " context="+std::to_string(requested)+
           " file_bytes="+std::to_string(g.fileBytes)+
           " npu="+mcnpu_backend_status()+
           " ggml_backend=MCNPU_ACCEL_MUL_MAT_F32_QNN "+
           " inference=LLAMA_SCHEDULER_MCNPU_READY";
}

static std::string generateModel(const std::string &prompt, int maxTokens) {
    if (g.mlx_loaded) return generateMlxModel(prompt, maxTokens);
#if !MCNPU_HAS_LLAMA
    return "ERR ORNITH15_RUNTIME llama_backend_not_compiled";
#else
    if (!g.loaded || !g.model || !g.ctx || !g.sampler)
        return "ERR ORNITH15_RUNTIME not_loaded";
    if (maxTokens <= 0 || maxTokens > 4096)
        return "ERR ORNITH15_RUNTIME bad_max_tokens";
    const llama_vocab * vocab = llama_model_get_vocab(g.model);
    int n = -llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, true);
    if (n <= 0) return "ERR ORNITH15_RUNTIME tokenize_failed";
    std::vector<llama_token> tokens((size_t)n);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), n, true, true) < 0)
        return "ERR ORNITH15_RUNTIME tokenize_failed";
    llama_sampler_reset(g.sampler);
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(g.ctx, batch) != 0)
        return "ERR ORNITH15_RUNTIME prompt_decode_failed";
    std::string out;
    out.reserve((size_t)maxTokens * 4);
    for (int i=0; i<maxTokens; ++i) {
        llama_token tok = llama_sampler_sample(g.sampler, g.ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char buf[4096];
        int m = llama_token_to_piece(vocab, tok, buf, (int32_t)sizeof(buf), 0, false);
        if (m < 0) return "ERR ORNITH15_RUNTIME token_to_piece_failed";
        out.append(buf, (size_t)m);
        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(g.ctx, batch) != 0)
            return "ERR ORNITH15_RUNTIME decode_failed";
    }
    return "OK ORNITH15_GENERATE/1 text=" + out;
#endif
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_Ornith15Runtime_nativeLoad(JNIEnv* env,jclass,jstring jpath,jlong requested) {
    const char* p=env->GetStringUTFChars(jpath,nullptr);
    std::string s=loadModel(p?p:"",(uint64_t)requested);
    if(p) env->ReleaseStringUTFChars(jpath,p);
    return env->NewStringUTF(s.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_Ornith15Runtime_nativeGenerate(JNIEnv* env,jclass,jstring jprompt,jint maxTokens) {
    const char* p=env->GetStringUTFChars(jprompt,nullptr);
    std::string s=generateModel(p?p:"",(int)maxTokens);
    if(p) env->ReleaseStringUTFChars(jprompt,p);
    return env->NewStringUTF(s.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_Ornith15Runtime_nativeInfo(JNIEnv* env,jclass) {
    std::string s=g.loaded
        ? "OK ORNITH15_RUNTIME/1 loaded=true path="+g.path+
          " context="+std::to_string(g.context)+
          " attention_window="+std::to_string(g.context)+
          " kv_storage="+(g.mlx_loaded ? "fp16" : "llama")+
          " mlx="+(g.mlx_loaded?"true":"false")+
          " last_gen_prompt_tokens="+std::to_string(g.last_prompt_tokens)+
          " last_gen_tokens="+std::to_string(g.last_generated_tokens)+
          " last_tokenize_us="+std::to_string(g.last_tokenize_us)+
          " last_prefill_us="+std::to_string(g.last_prefill_us)+
          " last_first_token_us="+std::to_string(g.last_first_token_us)+
          " last_decode_us="+std::to_string(g.last_decode_us)+
          " last_gen_mem="+(g.last_generation_memory.empty() ? ornith15_memory_status() : g.last_generation_memory)+
          " affine4_io="+mlx_affine4_io_status()
        : "OK ORNITH15_RUNTIME/1 loaded=false";
    return env->NewStringUTF(s.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_bslsjdk_mcnpu_Ornith15Runtime_nativeUnload(JNIEnv*,jclass) {
#if MCNPU_HAS_LLAMA
    if (g.sampler) { llama_sampler_free(g.sampler); g.sampler=nullptr; }
    if (g.ctx) { llama_free(g.ctx); g.ctx=nullptr; }
    if (g.model) { llama_model_free(g.model); g.model=nullptr; }
#endif
    g=RuntimeState{};
}

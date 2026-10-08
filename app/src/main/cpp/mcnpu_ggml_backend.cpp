#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "mcnpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct BufferCtx { void * base = nullptr; };

static ggml_backend_reg g_reg{};
static ggml_backend_device g_dev{};
static ggml_backend_buffer_type g_buft{};
static std::once_flag g_once;

static const char * reg_name(ggml_backend_reg_t) { return "MCNPU"; }
static const char * dev_name(ggml_backend_dev_t) { return "MCNPU"; }
static const char * dev_desc(ggml_backend_dev_t) { return "Qualcomm QNN HTP V73 in-process NPU"; }

static void dev_memory(ggml_backend_dev_t, size_t * free_mem, size_t * total_mem) {
    if (free_mem) *free_mem = 0;
    if (total_mem) *total_mem = 0;
}

static enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * p) {
    std::memset(p, 0, sizeof(*p));
    p->name = dev_name(dev);
    p->description = dev_desc(dev);
    p->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    p->caps.async = false;
    p->caps.host_buffer = true;
    p->caps.buffer_from_host_ptr = true;
    p->caps.events = false;
    p->caps.mmap_support = false;
}

static const char * buft_name(ggml_backend_buffer_type_t) { return "MCNPU_HOST"; }
static size_t buft_align(ggml_backend_buffer_type_t) { return 64; }
static size_t buft_max(ggml_backend_buffer_type_t) { return SIZE_MAX; }
static size_t buft_alloc(ggml_backend_buffer_type_t, const ggml_tensor * t) { return ggml_nbytes(t); }
static bool buft_host(ggml_backend_buffer_type_t) { return true; }

static ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new BufferCtx();
    ctx->base = std::malloc(size ? size : 1);
    if (!ctx->base) { delete ctx; return nullptr; }

    ggml_backend_buffer_i iface{};
    iface.free_buffer = [](ggml_backend_buffer_t b) {
        auto * c = static_cast<BufferCtx *>(b->context);
        if (c) { std::free(c->base); delete c; }
    };
    iface.get_base = [](ggml_backend_buffer_t b) -> void * {
        auto * c = static_cast<BufferCtx *>(b->context);
        return c ? c->base : nullptr;
    };
    iface.memset_tensor = [](ggml_backend_buffer_t b, ggml_tensor *, uint8_t v, size_t off, size_t sz) {
        auto * c = static_cast<BufferCtx *>(b->context);
        if (c && c->base) std::memset(static_cast<uint8_t *>(c->base) + off, v, sz);
    };
    iface.set_tensor = [](ggml_backend_buffer_t b, ggml_tensor *, const void * src, size_t off, size_t sz) {
        auto * c = static_cast<BufferCtx *>(b->context);
        if (c && c->base && src) std::memcpy(static_cast<uint8_t *>(c->base) + off, src, sz);
    };
    iface.get_tensor = [](ggml_backend_buffer_t b, const ggml_tensor *, void * dst, size_t off, size_t sz) {
        auto * c = static_cast<BufferCtx *>(b->context);
        if (c && c->base && dst) std::memcpy(dst, static_cast<uint8_t *>(c->base) + off, sz);
    };
    iface.clear = [](ggml_backend_buffer_t b, uint8_t v) {
        auto * c = static_cast<BufferCtx *>(b->context);
        if (c && c->base) std::memset(c->base, v, b->size);
    };
    return ggml_backend_buffer_init(buft, iface, ctx, size);
}

static ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t) { return &g_buft; }

static bool supported_op(ggml_backend_dev_t, const ggml_tensor * op) {
    if (!mcnpu_backend_ready() || !op || op->op != GGML_OP_MUL_MAT) return false;
    if (op->n_dims != 2 || !op->src[0] || !op->src[1] || !op->data) return false;
    if (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    const int64_t k = op->src[0]->ne[0], m = op->src[0]->ne[1], n = op->src[1]->ne[1];
    return k > 0 && m > 0 && n > 0 && k <= 65536 && m <= 65536 && n <= 65536;
}

static bool supported_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_cpu_buffer_type() || buft == &g_buft;
}

static bool offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return supported_op(dev, op);
}

static ggml_backend_t dev_init(ggml_backend_dev_t dev, const char *) {
    auto * b = new ggml_backend{};
    b->device = dev;
    b->context = nullptr;
    b->iface.get_name = [](ggml_backend_t) { return "MCNPU"; };
    b->iface.free = [](ggml_backend_t b) { delete b; };
    b->iface.graph_compute = [](ggml_backend_t b, ggml_cgraph * g) -> enum ggml_status {
        (void) b;
        for (int i = 0; i < g->n_nodes; ++i) {
            ggml_tensor * op = g->nodes[i];
            if (!op || op->op != GGML_OP_MUL_MAT || !supported_op(&g_dev, op)) return GGML_STATUS_FAILED;

            const ggml_tensor * a = op->src[0];
            const ggml_tensor * w = op->src[1];
            const int64_t k = a->ne[0], m = a->ne[1], n = w->ne[1];
            const size_t na = (size_t)k * (size_t)m;
            const size_t nw = (size_t)k * (size_t)n;
            const size_t nc = (size_t)m * (size_t)n;

            std::vector<int8_t> qa(na), qw(nw), qc(nc);
            const float * fa = static_cast<const float *>(a->data);
            const float * fw = static_cast<const float *>(w->data);
            float ma = 0.0f, mw = 0.0f;
            for (size_t j = 0; j < na; ++j) ma = std::max(ma, std::fabs(fa[j]));
            for (size_t j = 0; j < nw; ++j) mw = std::max(mw, std::fabs(fw[j]));
            if (ma == 0.0f || mw == 0.0f) {
                std::memset(op->data, 0, nc * sizeof(float));
                continue;
            }
            const float sa = ma / 127.0f, sw = mw / 127.0f;
            for (size_t j = 0; j < na; ++j) qa[j] = (int8_t) std::lrintf(fa[j] / sa);
            for (size_t j = 0; j < nw; ++j) qw[j] = (int8_t) std::lrintf(fw[j] / sw);

            float so = 1.0f;
            const std::string r = mcnpu_backend_matmul_int8(
                qw.data(), qa.data(), qc.data(), (uint32_t)m, (uint32_t)k, (uint32_t)n, so);
            if (r.rfind("OK", 0) != 0) return GGML_STATUS_FAILED;

            float * out = static_cast<float *>(op->data);
            const float scale = sw * sa * so;
            for (size_t j = 0; j < nc; ++j) out[j] = (float) qc[j] * scale;
        }
        return GGML_STATUS_SUCCESS;
    };
    return b;
}

static ggml_backend_reg_t make_reg() {
    std::call_once(g_once, [] {
        g_buft = {};
        g_buft.iface.get_name = buft_name;
        g_buft.iface.alloc_buffer = buft_alloc_buffer;
        g_buft.iface.get_alignment = buft_align;
        g_buft.iface.get_max_size = buft_max;
        g_buft.iface.get_alloc_size = buft_alloc;
        g_buft.iface.is_host = buft_host;
        g_buft.device = &g_dev;

        g_dev = {};
        g_dev.iface.get_name = dev_name;
        g_dev.iface.get_description = dev_desc;
        g_dev.iface.get_memory = dev_memory;
        g_dev.iface.get_type = dev_type;
        g_dev.iface.get_props = dev_props;
        g_dev.iface.init_backend = dev_init;
        g_dev.iface.get_buffer_type = dev_buft;
        g_dev.iface.supports_op = supported_op;
        g_dev.iface.supports_buft = supported_buft;
        g_dev.iface.offload_op = offload_op;
        g_dev.reg = &g_reg;

        g_reg = {};
        g_reg.api_version = GGML_BACKEND_API_VERSION;
        g_reg.iface.get_name = reg_name;
        g_reg.iface.get_device_count = [](ggml_backend_reg_t) { return (size_t) 1; };
        g_reg.iface.get_device = [](ggml_backend_reg_t, size_t) { return &g_dev; };
    });
    return &g_reg;
}

} // namespace

extern "C" ggml_backend_reg_t mcnpu_ggml_backend_reg(void) {
    return make_reg();
}

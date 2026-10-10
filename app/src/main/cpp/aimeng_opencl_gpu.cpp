#include <jni.h>
#include <dlfcn.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>

namespace {
using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_ulong = uint64_t;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_mem_flags = cl_bitfield;
using cl_bool = cl_uint;
using cl_context_properties = intptr_t;
using cl_platform_id = void*;
using cl_device_id = void*;
using cl_context = void*;
using cl_command_queue = void*;
using cl_program = void*;
using cl_kernel = void*;
using cl_mem = void*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_int CL_DEVICE_NOT_FOUND = -1;
constexpr cl_device_type CL_DEVICE_TYPE_GPU = (1ULL << 2);
constexpr cl_mem_flags CL_MEM_READ_ONLY = (1ULL << 2);
constexpr cl_mem_flags CL_MEM_WRITE_ONLY = (1ULL << 1);
constexpr cl_bool CL_TRUE = 1;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;

using GetPlatformIDs = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using GetDeviceIDs = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
using GetDeviceInfo = cl_int (*)(cl_device_id, cl_uint, size_t, void*, size_t*);
using CreateContext = cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
using CreateCommandQueue = cl_command_queue (*)(cl_context, cl_device_id, cl_bitfield, cl_int*);
using CreateProgramWithSource = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
using BuildProgram = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
using GetProgramBuildInfo = cl_int (*)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
using CreateKernel = cl_kernel (*)(cl_program, const char*, cl_int*);
using CreateBuffer = cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
using EnqueueWriteBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const void*, void*);
using SetKernelArg = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
using EnqueueNDRangeKernel = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
using EnqueueReadBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const void*, void*);
using Finish = cl_int (*)(cl_command_queue);
using ReleaseMemObject = cl_int (*)(cl_mem);
using ReleaseKernel = cl_int (*)(cl_kernel);
using ReleaseProgram = cl_int (*)(cl_program);
using ReleaseCommandQueue = cl_int (*)(cl_command_queue);
using ReleaseContext = cl_int (*)(cl_context);

void* g_lib = nullptr;
cl_platform_id g_platform = nullptr;
cl_device_id g_device = nullptr;
cl_context g_context = nullptr;
cl_command_queue g_queue = nullptr;
cl_program g_program = nullptr;
cl_kernel g_kernel = nullptr;
std::string g_deviceName;
std::mutex g_mutex;
bool g_ready = false;

GetPlatformIDs pGetPlatformIDs = nullptr;
GetDeviceIDs pGetDeviceIDs = nullptr;
GetDeviceInfo pGetDeviceInfo = nullptr;
CreateContext pCreateContext = nullptr;
CreateCommandQueue pCreateCommandQueue = nullptr;
CreateProgramWithSource pCreateProgramWithSource = nullptr;
BuildProgram pBuildProgram = nullptr;
GetProgramBuildInfo pGetProgramBuildInfo = nullptr;
CreateKernel pCreateKernel = nullptr;
CreateBuffer pCreateBuffer = nullptr;
EnqueueWriteBuffer pEnqueueWriteBuffer = nullptr;
SetKernelArg pSetKernelArg = nullptr;
EnqueueNDRangeKernel pEnqueueNDRangeKernel = nullptr;
EnqueueReadBuffer pEnqueueReadBuffer = nullptr;
Finish pFinish = nullptr;
ReleaseMemObject pReleaseMemObject = nullptr;
ReleaseKernel pReleaseKernel = nullptr;
ReleaseProgram pReleaseProgram = nullptr;
ReleaseCommandQueue pReleaseCommandQueue = nullptr;
ReleaseContext pReleaseContext = nullptr;

template <typename T> bool symbol(T& target, const char* name) {
    target = reinterpret_cast<T>(dlsym(g_lib, name));
    return target != nullptr;
}

void releaseLocked() {
    if (g_kernel && pReleaseKernel) pReleaseKernel(g_kernel);
    if (g_program && pReleaseProgram) pReleaseProgram(g_program);
    if (g_queue && pReleaseCommandQueue) pReleaseCommandQueue(g_queue);
    if (g_context && pReleaseContext) pReleaseContext(g_context);
    g_kernel = nullptr; g_program = nullptr; g_queue = nullptr; g_context = nullptr;
    g_device = nullptr; g_platform = nullptr; g_ready = false;
    if (g_lib) dlclose(g_lib);
    g_lib = nullptr;
    g_deviceName.clear();
}

bool loadApi() {
#define LOAD_CL(name) if (!symbol(p##name, "cl" #name)) return false
    LOAD_CL(GetPlatformIDs); LOAD_CL(GetDeviceIDs); LOAD_CL(GetDeviceInfo);
    LOAD_CL(CreateContext); LOAD_CL(CreateCommandQueue); LOAD_CL(CreateProgramWithSource);
    LOAD_CL(BuildProgram); LOAD_CL(GetProgramBuildInfo); LOAD_CL(CreateKernel);
    LOAD_CL(CreateBuffer); LOAD_CL(EnqueueWriteBuffer); LOAD_CL(SetKernelArg);
    LOAD_CL(EnqueueNDRangeKernel); LOAD_CL(EnqueueReadBuffer); LOAD_CL(Finish);
    LOAD_CL(ReleaseMemObject); LOAD_CL(ReleaseKernel); LOAD_CL(ReleaseProgram);
    LOAD_CL(ReleaseCommandQueue); LOAD_CL(ReleaseContext);
#undef LOAD_CL
    return true;
}
} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_bslsjdk_ornithnpu_OpenClGpuBackend_nativeInit(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ready) return JNI_TRUE;
    releaseLocked();
    g_lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_lib) g_lib = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!g_lib || !loadApi()) { releaseLocked(); return JNI_FALSE; }

    cl_uint platformCount = 0;
    if (pGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0) {
        releaseLocked(); return JNI_FALSE;
    }
    std::vector<cl_platform_id> platforms(platformCount);
    if (pGetPlatformIDs(platformCount, platforms.data(), nullptr) != CL_SUCCESS) {
        releaseLocked(); return JNI_FALSE;
    }
    for (cl_platform_id platform : platforms) {
        cl_uint deviceCount = 0;
        cl_int rc = pGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        if (rc != CL_SUCCESS || deviceCount == 0) continue;
        std::vector<cl_device_id> devices(deviceCount);
        if (pGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr) != CL_SUCCESS) continue;
        g_platform = platform; g_device = devices[0]; break;
    }
    if (!g_device) { releaseLocked(); return JNI_FALSE; }

    cl_int err = CL_SUCCESS;
    g_context = pCreateContext(nullptr, 1, &g_device, nullptr, nullptr, &err);
    if (!g_context || err != CL_SUCCESS) { releaseLocked(); return JNI_FALSE; }
    g_queue = pCreateCommandQueue(g_context, g_device, 0, &err);
    if (!g_queue || err != CL_SUCCESS) { releaseLocked(); return JNI_FALSE; }

    const char* source =
        "__kernel void aimeng_linear(__global const float* W, __global const float* x, "
        "__global const float* b, __global float* y, const int outN, const int inN) {"
        " size_t o=get_global_id(0); if(o >= (size_t)outN) return; "
        " float sum=b[o]; for(int i=0;i<inN;i++) sum += W[o*(size_t)inN+i]*x[i]; y[o]=sum; }";
    size_t sourceLength = std::strlen(source);
    g_program = pCreateProgramWithSource(g_context, 1, &source, &sourceLength, &err);
    if (!g_program || err != CL_SUCCESS) { releaseLocked(); return JNI_FALSE; }
    err = pBuildProgram(g_program, 1, &g_device, "-cl-std=CL1.2", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        pGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        if (logSize > 0 && logSize < 16384) {
            std::vector<char> log(logSize + 1, 0);
            pGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            __android_log_print(ANDROID_LOG_WARN, "AIMENG-GPU", "OpenCL build failed: %s", log.data());
        }
        releaseLocked(); return JNI_FALSE;
    }
    g_kernel = pCreateKernel(g_program, "aimeng_linear", &err);
    if (!g_kernel || err != CL_SUCCESS) { releaseLocked(); return JNI_FALSE; }
    size_t nameSize = 0;
    if (pGetDeviceInfo(g_device, CL_DEVICE_NAME, 0, nullptr, &nameSize) == CL_SUCCESS && nameSize > 0 && nameSize < 1024) {
        std::vector<char> name(nameSize + 1, 0);
        pGetDeviceInfo(g_device, CL_DEVICE_NAME, nameSize, name.data(), nullptr);
        g_deviceName = name.data();
    } else g_deviceName = "OpenCL GPU";
    g_ready = true;
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_ornithnpu_OpenClGpuBackend_nativeDeviceName(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return env->NewStringUTF(g_deviceName.empty() ? "OpenCL GPU" : g_deviceName.c_str());
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_bslsjdk_ornithnpu_OpenClGpuBackend_nativeLinear(JNIEnv* env, jclass,
        jfloatArray matrixArray, jfloatArray inputArray, jfloatArray biasArray, jint out, jint in) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ready || !matrixArray || !inputArray || out <= 0 || in <= 0 ||
        static_cast<int64_t>(out) * in > 16LL * 1024 * 1024) return nullptr;
    const jsize matrixLength = env->GetArrayLength(matrixArray);
    const jsize inputLength = env->GetArrayLength(inputArray);
    if (matrixLength != static_cast<int64_t>(out) * in || inputLength != in) return nullptr;
    const bool hasBias = biasArray != nullptr;
    if (hasBias && env->GetArrayLength(biasArray) != out) return nullptr;

    std::vector<float> matrix(matrixLength), input(in), bias(out, 0.f), output(out, 0.f);
    env->GetFloatArrayRegion(matrixArray, 0, matrixLength, matrix.data());
    env->GetFloatArrayRegion(inputArray, 0, inputLength, input.data());
    if (hasBias) env->GetFloatArrayRegion(biasArray, 0, out, bias.data());
    if (env->ExceptionCheck()) { env->ExceptionClear(); return nullptr; }

    cl_int err = CL_SUCCESS;
    cl_mem wm = pCreateBuffer(g_context, CL_MEM_READ_ONLY, matrix.size() * sizeof(float), nullptr, &err);
    if (!wm || err != CL_SUCCESS) return nullptr;
    cl_mem xm = pCreateBuffer(g_context, CL_MEM_READ_ONLY, input.size() * sizeof(float), nullptr, &err);
    if (!xm || err != CL_SUCCESS) { pReleaseMemObject(wm); return nullptr; }
    cl_mem bm = pCreateBuffer(g_context, CL_MEM_READ_ONLY, bias.size() * sizeof(float), nullptr, &err);
    if (!bm || err != CL_SUCCESS) { pReleaseMemObject(wm); pReleaseMemObject(xm); return nullptr; }
    cl_mem ym = pCreateBuffer(g_context, CL_MEM_WRITE_ONLY, output.size() * sizeof(float), nullptr, &err);
    if (!ym || err != CL_SUCCESS) { pReleaseMemObject(wm); pReleaseMemObject(xm); pReleaseMemObject(bm); return nullptr; }

    bool ok = pEnqueueWriteBuffer(g_queue, wm, CL_TRUE, 0, matrix.size() * sizeof(float), matrix.data(), 0, nullptr, nullptr) == CL_SUCCESS
        && pEnqueueWriteBuffer(g_queue, xm, CL_TRUE, 0, input.size() * sizeof(float), input.data(), 0, nullptr, nullptr) == CL_SUCCESS
        && pEnqueueWriteBuffer(g_queue, bm, CL_TRUE, 0, bias.size() * sizeof(float), bias.data(), 0, nullptr, nullptr) == CL_SUCCESS;
    const cl_int outN = out, inN = in;
    ok = ok && pSetKernelArg(g_kernel, 0, sizeof(cl_mem), &wm) == CL_SUCCESS
        && pSetKernelArg(g_kernel, 1, sizeof(cl_mem), &xm) == CL_SUCCESS
        && pSetKernelArg(g_kernel, 2, sizeof(cl_mem), &bm) == CL_SUCCESS
        && pSetKernelArg(g_kernel, 3, sizeof(cl_mem), &ym) == CL_SUCCESS
        && pSetKernelArg(g_kernel, 4, sizeof(cl_int), &outN) == CL_SUCCESS
        && pSetKernelArg(g_kernel, 5, sizeof(cl_int), &inN) == CL_SUCCESS;
    const size_t global = static_cast<size_t>(out);
    ok = ok && pEnqueueNDRangeKernel(g_queue, g_kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr) == CL_SUCCESS
        && pFinish(g_queue) == CL_SUCCESS
        && pEnqueueReadBuffer(g_queue, ym, CL_TRUE, 0, output.size() * sizeof(float), output.data(), 0, nullptr, nullptr) == CL_SUCCESS;
    pReleaseMemObject(wm); pReleaseMemObject(xm); pReleaseMemObject(bm); pReleaseMemObject(ym);
    if (!ok) return nullptr;
    jfloatArray result = env->NewFloatArray(out);
    if (result) env->SetFloatArrayRegion(result, 0, out, output.data());
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_bslsjdk_ornithnpu_OpenClGpuBackend_nativeShutdown(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    releaseLocked();
}

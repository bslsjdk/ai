#include <jni.h>
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {
std::mutex gMutex;
EGLDisplay gDisplay = EGL_NO_DISPLAY;
EGLContext gContext = EGL_NO_CONTEXT;
EGLSurface gSurface = EGL_NO_SURFACE;
GLuint gProgram = 0;
std::string gError = "not initialized";
const char* kShader =
    "#version 310 es\n"
    "layout(local_size_x=8, local_size_y=8, local_size_z=1) in;\n"
    "layout(std430, binding=0) readonly buffer ABlock { float A[]; };\n"
    "layout(std430, binding=1) readonly buffer BBlock { float B[]; };\n"
    "layout(std430, binding=2) writeonly buffer CBlock { float C[]; };\n"
    "uniform int M; uniform int K; uniform int N;\n"
    "void main(){ uint r=gl_GlobalInvocationID.y; uint c=gl_GlobalInvocationID.x; "
    "if(r>=uint(M)||c>=uint(N)) return; float s=0.0; "
    "for(int i=0;i<K;i++) s += A[int(r)*K+i]*B[i*N+int(c)]; "
    "C[int(r)*N+int(c)]=s; }\n";

bool initEgl() {
    if (gDisplay != EGL_NO_DISPLAY && gContext != EGL_NO_CONTEXT && gSurface != EGL_NO_SURFACE)
        return eglMakeCurrent(gDisplay, gSurface, gSurface, gContext) == EGL_TRUE;
    gDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (gDisplay == EGL_NO_DISPLAY) { gError = "ERR_EGL_NO_DISPLAY"; return false; }
    EGLint major=0, minor=0;
    if (!eglInitialize(gDisplay, &major, &minor)) { gError = "ERR_EGL_INITIALIZE"; return false; }
    if (!eglBindAPI(EGL_OPENGL_ES_API)) { gError = "ERR_EGL_BIND_GLES"; return false; }
    const EGLint cfgAttrs[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, 0x0040,
        EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_NONE };
    EGLConfig cfg=nullptr; EGLint count=0;
    if (!eglChooseConfig(gDisplay,cfgAttrs,&cfg,1,&count) || count<1) {
        gError = "ERR_EGL_NO_ES31_CONFIG"; return false;
    }
    const EGLint ctxAttrs[] = { EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE };
    gContext=eglCreateContext(gDisplay,cfg,EGL_NO_CONTEXT,ctxAttrs);
    if(gContext==EGL_NO_CONTEXT){gError="ERR_EGL_CREATE_CONTEXT";return false;}
    const EGLint surfAttrs[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
    gSurface=eglCreatePbufferSurface(gDisplay,cfg,surfAttrs);
    if(gSurface==EGL_NO_SURFACE){gError="ERR_EGL_CREATE_PBUFFER";return false;}
    if(!eglMakeCurrent(gDisplay,gSurface,gSurface,gContext)){gError="ERR_EGL_MAKE_CURRENT";return false;}
    const char* version=reinterpret_cast<const char*>(glGetString(GL_VERSION));
    if(!version){gError="ERR_GLES_VERSION_NULL";return false;}
    GLint majorGl=0,minorGl=0;
    glGetIntegerv(GL_MAJOR_VERSION,&majorGl); glGetIntegerv(GL_MINOR_VERSION,&minorGl);
    if(majorGl<3 || (majorGl==3 && minorGl<1)){gError=std::string("ERR_GLES_COMPUTE_UNSUPPORTED version=")+version;return false;}
    return true;
}
bool compileProgram() {
    if(gProgram) return true;
    GLuint shader=glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader,1,&kShader,nullptr); glCompileShader(shader);
    GLint ok=0; glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
    if(!ok){ GLint len=0; glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&len); std::vector<char> msg(std::max(1,len)); glGetShaderInfoLog(shader,len,nullptr,msg.data()); gError=std::string("ERR_GPU_SHADER_COMPILE ")+msg.data(); glDeleteShader(shader); return false; }
    gProgram=glCreateProgram(); glAttachShader(gProgram,shader); glLinkProgram(gProgram); glDeleteShader(shader);
    glGetProgramiv(gProgram,GL_LINK_STATUS,&ok);
    if(!ok){ GLint len=0; glGetProgramiv(gProgram,GL_INFO_LOG_LENGTH,&len); std::vector<char> msg(std::max(1,len)); glGetProgramInfoLog(gProgram,len,nullptr,msg.data()); gError=std::string("ERR_GPU_PROGRAM_LINK ")+msg.data(); glDeleteProgram(gProgram); gProgram=0; return false; }
    return true;
}
void setError(const std::string& s){gError=s;}
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_bslsjdk_ornithnpu_GpuComputeRuntime_nativeMatMul(JNIEnv* env,jclass,jfloatArray ja,jfloatArray jb,jint m,jint k,jint n){
    std::lock_guard<std::mutex> lock(gMutex);
    if(!ja||!jb||m<=0||k<=0||n<=0||(int64_t)m*k!=env->GetArrayLength(ja)||(int64_t)k*n!=env->GetArrayLength(jb)){
        setError("ERR_GPU_BAD_SHAPE");return nullptr;
    }
    const uint64_t aCount=(uint64_t)m*k,bCount=(uint64_t)k*n,cCount=(uint64_t)m*n;
    if((aCount+bCount+cCount)*sizeof(float)>64ULL*1024*1024){setError("ERR_GPU_WORKSET_LIMIT");return nullptr;}
    if(!initEgl()||!compileProgram()) return nullptr;
    std::vector<float> a(aCount),b(bCount),c(cCount);
    env->GetFloatArrayRegion(ja,0,(jsize)aCount,a.data());
    env->GetFloatArrayRegion(jb,0,(jsize)bCount,b.data());
    if(env->ExceptionCheck()){env->ExceptionClear();setError("ERR_GPU_INPUT_COPY");return nullptr;}
    GLuint buffers[3]={0,0,0}; glGenBuffers(3,buffers);
    const uint64_t sizes[3]={aCount*sizeof(float),bCount*sizeof(float),cCount*sizeof(float)};
    const void* data[3]={a.data(),b.data(),nullptr};
    for(int i=0;i<3;i++){glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[i]);glBufferData(GL_SHADER_STORAGE_BUFFER,(GLsizeiptr)sizes[i],data[i],i==2?GL_DYNAMIC_READ:GL_STATIC_DRAW);glBindBufferBase(GL_SHADER_STORAGE_BUFFER,i,buffers[i]);}
    glUseProgram(gProgram);
    glUniform1i(glGetUniformLocation(gProgram,"M"),m);glUniform1i(glGetUniformLocation(gProgram,"K"),k);glUniform1i(glGetUniformLocation(gProgram,"N"),n);
    glDispatchCompute((GLuint)((n+7)/8),(GLuint)((m+7)/8),1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
    glFinish();
    GLenum err=glGetError();
    if(err!=GL_NO_ERROR){for(GLuint id:buffers)if(id)glDeleteBuffers(1,&id);setError("ERR_GPU_GL_ERROR code="+std::to_string((unsigned)err));return nullptr;}
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffers[2]);
    void* mapped=glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,(GLsizeiptr)sizes[2],GL_MAP_READ_BIT);
    if(!mapped){for(GLuint id:buffers)if(id)glDeleteBuffers(1,&id);setError("ERR_GPU_MAP_RESULT");return nullptr;}
    std::memcpy(c.data(),mapped,(size_t)sizes[2]); glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    for(GLuint id:buffers)if(id)glDeleteBuffers(1,&id);
    jfloatArray out=env->NewFloatArray((jsize)cCount);
    if(!out){setError("ERR_GPU_ALLOC_JAVA_OUTPUT");return nullptr;}
    env->SetFloatArrayRegion(out,0,(jsize)cCount,c.data());
    if(env->ExceptionCheck()){env->ExceptionClear();setError("ERR_GPU_COPY_OUTPUT");return nullptr;}
    setError("OK");
    return out;
}
extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_ornithnpu_GpuComputeRuntime_nativeLastError(JNIEnv* env,jclass){
    std::lock_guard<std::mutex> lock(gMutex);return env->NewStringUTF(gError.c_str());
}
extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_ornithnpu_GpuComputeRuntime_nativeStatus(JNIEnv* env,jclass){
    std::lock_guard<std::mutex> lock(gMutex);
    if(!initEgl()) return env->NewStringUTF(gError.c_str());
    const char* version=reinterpret_cast<const char*>(glGetString(GL_VERSION));
    const char* renderer=reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const char* vendor=reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    GLint major=0,minor=0;glGetIntegerv(GL_MAJOR_VERSION,&major);glGetIntegerv(GL_MINOR_VERSION,&minor);
    std::string result="GPU_GLES_CONTEXT_READY version="+std::to_string(major)+"."+std::to_string(minor)
      +" vendor="+(vendor?vendor:"unknown")+" renderer="+(renderer?renderer:"unknown")
      +" gl_version="+(version?version:"unknown");
    return env->NewStringUTF(result.c_str());
}

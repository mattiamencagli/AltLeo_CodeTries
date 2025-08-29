#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// #define DEBUG

#define WIDTH 1024
#define HEIGHT 1024
#define CHANNELS 4

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        exit(code);
    }
}

#define GL_SAFE_CALL(ans) { OpenGlAssert((ans), __FILE__, __LINE__); }
inline void OpenGlAssert(GLenum code, const char *file, int line) {
    if (code != GL_NO_ERROR) {
        std::cerr << "OpenGlAssert: " << std::hex << code << " " << file << " " << line << std::endl;
        exit(code);
    }
}

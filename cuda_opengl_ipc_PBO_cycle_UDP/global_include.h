#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#define DEBUG

#define WIDTH 1024
#define HEIGHT 1024

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



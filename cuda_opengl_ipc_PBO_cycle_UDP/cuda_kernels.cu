// cuda_kernels.cu
#include <cuda_runtime.h>

__global__ void fill_matrix(unsigned char* data, int width, int height, int frame, int speed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        data[idx] = (x + y + frame*speed) % 256;
    }
}

// Wrapper per chiamata da C++
extern "C" void launch_fill_matrix(unsigned char* d_ptr, int width, int height, int frame, int speed) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    fill_matrix<<<grid, block>>>(d_ptr, width, height, frame, speed);
    cudaDeviceSynchronize();
}

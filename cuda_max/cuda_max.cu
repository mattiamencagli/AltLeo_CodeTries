#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define N_THREADS 512
#define N_BLOCKS 24
static const int blockSize = 512;
static const int gridSize = 24;

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


__device__ bool lastBlockMaxAbs(unsigned int* counter) {
    __threadfence();
    int last = 0;
    if(threadIdx.x == 0)
        last = atomicAdd(counter,1);
    return __syncthreads_or(last == gridDim.x-1);
}

__global__ void localmaxabs(float *input, const int arraySize, float *output, unsigned int* lastBlockCounter) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    float value;

    float valuemax = 0;

    int h = gthIdx - (gthIdx/1280)*1280;
    int k = gthIdx/1280;

    for(int i = gthIdx; i < arraySize; i += gridSize)
    {
        value = fabs(input[i]);
        if(value >  valuemax  && h>=4  &&h<1276 && k>=4 && k<716) { //Esclusione dei bordi dal calcolo del max
            valuemax = value;

        }
        output[i] = 0.0f;
    }

    __shared__  float shArr[blockSize];
    shArr[thIdx] = valuemax;
    __syncthreads();
    for(int size = blockSize/2; size>0; size/=2) {
        if(thIdx<size)
        {
            float value1 = shArr[thIdx];
            float value2 = shArr[thIdx+size];
            if(value2 > value1)
            {
                shArr[thIdx]=value2;
            }
        }
        __syncthreads();
    }
    if(thIdx == 0)
        output[blockIdx.x] = shArr[0];
    if(lastBlockMaxAbs(lastBlockCounter)) {
        shArr[thIdx] = thIdx<gridSize ? output[thIdx] : 0;
        __syncthreads();
        for(int size = blockSize/2; size>0; size/=2) {
            if(thIdx<size)
            {
                float value1 = shArr[thIdx];
                float value2 = shArr[thIdx+size];
                if(value2 > value1)
                {
                    shArr[thIdx]=value2;
                }

            }
            __syncthreads();
        }
        if(thIdx == 0)
            output[0] = shArr[0];
    }
}



int main(){

	int deviceId, numberOfSMs;
	CUDA_SAFE_CALL(cudaGetDevice(&deviceId));
	CUDA_SAFE_CALL(cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));
	printf("device ID = %d; Number of SMs = %d\n", deviceId, numberOfSMs);

	cudaEvent_t start, stop;
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	//cudaStream_t cudaStream;
	//CUDA_SAFE_CALL(cudaStreamCreate(&cudaStream));

	const int nx = 1280;
	const int ny = 720;
	const int N  = nx*ny;

	float * im = (float*) malloc(N*sizeof(float));
	memset(im, 0, N*sizeof(float));
	im[1401]   = 7.0f;
	im[2008]   = 9.0f;
	im[3402]   = 11.0f;
	im[7007]   = 13.0f;
	im[9999]   = 15.0f;
	im[12345]  = 17.0f;
	im[98765]  = 19.0f;
	im[123456] = 21.0f;
	im[500500] = 23.0f;
	im[920123] = 42.0f;

	float * im_gpu;
	CUDA_SAFE_CALL(cudaMalloc<float>(&im_gpu, N*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(im_gpu, im, N*sizeof(float), cudaMemcpyHostToDevice));

	float * MAX_gpu;
	CUDA_SAFE_CALL(cudaMalloc<float>(&MAX_gpu, sizeof(float)));

	unsigned int * lastblockcounter;
   	CUDA_SAFE_CALL(cudaMalloc(&lastblockcounter, sizeof(unsigned int)));
	
	CUDA_SAFE_CALL(cudaEventRecord(start, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(start));
	localmaxabs<<<gridSize,blockSize>>>(im_gpu, N, MAX_gpu, lastblockcounter);
    CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	float time_ms;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&time_ms, start, stop));

	float MAX;
	cudaMemcpy(MAX_gpu, &MAX, sizeof(float), cudaMemcpyDeviceToHost);

	printf("MAX = %f (expected=42);   time = %f ms\n",MAX, time_ms);

	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(stop));
	//CUDA_SAFE_CALL(cudaStreamDestroy(&cudaStream));

}


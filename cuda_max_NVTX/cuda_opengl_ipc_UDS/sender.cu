// cuda_writer.cpp

#include <QCoreApplication>
#include <QSemaphore>
#include <QLocalSocket>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define WIDTH  512
#define HEIGHT 512

__global__ void fill_matrix(unsigned char* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        data[idx] = (x + y) % 256;
    }
}

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);

    // Alloca memoria GPU
    unsigned char* d_matrix;
    cudaMalloc(&d_matrix, WIDTH * HEIGHT);

    // Lancia kernel
    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    fill_matrix<<<grid, block>>>(d_matrix, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Crea handle IPC
    cudaIpcMemHandle_t memHandle;
    cudaIpcGetMemHandle(&memHandle, d_matrix);

    // Invia handle via QLocalSocket
    QLocalSocket socket;
    socket.connectToServer("cuda_ipc_server");
    if (!socket.waitForConnected(1000)) {
        std::cerr << "Connection failed\n";
        return -1;
    }
    socket.write(reinterpret_cast<char*>(&memHandle), sizeof(memHandle));
    socket.flush();
    socket.waitForBytesWritten();

    //QUdpSocket _udpSocket;
    //_udpSocket.writeDatagram(reinterpret_cast<char*>(&memHandle), sizeof(memHandle), QHostAddress("127.0.0.1"), 9999);


    std::cout << "Handle sent, waiting...\n";

    // Attendi il lettore
    QSemaphore semaphore(0);
    semaphore.acquire(); // qui per semplicità si assume sincronizzazione esterna

    // Libera risorse
    cudaFree(d_matrix);
    return 0;
}

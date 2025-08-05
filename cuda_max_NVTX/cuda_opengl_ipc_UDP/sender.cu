// cuda_writer.cpp

#include <QApplication>
#include <QCoreApplication>
#include <QSemaphore>
#include <QUdpSocket>
#include <QHostAddress>
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

    // Invia handle via QUdpSocket
    QUdpSocket udpSocket;
    QByteArray datagram(reinterpret_cast<char*>(&memHandle), sizeof(memHandle));
    udpSocket.writeDatagram(datagram, QHostAddress("127.0.0.1"), 9999);

    std::cout << "Handle sent via UDP, waiting for ACK...\n";

    // Attendi a oltranza
    // QSemaphore semaphore(0);
    // semaphore.acquire(); // qui per semplicità si assume sincronizzazione esterna

    // Ascolta l'ACK sulla porta 9998
    QUdpSocket ackSocket;
    if (!ackSocket.bind(9998)) {
        std::cerr << "Failed to bind to port 9998 for ACK\n";
        return -1;
    }

    bool ackReceived = false;

    while (!ackReceived) {
        while (ackSocket.hasPendingDatagrams()) {
            QByteArray ackDatagram;
            ackDatagram.resize(int(ackSocket.pendingDatagramSize()));
            ackSocket.readDatagram(ackDatagram.data(), ackDatagram.size());

            if (ackDatagram == "ACK") {
                ackReceived = true;
                std::cout << "ACK received from receiver. Exiting.\n";
                break;
            }
        }
    }

    // Libera risorse
    cudaFree(d_matrix);
    return 0;
}

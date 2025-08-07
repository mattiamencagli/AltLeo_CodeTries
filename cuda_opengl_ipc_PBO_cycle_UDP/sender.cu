#include <QCoreApplication>
#include <QUdpSocket>
#include <QTimer>
#include <QHostAddress>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define WIDTH 1024
#define HEIGHT 1024

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void fill_matrix(unsigned char* data, int width, int height, int frame, int speed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        data[idx] = (x + y + frame*speed) % 256;
    }
}

class CudaSender : public QObject {
    Q_OBJECT

public:
    CudaSender() {
        ackSocket.bind(9998);

        CUDA_SAFE_CALL(cudaMalloc(&d_working, size));
        CUDA_SAFE_CALL(cudaMalloc(&d_framesend, size));

        CUDA_SAFE_CALL(cudaIpcGetMemHandle(&memHandle, d_framesend));

        QByteArray datagram(reinterpret_cast<char*>(&memHandle), sizeof(memHandle));
        udpSocket.writeDatagram(datagram, QHostAddress("127.0.0.1"), 9999);

        connect(&timer, &QTimer::timeout, this, &CudaSender::sendFrame);
        timer.start(33);

        connect(&ackSocket, &QUdpSocket::readyRead, this, &CudaSender::handleIncoming);
    }

    ~CudaSender() {
        timer.stop();
        CUDA_SAFE_CALL(cudaFree(d_working));
        CUDA_SAFE_CALL(cudaFree(d_framesend));
    }

public slots:
    void sendFrame() {
        frame++;

        dim3 block(16, 16);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
        fill_matrix<<<grid, block>>>(d_working, WIDTH, HEIGHT, frame, 1);
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        unsigned char h_debug;
        cudaMemcpy(&h_debug, d_working, 1, cudaMemcpyDeviceToHost);

        CUDA_SAFE_CALL(cudaMemcpy(d_framesend, d_working, size, cudaMemcpyDeviceToDevice));

        udpSocket.writeDatagram("FRAME_READY", QHostAddress("127.0.0.1"), 9999);
    }

    void handleIncoming() {
        while (ackSocket.hasPendingDatagrams()) {
            QByteArray datagram;
            datagram.resize(ackSocket.pendingDatagramSize());
            ackSocket.readDatagram(datagram.data(), datagram.size());

            if (datagram == "STOP") {
                std::cout << "Received STOP. Exiting...\n";
                QCoreApplication::quit();
            }
        }
    }

private:
    int frame = 0;
    const size_t size = WIDTH * HEIGHT;
    unsigned char *d_working = nullptr;
    unsigned char *d_framesend = nullptr;
    cudaIpcMemHandle_t memHandle;
    QUdpSocket udpSocket, ackSocket;
    QTimer timer;
};

#include ".sender.moc"

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    CudaSender sender;
    return app.exec();
}

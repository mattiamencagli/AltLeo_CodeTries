#include <QCoreApplication>
#include <QUdpSocket>
#include <QTimer>
#include <QHostAddress>

#include "global_include.h"

__global__ void fill_matrix_linear(unsigned char* data, int width, int height, int frame) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        data[idx] = (x + y + frame) % 256;
    }
}

__global__ void fill_matrix_spiral(unsigned char* data, int width, int height, int frame) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        float cx = width * 0.5f;
        float cy = height * 0.5f;
        float dx = x - cx;
        float dy = y - cy;
        float r = sqrtf(dx * dx + dy * dy);
        float theta = atan2f(dy, dx);
        float t = frame * 0.2f;
        // float spiral = sinf(r * 0.05f + theta - t);
        float spiral = sinf( 10.0f * logf(r + 1e-6f) + theta - t);
        int val = int((spiral + 1.0f) * 0.5f * 255.0f);
        data[idx] = (unsigned char)(val % 256);
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
        fill_matrix_spiral<<<grid, block>>>(d_working, WIDTH, HEIGHT, frame);
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        #ifdef DEBUG
            unsigned char* h_debug = new unsigned char[10];
            CUDA_SAFE_CALL(cudaMemcpy(h_debug, d_working, 10, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 3; ++i) {
                std::cout << "SENDER - h_debug[" << i << "] = " << (int)h_debug[i] << std::endl;
            }
        #endif

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

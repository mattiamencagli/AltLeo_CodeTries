#include <QCoreApplication>
#include <QUdpSocket>
#include <QTimer>
#include <QHostAddress>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define WIDTH 1024
#define HEIGHT 1024

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
        ackSocket.bind(9998); // per ricevere STOP

        // Alloca memoria GPU
        cudaMalloc(&d_matrix, WIDTH * HEIGHT);

        // Crea handle IPC
        cudaIpcGetMemHandle(&memHandle, d_matrix);

        // Invia handle al receiver una volta sola
        QByteArray datagram(reinterpret_cast<char*>(&memHandle), sizeof(memHandle));
        udpSocket.writeDatagram(datagram, QHostAddress("127.0.0.1"), 9999);

        // Timer per inviare frame continui
        connect(&timer, &QTimer::timeout, this, &CudaSender::sendFrame);
        timer.start(33); // circa 30 FPS

        // Ascolta per messaggi in arrivo (tipo "STOP")
        connect(&ackSocket, &QUdpSocket::readyRead, this, &CudaSender::handleIncoming);
    }

    ~CudaSender() {
        timer.stop();
        cudaFree(d_matrix);
    }

public slots:
    void sendFrame() {
        static int frame = 0;
        frame++;

        dim3 block(16, 16);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
        fill_matrix<<<grid, block>>>(d_matrix, WIDTH, HEIGHT, frame, 4);
        cudaDeviceSynchronize();

        // Notifica che un nuovo frame è pronto
        QByteArray msg("FRAME_READY");
        udpSocket.writeDatagram(msg, QHostAddress("127.0.0.1"), 9999);
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
    unsigned char* d_matrix = nullptr;
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

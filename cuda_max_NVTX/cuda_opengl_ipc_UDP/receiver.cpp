// opengl_reader.cpp

#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QUdpSocket>
#include <QSemaphore>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define WIDTH  512
#define HEIGHT 512

class Renderer : public QOpenGLWidget, protected QOpenGLFunctions {
public:
    unsigned char* d_matrix = nullptr;

    void setCudaPointer(unsigned char* ptr) { d_matrix = ptr; }

protected:
    void initializeGL() override {
        initializeOpenGLFunctions();
        glClearColor(0, 0, 0, 1);
    }

    void paintGL() override {
        if (!d_matrix) return;
        std::vector<unsigned char> h_matrix(WIDTH * HEIGHT);
        cudaMemcpy(h_matrix.data(), d_matrix, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
        glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_UNSIGNED_BYTE, h_matrix.data());
    }

    void closeEvent(QCloseEvent* event) override {
        if(d_matrix) {
            cudaIpcCloseMemHandle(d_matrix);
            d_matrix = nullptr;
        }
        QOpenGLWidget::closeEvent(event);
        QCoreApplication::quit();
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Ricevi handle via UDP
    QUdpSocket udpSocket;
    udpSocket.bind(9999, QUdpSocket::ShareAddress);

    std::cout << "Waiting for UDP datagram...\n";

    if (!udpSocket.waitForReadyRead(5000)) {
        std::cerr << "No datagram received\n";
        return -1;
    }

    while (udpSocket.hasPendingDatagrams()) {
        QByteArray datagram;
        datagram.resize(udpSocket.pendingDatagramSize());
        udpSocket.readDatagram(datagram.data(), datagram.size());

        if (datagram.size() == sizeof(cudaIpcMemHandle_t)) {
            cudaIpcMemHandle_t memHandle;
            memcpy(&memHandle, datagram.data(), sizeof(memHandle));

            // Accedi alla memoria condivisa
            unsigned char* d_matrix;
            cudaIpcOpenMemHandle(reinterpret_cast<void**>(&d_matrix), memHandle, cudaIpcMemLazyEnablePeerAccess);

            Renderer w;
            w.setCudaPointer(d_matrix);
            w.resize(WIDTH, HEIGHT);
            w.show();

            int ret = app.exec();
            std::cerr << "Receiver has closed the window\n";

            QUdpSocket ackSocket;
            QByteArray ack("ACK");
            ackSocket.writeDatagram(ack, QHostAddress("127.0.0.1"), 9998);

            return ret;
        } else {
            std::cerr << "Received unexpected datagram size\n";
        }
    }

    return 0;
}

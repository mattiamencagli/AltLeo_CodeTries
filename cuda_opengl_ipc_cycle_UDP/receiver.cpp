#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QUdpSocket>
#include <QCloseEvent>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define WIDTH 1024
#define HEIGHT 1024

class Renderer : public QOpenGLWidget, protected QOpenGLFunctions {
public:
    void setCudaPointer(unsigned char* ptr) { d_matrix = ptr; }
    void triggerUpdate() { update(); }

protected:
    void initializeGL() override {
        initializeOpenGLFunctions();
        glClearColor(0, 0, 0, 1);
        // Stampa GPU info
        std::cout << "OpenGL Context Info: " << glGetString(GL_VERSION) << ", " << glGetString(GL_VENDOR) 
                  << ", " << glGetString(GL_RENDERER) << std::endl;
    }

    void paintGL() override {
        if (!d_matrix) return;
        std::vector<unsigned char> h_matrix(WIDTH * HEIGHT);
        cudaMemcpy(h_matrix.data(), d_matrix, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
        glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_UNSIGNED_BYTE, h_matrix.data());
    }

    void closeEvent(QCloseEvent* event) override {
        // Invia STOP al sender
        QUdpSocket ackSocket;
        ackSocket.writeDatagram("STOP", QHostAddress("127.0.0.1"), 9998);

        if (d_matrix) {
            cudaIpcCloseMemHandle(d_matrix);
            d_matrix = nullptr;
        }

        QOpenGLWidget::closeEvent(event);
        QCoreApplication::quit();
    }

private:
    unsigned char* d_matrix = nullptr;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QUdpSocket udpSocket;
    udpSocket.bind(9999, QUdpSocket::ShareAddress);

    Renderer renderer;

    std::cout << "Waiting for IPC handle...\n";

    cudaIpcMemHandle_t memHandle;
    bool handleReceived = false;

    // Riceve handle e crea memoria
    while (!handleReceived && udpSocket.waitForReadyRead(5000)) {
        QByteArray datagram;
        datagram.resize(udpSocket.pendingDatagramSize());
        udpSocket.readDatagram(datagram.data(), datagram.size());

        if (datagram.size() == sizeof(cudaIpcMemHandle_t)) {
            memcpy(&memHandle, datagram.data(), sizeof(memHandle));
            unsigned char* d_matrix;
            cudaIpcOpenMemHandle(reinterpret_cast<void**>(&d_matrix), memHandle, cudaIpcMemLazyEnablePeerAccess);
            renderer.setCudaPointer(d_matrix);
            handleReceived = true;
            renderer.resize(WIDTH, HEIGHT);
            renderer.show();
            break;
        }
    }

    if (!handleReceived) {
        std::cerr << "Failed to receive IPC handle.\n";
        return -1;
    }

    // Ricevi segnali FRAME_READY per aggiornare
    QObject::connect(&udpSocket, &QUdpSocket::readyRead, [&]() {
        while (udpSocket.hasPendingDatagrams()) {
            QByteArray datagram;
            datagram.resize(udpSocket.pendingDatagramSize());
            udpSocket.readDatagram(datagram.data(), datagram.size());

            if (datagram == "FRAME_READY") {
                renderer.triggerUpdate();
            }
        }
    });

    return app.exec();
}

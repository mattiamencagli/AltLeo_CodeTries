// opengl_receiver.cpp
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
    Q_OBJECT

public:
    void setCudaPointer(unsigned char* ptr) { d_matrix = ptr; }
    void triggerUpdate() { update(); }

protected:
    void initializeGL() override {
        initializeOpenGLFunctions();

        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    void paintGL() override {

        if (!d_matrix) return;

        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT, nullptr, GL_DYNAMIC_DRAW);

        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, WIDTH * HEIGHT, d_matrix);
        glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    void closeEvent(QCloseEvent* event) override {
        QUdpSocket ackSocket;
        ackSocket.writeDatagram("STOP", QHostAddress("127.0.0.1"), 9998);
        qDebug() << "Sending STOP to sender";
        if (d_matrix) {
            cudaIpcCloseMemHandle(d_matrix);
            d_matrix = nullptr;
        }
        QOpenGLWidget::closeEvent(event);
        QCoreApplication::quit();
    }

private:
    unsigned char* d_matrix = nullptr;
    GLuint pbo = 0;
};

#include ".receiver.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QUdpSocket udpSocket;
    if (!udpSocket.bind(9999, QUdpSocket::ShareAddress)) {
        qWarning() << "Receiver: bind failed!";
    }

    Renderer renderer;

    cudaIpcMemHandle_t memHandle;
    bool handleReceived = false;

    while (!handleReceived && udpSocket.waitForReadyRead(5000)) {
        QByteArray datagram;
        datagram.resize(udpSocket.pendingDatagramSize());
        udpSocket.readDatagram(datagram.data(), datagram.size());
        memcpy(&memHandle, datagram.data(), sizeof(memHandle));
        unsigned char* d_matrix;
        cudaIpcOpenMemHandle(reinterpret_cast<void**>(&d_matrix), memHandle, cudaIpcMemLazyEnablePeerAccess);
        renderer.setCudaPointer(d_matrix);
        renderer.resize(WIDTH, HEIGHT);
        renderer.show();
        handleReceived = true;
        break;
    }

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

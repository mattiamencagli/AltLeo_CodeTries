#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QUdpSocket>
#include <QCloseEvent>

#include "global_include.h"

class Renderer : public QOpenGLWidget, protected QOpenGLFunctions {
public:
    void setCudaPointer(unsigned char* ptr) {d_matrix = ptr;}

    void triggerUpdate() { 
        update();
        // QCoreApplication::processEvents();  // update a volte può rimanere appesto, così Forzo l'elaborazione immediata degli eventi
        // repaint();  // Forza immediatamente paintGL()
     }

protected:
    void initializeGL() override {
        initializeOpenGLFunctions();
        glClearColor(0, 0, 0, 1);

        // Genera e configura il PBO
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        std::cout << "OpenGL Context Info: " << glGetString(GL_VERSION) << ", " << glGetString(GL_VENDOR) 
                  << ", " << glGetString(GL_RENDERER) << std::endl;

        // Registra il buffer con CUDA
        CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsRegisterFlagsReadOnly));
    }

    void paintGL() override {
        if (!d_matrix) return;

        // Mappa risorsa PBO
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cudaPboResource, 0));
        unsigned char* pboPtr;
        size_t size_rcv;
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&pboPtr, &size_rcv, cudaPboResource));
        if (size != size_rcv){
            std::cerr << "Different sizes.\n";
            exit;
        }

        // Copia da handle IPC al buffer OpenGL
        CUDA_SAFE_CALL(cudaMemcpy(pboPtr, d_matrix, size, cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cudaPboResource, 0));

        #ifdef DEBUG
            unsigned char* dbg_val_receiver = new unsigned char[10];
            unsigned char* host_buf = new unsigned char[10];
            CUDA_SAFE_CALL(cudaMemcpy(dbg_val_receiver, d_matrix, 10, cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(host_buf, pboPtr, 10, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 3; ++i) {
                std::cout << "RECEIVER - pboPtr[" << i << "] = " << (int)host_buf[i] << "; d_matrix[" << i << "] = " 
                        << (int)dbg_val_receiver[i] << std::endl;
            }
            delete[] host_buf;
            delete[] dbg_val_receiver;
        #endif

        // Rendering da PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    }

    void closeEvent(QCloseEvent* event) override {
        QUdpSocket ackSocket;
        ackSocket.writeDatagram("STOP", QHostAddress("127.0.0.1"), 9998);

        if (d_matrix) {
            CUDA_SAFE_CALL(cudaIpcCloseMemHandle(d_matrix));
            d_matrix = nullptr;
        }
        if (cudaPboResource) {
            CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cudaPboResource));
        }
        if (pbo) {
            glDeleteBuffers(1, &pbo);
        }

        QOpenGLWidget::closeEvent(event);
        QCoreApplication::quit();
    }

private:
    const size_t size = WIDTH * HEIGHT;
    unsigned char* d_matrix = nullptr;
    GLuint pbo = 0;
    cudaGraphicsResource* cudaPboResource = nullptr;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QUdpSocket udpSocket;
    udpSocket.bind(9999, QUdpSocket::ShareAddress);

    Renderer renderer;
    std::cout << "Waiting for IPC handle...\n";

    cudaIpcMemHandle_t memHandle;
    bool handleReceived = false;

    while (!handleReceived && udpSocket.waitForReadyRead(5000)) {
        QByteArray datagram;
        datagram.resize(udpSocket.pendingDatagramSize());
        udpSocket.readDatagram(datagram.data(), datagram.size());

        if (datagram.size() == sizeof(cudaIpcMemHandle_t)) {
            memcpy(&memHandle, datagram.data(), sizeof(memHandle));
            unsigned char* d_matrix;
            CUDA_SAFE_CALL(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&d_matrix), memHandle, cudaIpcMemLazyEnablePeerAccess));
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

// opengl_reader.cpp

#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QLocalServer>
#include <QLocalSocket>
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
        // Qui si dovrebbe copiare da GPU a CPU o usare Pixel Buffer Object (PBO) per evitare copia
        std::vector<unsigned char> h_matrix(WIDTH * HEIGHT);
        cudaMemcpy(h_matrix.data(), d_matrix, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

        glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_UNSIGNED_BYTE, h_matrix.data());
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Socket per ricevere l'handle
    QLocalServer server;
    server.removeServer("cuda_ipc_server");
    server.listen("cuda_ipc_server");

    QLocalSocket *socket = server.nextPendingConnection();
    if (!server.waitForNewConnection(5000)) {
        std::cerr << "No connection received\n";
        return -1;
    }

    socket = server.nextPendingConnection();
    if (!socket->waitForReadyRead(1000)) {
        std::cerr << "Read failed\n";
        return -1;
    }

    cudaIpcMemHandle_t memHandle;
    socket->read(reinterpret_cast<char*>(&memHandle), sizeof(memHandle));

    // Accedi alla memoria condivisa
    unsigned char* d_matrix;
    cudaIpcOpenMemHandle(reinterpret_cast<void**>(&d_matrix), memHandle, cudaIpcMemLazyEnablePeerAccess);

    Renderer w;
    w.setCudaPointer(d_matrix);
    w.resize(WIDTH, HEIGHT);
    w.show();

    int ret = app.exec();

    // Libera memoria condivisa
    cudaIpcCloseMemHandle(d_matrix);

    return ret;
}

// cuda_sender.cpp
#include <QGuiApplication>
#include <QTimer>
#include <QUdpSocket>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QSurfaceFormat>
#include <QOpenGLFunctions>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <GL/gl.h>

#define WIDTH 1024
#define HEIGHT 1024

// Prototipo della funzione nel .cu
extern "C" void launch_fill_matrix(unsigned char* d_ptr, int width, int height, int frame, int speed);

class CudaSender : public QObject {
    Q_OBJECT

public:
    CudaSender() {

        if (!ackSocket.bind(9998, QUdpSocket::ShareAddress)) {
            qWarning() << "Sender: bind failed!";
        }

        format.setVersion(3, 3);
        format.setProfile(QSurfaceFormat::CompatibilityProfile);
        QSurfaceFormat::setDefaultFormat(format);

        context.setFormat(format);
        context.create();
        if (!context.isValid()) {
            qFatal("OpenGL context creation failed!");
        }

        surface.setFormat(format);
        surface.create();
        if (!surface.isValid()) {
            qFatal("Offscreen surface creation failed!");
        }

        if (!context.makeCurrent(&surface)) {
            qFatal("Failed to make OpenGL context current!");
        }


        gl = context.functions();
        gl->glGenBuffers(1, &pbo);
        gl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

        // DEBUGAQ
        GLint sizeGL = 0;
        gl->glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &sizeGL);
        qDebug() << "GL buffer size allocated:" << sizeGL;
        const GLubyte* renderer = gl->glGetString(GL_RENDERER);
        const GLubyte* vendor = gl->glGetString(GL_VENDOR);
        const GLubyte* version = gl->glGetString(GL_VERSION);
        qDebug() << "GL_RENDERER:" << reinterpret_cast<const char*>(renderer);
        qDebug() << "GL_VENDOR:" << reinterpret_cast<const char*>(vendor);
        qDebug() << "GL_VERSION:" << reinterpret_cast<const char*>(version);

        gl->glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT, nullptr, GL_DYNAMIC_DRAW);
        gl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);
        if (err != cudaSuccess) {
            qFatal("Failed to register GL buffer with CUDA: %s", cudaGetErrorString(err));
        }
        cudaMalloc((void **)&d_working, WIDTH * HEIGHT);

        // Ottieni handle da inviare
        cudaGraphicsMapResources(1, &cudaPboResource, 0);
        void* dev_ptr;
        size_t size;
        cudaGraphicsResourceGetMappedPointer(&dev_ptr, &size, cudaPboResource);

        cudaIpcMemHandle_t handle;
        cudaIpcGetMemHandle(&handle, dev_ptr);
        cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

        udpSocket.writeDatagram(reinterpret_cast<char*>(&handle), sizeof(handle), QHostAddress("127.0.0.1"), 9999);

        connect(&ackSocket, &QUdpSocket::readyRead, this, &CudaSender::handleStop);
        connect(&timer, &QTimer::timeout, this, &CudaSender::loop);

        timer.start(33);
    }

    ~CudaSender() {
        timer.stop();
        gl->glDeleteBuffers(1, &pbo);
        cudaFree(d_working);
        cudaGraphicsUnregisterResource(cudaPboResource);
    }

private slots:
    void loop() {
        static int frame = 0;
        frame++;

        launch_fill_matrix(d_working, WIDTH, HEIGHT, frame, 4);

        cudaGraphicsMapResources(1, &cudaPboResource, 0);
        void* dev_ptr;
        size_t size;
        cudaGraphicsResourceGetMappedPointer(&dev_ptr, &size, cudaPboResource);
        cudaMemcpy(dev_ptr, d_working, WIDTH * HEIGHT, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

    qDebug() << "Frame copied to PBO, ptr:" << dev_ptr;
        udpSocket.writeDatagram("FRAME_READY", QHostAddress("127.0.0.1"), 9999);
    }

    void handleStop() {
        while (ackSocket.hasPendingDatagrams()) {
            QByteArray msg;
            msg.resize(ackSocket.pendingDatagramSize());
            ackSocket.readDatagram(msg.data(), msg.size());
            qDebug() << "Sender received STOP. Exiting...";
            if (msg == "STOP") {
                QCoreApplication::quit();
            }
        }
    }

private:
    QSurfaceFormat format;
    QOpenGLContext context;
    QOffscreenSurface surface;
    QOpenGLFunctions* gl;
    QTimer timer;
    QUdpSocket udpSocket, ackSocket;

    GLuint pbo = 0;
    unsigned char* d_working = nullptr;
    cudaGraphicsResource* cudaPboResource = nullptr;
};

#include ".sender.moc"

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);
    CudaSender s;
    return app.exec();
}

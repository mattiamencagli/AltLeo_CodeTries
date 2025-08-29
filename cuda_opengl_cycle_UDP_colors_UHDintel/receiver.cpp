// Receiver.cpp
#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QUdpSocket>
#include <QCloseEvent>
#include <QSharedMemory>
#include <QTimer>

#include "global_include.h"

// --- Stessa chiave del sender ---
static const char* SHM_KEY = "CudaFrameSharedMemory_v1";

class Renderer : public QOpenGLWidget, protected QOpenGLFunctions {
public:
    Renderer(QWidget* parent=nullptr) : QOpenGLWidget(parent), shm(QString::fromLatin1(SHM_KEY)) {}

    // chiamato dal main quando arriva "FRAME_READY"
    void triggerUpdate() { update(); }

protected:
    void initializeGL() override {
        initializeOpenGLFunctions();
        glClearColor(0, 0, 0, 1);

        // Texture per visualizzare i dati
        glGenTextures(1, &texId);
        glBindTexture(GL_TEXTURE_2D, texId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        GL_SAFE_CALL(glGetError());

        // Stampa info GPU (ora sarà Intel)
        std::cout << "OpenGL Context Info: " << glGetString(GL_VERSION) << ", "
                  << glGetString(GL_VENDOR) << ", " << glGetString(GL_RENDERER) << std::endl;

        // Prova ad attaccarsi alla shared memory (può già esistere se il sender è avviato)
        attachSharedMemory();
    }

    void paintGL() override {
        glClear(GL_COLOR_BUFFER_BIT);

        if (texId == 0) return;

        // Se non siamo attaccati, non possiamo leggere frame
        if (!shm.isAttached()) {
            // Ritenta l'attach (ad es. se il sender è partito dopo)
            attachSharedMemory();
            return;
        }

        // Carica i dati dalla shared memory nella texture
        if (!shm.lock()) {
            qWarning("QSharedMemory lock() failed: %s", shm.errorString().toLatin1().constData());
            return;
        }
        const void* src = shm.constData();
        if (!src) {
            qWarning("QSharedMemory constData() null");
            shm.unlock();
            return;
        }

        glBindTexture(GL_TEXTURE_2D, texId);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        // Nota: passiamo direttamente il puntatore ai byte della shared memory
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, src);
        glBindTexture(GL_TEXTURE_2D, 0);

        shm.unlock();

        // draw full-screen textured quad (compatibility)
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texId);

        glBegin(GL_QUADS);
        glTexCoord2f(0.f, 0.f); glVertex2f(-1.f, -1.f);
        glTexCoord2f(1.f, 0.f); glVertex2f(+1.f, -1.f);
        glTexCoord2f(1.f, 1.f); glVertex2f(+1.f, +1.f);
        glTexCoord2f(0.f, 1.f); glVertex2f(-1.f, +1.f);
        glEnd();

        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);

        GL_SAFE_CALL(glGetError());
    }

    void resizeGL(int w, int h) override {
        glViewport(0, 0, w, h);
    }

    void closeEvent(QCloseEvent* event) override {
        QUdpSocket ackSocket;
        ackSocket.writeDatagram("STOP", QHostAddress("127.0.0.1"), 9998);

        if (texId) {
            glDeleteTextures(1, &texId);
            texId = 0;
        }
        if (shm.isAttached()) shm.detach();

        QOpenGLWidget::closeEvent(event);
        QCoreApplication::quit();
    }

private:
    void attachSharedMemory() {
        if (shm.isAttached()) return;
        if (!shm.attach()) {
            // Non è ancora creato dal sender — va bene, ritenteremo quando arrivano eventi
            // oppure puoi attivare un timer per riprovare periodicamente.
        } else {
            std::cout << "Attached to shared memory: " << SHM_KEY << " ("
                      << (WIDTH*HEIGHT*CHANNELS) << " bytes)" << std::endl;
        }
    }

private:
    GLuint texId = 0;
    QSharedMemory shm;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QUdpSocket udpSocket;
    udpSocket.bind(9999, QUdpSocket::ShareAddress);

    Renderer renderer;
    renderer.resize(WIDTH, HEIGHT);
    renderer.show();

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

    // Fallback: se per qualche motivo non arriva la UDP, aggiorna comunque (debug)
    QTimer fallbackTimer;
    fallbackTimer.setInterval(16); // ~60 FPS
    QObject::connect(&fallbackTimer, &QTimer::timeout, &renderer, [&renderer]() { renderer.triggerUpdate(); });
    fallbackTimer.start();

    return app.exec();
}

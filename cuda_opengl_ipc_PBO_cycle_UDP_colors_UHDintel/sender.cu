// sender.cu
#include <QCoreApplication>
#include <QUdpSocket>
#include <QTimer>
#include <QHostAddress>
#include <QSharedMemory>

#include "global_include.h"

// --- chiave condivisa della shared memory ---
static const char* SHM_KEY = "CudaFrameSharedMemory_v1";

__device__ __constant__ unsigned char palette[256][3] = {
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3},
    {66,30,15},{25,7,26},{9,1,47},{4,4,73},{0,7,100},{12,44,138},{24,82,177},{57,125,209},
    {134,181,229},{211,236,248},{241,233,191},{248,201,95},{255,170,0},{204,128,0},{153,87,0},{106,52,3}
};

__global__ void fill_matrix_linear_rgba(unsigned char* data, int width, int height, int frame) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        data[idx + 0] = (x + frame) % 256;
        data[idx + 1] = (y + frame) % 256;
        data[idx + 2] = ((x + y) + frame) % 256;
        data[idx + 3] = 255;
    }
}

__global__ void fill_matrix_spiral_rgba(unsigned char* data, int width, int height, int frame) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        float cx = width * 0.5f, cy = height * 0.5f;
        float dx = x - cx, dy = y - cy;
        float r = sqrtf(dx*dx + dy*dy);
        float theta = atan2f(dy, dx);
        float t = frame * 0.2f;
        float log_spiral = sinf(10.0f * logf(r + 1e-6f) + theta - t);
        unsigned char val = (unsigned char)((log_spiral + 1.0f) * 0.5f * 255.0f);
        data[idx + 0] = val;
        data[idx + 1] = 255 - val;
        data[idx + 2] = (val + frame) % 256;
        data[idx + 3] = 255;
    }
}

__global__ void fill_matrix_mandelbrot_rgba(unsigned char* data, int width, int height, int frame, float cX, float cY, float baseScale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;
        double scale = baseScale * powf(0.99f, frame);
        int maxIter = 2000;
        float jx = cX + (x - width/2.0f) * scale;
        float jy = cY + (y - height/2.0f) * scale;
        float zx = 0, zy = 0;
        int iter = 0;
        while (zx*zx + zy*zy < 4.0f && iter < maxIter) {
            float tmp = zx*zx - zy*zy + jx;
            zy = 2.0f*zx*zy + jy;
            zx = tmp;
            iter++;
        }
        unsigned char colorIdx = (unsigned char)((iter * 255) / maxIter);
        data[idx+0] = palette[colorIdx][0];
        data[idx+1] = palette[colorIdx][1];
        data[idx+2] = palette[colorIdx][2];
        data[idx+3] = 255;
    }
}

class CudaSender : public QObject {
    Q_OBJECT
public:
    CudaSender() : shm(QString::fromLatin1(SHM_KEY)) {
        ackSocket.bind(9998);

        // GPU buffers
        CUDA_SAFE_CALL(cudaMalloc(&d_working, size));

        // Shared memory (host)
        if (shm.isAttached()) shm.detach();
        if (!shm.create((int)size)) {
            qFatal("QSharedMemory create() failed: %s", shm.errorString().toLatin1().constData());
        }

        connect(&timer, &QTimer::timeout, this, &CudaSender::sendFrame);
        timer.start(16); // 60 FPS

        connect(&ackSocket, &QUdpSocket::readyRead, this, &CudaSender::handleIncoming);
    }

    ~CudaSender() override {
        timer.stop();
        if (shm.isAttached()) shm.detach();
        if (d_working) cudaFree(d_working);
    }

public slots:
    void sendFrame() {
        frame++;

        dim3 block(16, 16);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
        //fill_matrix_linear_rgba<<<grid, block>>>(d_working, WIDTH, HEIGHT, frame);
        //fill_matrix_spiral_rgba<<<grid, block>>>(d_working, WIDTH, HEIGHT, frame);
        fill_matrix_mandelbrot_rgba<<<grid, block>>>(d_working, WIDTH, HEIGHT, frame,
                                                     -0.7436438870371587f, 0.13182590420531197f, 16.0f/WIDTH);
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        // Copia dal device → host (shared memory)
        if (!shm.lock()) {
            qWarning("QSharedMemory lock() failed: %s", shm.errorString().toLatin1().constData());
            return;
        }
        void* host_ptr = shm.data();
        if (!host_ptr) {
            qWarning("QSharedMemory data() returned null");
            shm.unlock();
            return;
        }
        CUDA_SAFE_CALL(cudaMemcpy(host_ptr, d_working, size, cudaMemcpyDeviceToHost));
        shm.unlock();

        // Notifica al receiver che il frame è pronto
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
    const size_t size = WIDTH * HEIGHT * CHANNELS;
    unsigned char *d_working = nullptr;

    QSharedMemory shm;
    QUdpSocket udpSocket, ackSocket;
    QTimer timer;
};

#include ".sender.moc"

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    CudaSender sender;
    return app.exec();
}

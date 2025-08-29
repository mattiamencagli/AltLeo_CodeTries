#include <QCoreApplication>
#include <QUdpSocket>
#include <QTimer>
#include <QHostAddress>

#include "global_include.h"

#define CX -0.7436438870371587f
#define CY +0.13182590420531197f

__global__ void fill_matrix_linear_rgba(unsigned char* data, int width, int height, int frame) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 4; // 4 canali
        data[idx + 0] = (x + frame) % 256; // R
        data[idx + 1] = (y + frame) % 256; // G
        data[idx + 2] = ((x + y) + frame) % 256; // B
        data[idx + 3] = 255; // A
    }
}

__global__ void fill_matrix_spiral_rgba(unsigned char* data, int width, int height, int frame) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = (y * width + x) * 4;

        float cx = width * 0.5f;
        float cy = height * 0.5f;
        float dx = x - cx;
        float dy = y - cy;
        float r = sqrtf(dx*dx + dy*dy);
        float theta = atan2f(dy, dx);
        float t = frame * 0.2f;
        // float lin_spiral = sinf(r * 0.05f + theta - t);
        float log_spiral = sinf(10.0f * logf(r + 1e-6f) + theta - t);
        unsigned char val = (unsigned char)((log_spiral + 1.0f) * 0.5f * 255.0f);
        data[idx + 0] = val;          // R
        data[idx + 1] = 255 - val;    // G
        data[idx + 2] = (val + frame) % 256; // B
        data[idx + 3] = 255;          // A
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
        while(zx*zx + zy*zy < 4.0f && iter < maxIter){
            float tmp = zx*zx - zy*zy + jx;
            zy = 2.0f*zx*zy + jy;
            zx = tmp;
            iter++;
        }
        // Normalizza iter in [0,255]
        unsigned char colorIdx = (unsigned char)((iter * 255) / maxIter);
        // Applica palette fissa
        data[idx+0] = palette_gist_ncar[colorIdx][0]; // R
        data[idx+1] = palette_gist_ncar[colorIdx][1]; // G
        data[idx+2] = palette_gist_ncar[colorIdx][2]; // B
        data[idx+3] = 255;                  // A
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
        timer.start(16); // Un fotogramma ogni 16 ms -> 60 FPS
        // timer.start(33); // Un fotogramma ogni 33 ms -> 30 FPS

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
        //fill_matrix_linear_rgba<<<grid, block>>>(d_working, WIDTH, HEIGHT, frame);
        //fill_matrix_spiral_rgba<<<grid, block>>>(d_working, WIDTH, HEIGHT, frame);
        fill_matrix_mandelbrot_rgba<<<grid, block>>>(d_working, WIDTH, HEIGHT, frame, CX, CY, 16.0f/WIDTH);
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        #ifdef DEBUG
            unsigned char* h_debug = new unsigned char[3];
            CUDA_SAFE_CALL(cudaMemcpy(h_debug, d_working, 3, cudaMemcpyDeviceToHost));
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
    const size_t size = WIDTH * HEIGHT * CHANNELS;
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

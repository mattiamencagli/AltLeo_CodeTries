#include <QCoreApplication>
#include <QUdpSocket>
#include <QTimer>
#include <QHostAddress>

#include "global_include.h"

__device__ __constant__ unsigned char palette[256][3] = {
    {0,0,128}, {0,7,118}, {0,14,109}, {0,21,99}, {0,29,90}, {0,36,80}, {0,43,71}, {0,51,62}, 
    {0,58,52}, {0,65,43}, {0,72,33}, {0,80,24}, {0,87,15}, {0,94,5}, {0,88,22}, {0,81,38}, 
    {0,74,55}, {0,67,72}, {0,61,88}, {0,54,105}, {0,47,121}, {0,40,138}, {0,33,155}, {0,27,171}, 
    {0,20,188}, {0,13,205}, {0,6,221}, {0,0,238}, {0,14,255}, {0,28,255}, {0,42,255}, {0,56,255}, 
    {0,70,255}, {0,84,255}, {0,98,255}, {0,112,255}, {0,127,255}, {0,141,255}, {0,155,255}, {0,169,255}, 
    {0,183,255}, {0,192,255}, {0,197,255}, {0,202,255}, {0,206,255}, {0,210,255}, {0,215,255}, {0,219,255}, 
    {0,224,255}, {0,228,255}, {0,232,255}, {0,237,255}, {0,241,254}, {0,246,248}, {0,250,241}, {0,254,235}, 
    {0,254,228}, {0,254,222}, {0,253,215}, {0,253,209}, {0,252,202}, {0,252,195}, {0,251,189}, {0,251,182}, 
    {0,250,176}, {0,250,169}, {0,250,163}, {0,250,156}, {0,250,146}, {0,250,135}, {0,250,125}, {0,250,114}, 
    {0,251,104}, {0,251,93}, {0,252,83}, {0,252,73}, {0,252,62}, {0,253,52}, {0,253,41}, {0,254,31}, 
    {6,254,20}, {12,254,10}, {19,251,0}, {25,247,0}, {31,243,0}, {38,239,0}, {44,236,0}, {50,232,0}, 
    {57,228,0}, {63,224,0}, {70,221,0}, {76,217,0}, {82,213,0}, {89,209,0}, {95,206,0}, {101,209,0}, 
    {103,212,0}, {105,215,0}, {107,219,0}, {109,222,0}, {111,225,0}, {113,228,0}, {115,232,0}, {117,235,0}, 
    {119,238,0}, {121,241,0}, {123,245,0}, {125,248,3}, {127,251,7}, {132,254,11}, {136,255,15}, {141,255,19}, 
    {145,255,23}, {150,255,27}, {154,255,31}, {159,255,35}, {164,255,39}, {168,255,43}, {173,255,47}, {177,255,51}, 
    {182,255,55}, {186,255,59}, {191,255,55}, {195,255,51}, {200,255,47}, {204,255,43}, {209,255,39}, {214,255,35}, 
    {218,255,31}, {223,255,27}, {227,255,23}, {232,255,19}, {236,255,15}, {241,255,11}, {245,252,7}, {250,250,3}, 
    {255,247,0}, {255,245,0}, {255,242,0}, {255,240,0}, {255,237,0}, {255,235,0}, {255,232,0}, {255,230,0}, 
    {255,227,0}, {255,225,0}, {255,222,0}, {255,220,0}, {255,218,0}, {255,215,1}, {255,213,2}, {255,210,3}, 
    {255,208,4}, {255,205,5}, {255,203,6}, {255,200,7}, {255,198,8}, {255,195,9}, {255,193,10}, {255,190,11}, 
    {255,188,12}, {255,185,13}, {255,177,13}, {255,169,12}, {255,161,11}, {255,153,10}, {255,145,9}, {255,136,8}, 
    {255,128,7}, {255,120,6}, {255,112,5}, {255,104,4}, {255,95,3}, {255,87,2}, {255,79,1}, {255,71,0}, 
    {255,66,0}, {255,61,0}, {255,57,0}, {255,52,0}, {255,47,0}, {255,42,0}, {255,38,0}, {255,33,0}, 
    {255,28,0}, {255,23,0}, {255,19,0}, {255,14,0}, {255,9,0}, {255,4,17}, {255,0,35}, {255,0,53}, 
    {255,0,70}, {255,0,88}, {255,0,106}, {255,0,123}, {255,0,141}, {255,0,159}, {255,0,177}, {255,0,194}, 
    {255,0,212}, {255,0,230}, {255,0,248}, {248,3,251}, {241,6,255}, {234,10,255}, {227,13,255}, {220,17,255}, 
    {213,20,255}, {206,24,255}, {199,27,255}, {193,30,255}, {186,34,255}, {179,37,255}, {172,41,255}, {165,44,254}, 
    {158,50,253}, {164,56,252}, {170,62,251}, {176,68,250}, {182,74,248}, {188,80,247}, {194,86,246}, {199,92,245}, 
    {205,97,244}, {211,103,242}, {217,109,241}, {223,115,240}, {229,121,239}, {235,127,238}, {236,132,238}, {236,136,239}, 
    {237,141,240}, {238,146,240}, {239,150,241}, {239,155,241}, {240,159,242}, {241,164,243}, {241,169,243}, {242,173,244}, 
    {243,178,244}, {244,183,245}, {244,187,246}, {245,192,246}, {246,197,247}, {246,201,247}, {247,206,248}, {248,210,249}, 
    {249,215,249}, {249,220,250}, {250,224,250}, {251,229,251}, {251,234,252}, {252,238,252}, {253,243,253}, {254,247,254}
};

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
        data[idx+0] = palette[colorIdx][0]; // R
        data[idx+1] = palette[colorIdx][1]; // G
        data[idx+2] = palette[colorIdx][2]; // B
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
        fill_matrix_mandelbrot_rgba<<<grid, block>>>(d_working, WIDTH, HEIGHT, frame, -0.7436438870371587f, 0.13182590420531197f, 16.0f/WIDTH);
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

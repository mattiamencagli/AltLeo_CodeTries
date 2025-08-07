# sender
nvcc -c cuda_kernels.cu -o .cuda_kernels.o
moc sender.cpp -o .sender.moc
g++ sender.cpp .cuda_kernels.o -o sender.x \
    -fPIC -std=c++17 -lGL -lcuda -lcudart \
    `pkg-config --cflags --libs Qt5Widgets Qt5Network Qt5Gui Qt5Core`

# receiver
moc receiver.cpp -o .receiver.moc
g++ receiver.cpp -o receiver.x \
    -lQt5Widgets -lGL -lGLU -lpthread -fPIC -lcuda -lcudart -std=c++17 -lGL \
    `pkg-config --cflags --libs Qt5Widgets Qt5Network Qt5Gui Qt5Core` 


# sender
moc sender.cu -o .sender.moc
nvcc sender.cu -o sender.x \
    `pkg-config --cflags --libs Qt5Widgets Qt5Network Qt5Gui Qt5Core` \
    -Xcompiler -fPIC -std=c++17

# receiver
g++ receiver.cpp -o receiver.x \
    -lQt5Widgets -lGL -lGLU -lpthread -fPIC -lcuda -lcudart -std=c++17 \
    `pkg-config --cflags --libs Qt5Widgets Qt5Network Qt5Gui Qt5Core` 


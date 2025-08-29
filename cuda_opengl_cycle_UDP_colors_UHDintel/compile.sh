#clean
rm -f .sender.moc receiver.x sender.x

# sender
moc sender.cu -o .sender.moc
nvcc sender.cu -o sender.x \
    -Xcompiler -fPIC -std=c++17 -lm \
    `pkg-config --cflags --libs Qt5Widgets Qt5Network Qt5Gui Qt5Core`

# receiver
g++ receiver.cpp -o receiver.x \
    -lQt5Widgets -lGL -lGLU -lpthread -fPIC -lcuda -lcudart -std=c++17 -lm \
    `pkg-config --cflags --libs Qt5Widgets Qt5Network Qt5Gui Qt5Core` 


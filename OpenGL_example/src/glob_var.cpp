#include "glob_var.hpp"

int CurrentWidth = 600;
int CurrentHeight = 600;
int WindowHandle = 0;
unsigned int FrameCount = 0;

GLuint ProjectionMatrixUniformLocation{};
GLuint ViewMatrixUniformLocation{};
GLuint ModelMatrixUniformLocation{};
GLuint BufferIds[3]{};
GLuint ShaderIds[3]{};

Matrix ProjectionMatrix{};
Matrix ViewMatrix{};
Matrix ModelMatrix{};

float CubeRotation = 0;
clock_t LastTime = 0;

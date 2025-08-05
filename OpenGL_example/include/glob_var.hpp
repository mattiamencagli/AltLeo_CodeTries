#ifndef GLOB_VAR_H
#define GLOB_VAR_H

#include "opengl_headers.hpp"
#include "utils.hpp"

#define WINDOW_TITLE_PREFIX "OpenGL project 0"

extern int CurrentWidth;
extern int CurrentHeight;
extern int WindowHandle;
extern unsigned int FrameCount;

extern GLuint ProjectionMatrixUniformLocation;
extern GLuint ViewMatrixUniformLocation;
extern GLuint ModelMatrixUniformLocation;
extern GLuint BufferIds[3];
extern GLuint ShaderIds[3];

extern float CubeRotation;
extern clock_t LastTime;

#endif
#ifndef UTILS_H
#define UTILS_H

#include "headers.hpp"
#include "opengl_headers.hpp"

static const double PI = 3.14159265358979323846;

typedef struct Vertex_
{
    float Position[4];
    float Color[4];
} Vertex;

typedef struct Matrix_
{
    float m[16];
} Matrix;

extern const Matrix IDENTITY_MATRIX;

extern Matrix ProjectionMatrix;
extern Matrix ViewMatrix;
extern Matrix ModelMatrix;

float Cotangent(const float &angle);
float DegreesToRadians(const float &degrees);
float RadiansToDegrees(const float &radians);

Matrix MultiplyMatrices(const Matrix *m1, const Matrix *m2);
void RotateAboutX(Matrix *m, const float &angle);
void RotateAboutY(Matrix *m, const float &angle);
void RotateAboutZ(Matrix *m, const float &angle);
void ScaleMatrix(Matrix *m, const float &x, const float &y, const float &z);
void TranslateMatrix(Matrix *m, const float &x, const float &y, const float &z);

Matrix CreateProjectionMatrix(const float &fovy, const float &aspect_ratio, const float &near_plane, const float &far_plane);

void ExitOnGLError(const char *error_message);
GLuint LoadShader(const char *filename, GLenum shader_type);

#endif
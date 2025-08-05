#include "window.hpp"

void ResizeFunction(int Width, int Height)
{
    CurrentWidth = Width;
    CurrentHeight = Height;
    // std::cout << " new Width and Height : " << Width << " x " << Height << std::endl;
    glViewport(0, 0, CurrentWidth, CurrentHeight);

    ProjectionMatrix = CreateProjectionMatrix(60, (float)CurrentWidth / CurrentHeight, 1.0f, 100.0f);

    glUseProgram(ShaderIds[0]);
    glUniformMatrix4fv(ProjectionMatrixUniformLocation, 1, GL_FALSE, ProjectionMatrix.m);
    glUseProgram(0);
}

void RenderFunction()
{
    ++FrameCount;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLuint queryId;
    GLint timeElapsed;
    glGenQueries(1, &queryId);
    glBeginQuery(GL_TIME_ELAPSED, queryId);

    DrawCube();

    glutSwapBuffers();
    
    glEndQuery(GL_TIME_ELAPSED);
    glGetQueryObjectiv(queryId, GL_QUERY_RESULT, &timeElapsed);
    printf("TIME per frame = %g ms\n", timeElapsed/1e6);
    
}

void TimerFunction(int Value)
{
    if (Value)
    {
        char *TempString = (char *)
            malloc(512 + strlen(WINDOW_TITLE_PREFIX));

        sprintf(TempString, "%s: %d Frames Per Second @ %d x %d", WINDOW_TITLE_PREFIX, FrameCount, CurrentWidth, CurrentHeight);

        glutSetWindowTitle(TempString);
        free(TempString);
    }

    FrameCount = 0;
    glutTimerFunc(1000, TimerFunction, 1);
    // every 1000 ms call TimerFunction with argument "1"
}

void IdleFunction()
{
    glutPostRedisplay();
}

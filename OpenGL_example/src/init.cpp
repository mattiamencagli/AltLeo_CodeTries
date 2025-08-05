#include "init.hpp"

void Initialize(int *argc, char **argv)
{

    InitWindow(argc, argv);

    std::cout << "INFO: OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    GLenum GlewInitResult = glewInit();

    if (GLEW_OK != GlewInitResult)
    {
        fprintf(stderr, "ERROR: %s\n", glewGetErrorString(GlewInitResult));
        exit(EXIT_FAILURE);
    }

    glGetError();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    ExitOnGLError("ERROR: Could not set OpenGL depth testing options");

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    ExitOnGLError("ERROR: Could not set OpenGL culling options");

    ModelMatrix = IDENTITY_MATRIX;
    ProjectionMatrix = IDENTITY_MATRIX;
    ViewMatrix = IDENTITY_MATRIX;
    TranslateMatrix(&ViewMatrix, 0, 0, -2);

    CreateCube();
}

void InitWindow(int *argc, char **argv)
{

    glutInit(argc, argv);

    glutInitContextVersion(4, 0);
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
    glutInitContextProfile(GLUT_CORE_PROFILE);

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    glutInitWindowSize(CurrentWidth, CurrentHeight);

    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    // GLUT_DEPTH enables the usage of the depth buffer, an important mechanism in 3D computer graphics. The depth buffer (also called Z-buffer) contains the floating point Z-depth information of each pixel rendered to the screen
    // GLUT_DOUBLE is a flag that enables the usage of double-buffering, which is a feature that reduces image flickering. With double buffering, all of the draw commands are executed on an off-screen buffer, which is sent to the screen when all of the drawing for a frame has been completed so that no incomplete images are displayed.
    // GLUT_RGBA is a flag that defines the way colours are composited by using individual Red, Green, Blue, and Alpha values.

    WindowHandle = glutCreateWindow(WINDOW_TITLE_PREFIX);
    // The WINDOW_TITLE_PREFIX pre-processor definition simply serves as a placeholder for our window title.

    if (WindowHandle < 1)
    {
        fprintf(stderr, "ERROR: Could not create a new rendering window.\n");
        exit(EXIT_FAILURE);
    }

    glutReshapeFunc(ResizeFunction);
    glutDisplayFunc(RenderFunction);
    glutIdleFunc(IdleFunction);
    glutTimerFunc(0, TimerFunction, 0); // call TimerFunction immediatelly (after 0 ms) with argument "0"

    glutCloseFunc(DestroyCube);
};
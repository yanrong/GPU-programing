#include <iostream>
#include <string.h>

#include <cmath>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "ogldev_math_3d.h"

static void renderSceneCB();
static void initializeGLUTCallbacks();
static void createVertexBuffer();
static void addShaders(GLuint program, const char* souce, GLenum type);
static void compileShader(const char* vertex, const char *fragment);

GLuint VBO, worldLocation;

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(1024, 768);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Tutorial 09");

    initializeGLUTCallbacks();

    //musty be done after glut is initialized
    GLenum res = glewInit();
    if (res!= GLEW_OK) {
        std::cerr << "Error :" << glewGetErrorString(res) << std::endl;
        return -1;
    }

    std::cout <<"GL version :" << glGetString(GL_VERSION);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    createVertexBuffer();
    compileShader("shader.vs", "shader.fs");

    glutMainLoop();
    return 0;
}

static void renderSceneCB()
{
    glClear(GL_COLOR_BUFFER_BIT);

    static float scale = 0.0f;
    Matrix4f world;
    scale += 0.001f;

    world.m[0][0] = sinf(scale); world.m[0][1] = 0.0f;        world.m[0][2] = 0.0f;         world.m[0][3] = 0.0f;
    world.m[1][0] = 0.0f;        world.m[1][1] = sinf(scale); world.m[1][2] = 0.0f;         world.m[1][3] = 0.0f;
    world.m[2][0] = 0.0f;        world.m[2][1] = 0.0f;        world.m[2][2] = sinf(scale);  world.m[2][3] = 0.0f;
    world.m[3][0] = 0.0f;        world.m[3][1] = 0.0f;        world.m[3][2] = 0.0f;         world.m[3][3] = 1.0f;

    glUniformMatrix4fv(worldLocation, 1, GL_TRUE, &world.m[0][0]);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glDisableVertexAttribArray(0);
    glutSwapBuffers();
}

static void initializeGLUTCallbacks()
{
    glutDisplayFunc(renderSceneCB);
    glutIdleFunc(renderSceneCB);
}

static void createVertexBuffer()
{
    GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         0.0f,  1.0f, 0.0f
    };

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

static void addShaders(GLuint program, const char* source, GLenum type)
{
    GLint length, success;
    GLuint shader = glCreateShader(type);

    if (shader == 0) {
        std::cerr << "Error creating shader type:" << type << std::endl;
        return ;
    }

    length = strlen(source);
    glShaderSource(shader, 1, &source, &length);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[1024];
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cerr << "Error compile the shader source, Type " << type << " " << infoLog << std::endl;
        return ;
    }

    glAttachShader(program, shader);
}

static void compileShader(const char* vertex, const char *fragment)
{
    std::string vert, frag;
    GLint success;
    GLchar errorLog[1024] = {0};
    GLuint program = glCreateProgram();

    if (program == 0) {
        std::cerr << "Error creating shader program" << std::endl;
        return ;
    }

    if (!ReadFile(vertex, vert)) {
        std::cerr << "read vertex shader source file error" << std::endl;
        return;
    }

    if (!ReadFile(fragment, frag)) {
        std::cerr << "read vertex shader source file error" << std::endl;
        return;
    }

    addShaders(program, vert.c_str(), GL_VERTEX_SHADER);
    addShaders(program, frag.c_str(), GL_FRAGMENT_SHADER);

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == 0) {
        glGetProgramInfoLog(program, sizeof(errorLog), NULL, errorLog);
        std::cerr << "Error link shader program: " << errorLog << std::endl;
        return ;
    }

    glValidateProgram(program);
    glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, sizeof(errorLog), NULL, errorLog);
        std::cerr << "Invalide shader program : " << errorLog << std::endl;
        return ;
    }

    glUseProgram(program);
    worldLocation = glGetUniformLocation(program, "gWorld");
    assert(worldLocation != 0xffffffff);
}
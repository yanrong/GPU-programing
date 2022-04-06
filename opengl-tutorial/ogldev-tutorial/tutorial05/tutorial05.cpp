/**
 * @file tutorial05.cpp
 * @author
 * @brief
 * @version 0.1
 * @date 2022-04-06
 *
 * @copyright Copyright (c) 2022
 * The code text origin from Etay Meiri' Demo
 * Release under the GNU General Public License
 */

#include <iostream>
#include <cstring>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "ogldev_math_3d.h"

static GLuint VBO, scaleLocation;

static void renderSceneCB()
{
    static float scale = 0.0f;
    static float delta = 0.001f;

    glClear(GL_COLOR_BUFFER_BIT);

    scale += delta;
    if ((scale >= 1.0f) || (scale <= -1.0f)) {
        delta *= -1.0f;
    }

    glUniform1f(scaleLocation, scale);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisableVertexAttribArray(0);

    glutPostRedisplay();
    glutSwapBuffers();
}

static void createVertexBuffer()
{
    Vector3f vertices[3];
    vertices[0] = Vector3f(-1.0f, -1.0f, 0.0f); //bottom left
    vertices[1] = Vector3f(1.0f, -1.0f, 0.0f); //bottom right
    vertices[2] = Vector3f(0.0f, 1.0f, 0.0f); //top

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
    scaleLocation = glGetUniformLocation(program, "gScale");
    if (scaleLocation == -1) {
        std::cerr << "Error getting uniform location" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    int width = 1920, height = 1080;
    int x = 200, y = 100;
    GLclampf Red = 0.0f, Green = 0.0f, Blue = 0.0f, Alpha = 0.0f;

    glutInit(&argc, argv); //init glut
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); //set display mode
    glutInitWindowSize(width, height); // set window size
    glutInitWindowPosition(x, y); //set (left upper)window position

    int win = glutCreateWindow("Tutorial 05"); // creat window with title
    std::cout << "window id: " << win << std::endl;

    //must be done after glut is initialzed!
    GLenum res = glewInit();
    if (res != GLEW_OK) {
        std::cerr << "Erros :" << glewGetErrorString(res) << std::endl;
        return -1;
    }

    glClearColor(Red, Green, Blue, Alpha); // clear and set color

    createVertexBuffer();
    compileShader("shader.vs", "shader.fs");

    glutDisplayFunc(renderSceneCB); //actually display(draw) function

    glutMainLoop();

    return 0;
}

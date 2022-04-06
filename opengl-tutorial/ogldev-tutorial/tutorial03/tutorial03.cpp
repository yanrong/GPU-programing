/**
 * @file tutorial04.cpp
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
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "ogldev_math_3d.h"

static GLuint VBO;

static void renderSceneCB()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisableVertexAttribArray(0);
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

int main(int argc, char *argv[])
{
    int width = 1920, height = 1080;
    int x = 200, y = 100;
    GLclampf Red = 0.0f, Green = 0.0f, Blue = 0.0f, Alpha = 0.0f;

    glutInit(&argc, argv); //init glut
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); //set display mode
    glutInitWindowSize(width, height); // set window size
    glutInitWindowPosition(x, y); //set (left upper)window position

    int win = glutCreateWindow("Tutorial 03"); // creat window with title
    std::cout << "window id: " << win << std::endl;

    //must be done after glut is initialzed!
    GLenum res = glewInit();
    if (res != GLEW_OK) {
        std::cerr << "Erros :" << glewGetErrorString(res) << std::endl;
        return -1;
    }

    glClearColor(Red, Green, Blue, Alpha); // clear and set color

    createVertexBuffer();

    glutDisplayFunc(renderSceneCB); //actually display(draw) function

    glutMainLoop();

    return 0;
}

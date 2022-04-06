/**
 * @file tutorial01.cpp
 * @author
 * @brief
 * @version 0.1
 * @date 2022-04-06
 *
 * @copyright Copyright (c) 2022
 * The code text origin from Etay Meiri
 * Release under the GNU General Public License
 */

#include <GL/freeglut.h>
#include <iostream>

static void renderSceneCB()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
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
    int win = glutCreateWindow("Tutorial 01"); // creat window with title
    std::cout << "window id: " << win << std::endl;
    glClearColor(Red, Green, Blue, Alpha); // clear and set color
    glutDisplayFunc(renderSceneCB); //actually display(draw) function

    glutMainLoop();

    return 0;
}

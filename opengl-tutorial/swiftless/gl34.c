#include <GL/gl.h>
#include <GL/glut.h>
#include <stdio.h>

int w1;
int h1;

void orthogonalStart(void)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w1, 0, h1);
    glScalef(1, -1, 1);
    glTranslatef(0, -h1, 0);
    glMatrixMode(GL_MODELVIEW);
}

void orthogonalEnd(void)
{
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void display(void)
{
    glClearColor(1.0,0.0,0.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    orthogonalStart();

    glBegin(GL_QUADS);
    glVertex2f(125, 125);
    glVertex2f(125, 375);
    glVertex2f(375, 375);
    glVertex2f(375, 125);
    glEnd();

    orthogonalEnd();

    glutSwapBuffers();
}

void reshape(int w, int h)
{
    glViewport(0, 0,(GLsizei)w,(GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60,(GLfloat)w /(GLfloat)h, 0.1, 1000.0);
    w1 = w;
    h1 = h;
    glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("A Basic Window");
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutReshapeFunc(reshape);
    glutMainLoop();
    return 0;
}
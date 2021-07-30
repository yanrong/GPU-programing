#include <GL/gl.h>
#include <GL/glut.h>
#include <math.h>

GLfloat d;
GLfloat p1x;
GLfloat p1y;
GLfloat p1z;
const GLfloat p1raduis = 1;
const GLfloat p2raduis = 0;
GLfloat p2x;
GLfloat p2y;
GLfloat p2z;

void collision(void)
{
	d = sqrt((p1x- p2x)*(p1x - p2x) + (p1y - p2y)*(p1y - p2y) + (p1z - p2z)*(p1z - p2z));
}

void pointz(void)
{
	glPushMatrix();
	if (d <= p2raduis + p1raduis)
	{
		glColor3f(1, 0, 0);
	}
	else
	{
		glColor3f(0, 0, 1);
	}

	glBegin(GL_POINTS);
	glVertex3f(p1x, p1y, p1z);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glColor3f(0, 1, 0);
	glBegin(GL_POINTS);
	glVertex3f(p2x, p2y, p2z);
	glEnd();
	glPopMatrix();
}

void display(void) 
{
    glClearColor(0.0,0.0,0.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();  
    gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    glPointSize(15);
    collision();
    pointz();
    glutSwapBuffers();
}

void reshape(int w, int h)
{
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, (GLfloat)w / (GLfloat)h, 1.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

void keyboard(unsigned char key, int x, int y) {
	switch(key)
	{
	case 'q':
		p1z = p1z - 0.1;
		break;
	case 'z':
		p1z = p1z + 0.1;
		break;
	case 'w':
		p1y = p1y + 0.1;
		break;
	case 's':
		p1y = p1y - 0.1;
		break;
	case 'a':
		p1x = p1x - 0.1;
		break;
	case 'd':
		p1x = p1x + 0.1;
		break;
	case 'i':
		p2y = p2y + 0.1;
		break;
	case'k':
		p2y = p2y - 0.1;
		break;
	case 'j':
		p2x = p2x - 0.1;
		break;
	case 'l':
		p2x = p2x + 0.1;
		break;
	case 27: //27 is the ascii code for the ESC key
	    exit(0); //end the program
	    break;
	default:
		break;
	}
}

int main (int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE); //set up the double buffering
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("A basic OpenGL Window");
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);//the call for the keyboard function.
    glutMainLoop();
    return 0;
}
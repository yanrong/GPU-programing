#include <GL/gl.h>
#include <GL/glut.h>
#include <stdio.h>
#include <math.h>

#define PI 3.141592654f 
//angle of rotation
float xpos = 0, ypos = 0, zpos = 0, xrot = 0, yrot = 0, angle = 0.0;
//position of the cubes
float positionz[10];
float positionx[10];

//set the positions for the cube
void cubepositions(void)
{
	for(int i = 0; i < 10; i++)
	{
		positionz[i] = rand() % 5 + 5;
		positionx[i] = rand() % 5 + 5;
	}
}

//draw the cubes
void cube(void)
{
	for(int i = 0; i < 10; i++)
	{
		glPushMatrix();
		//translate the cube
		glTranslated(-positionx[i + 1] * 10, 0, -positionz[i + 1] * 10);
	 	glutSolidCube(2); //draw the cube
	 	glPopMatrix();
	}
}

void init(void)
{
	cubepositions();
}

void enable(void)
{
	glEnable(GL_DEPTH_TEST); //enable the depth testing
	glEnable(GL_LIGHTING); //enable the lighting
	glEnable(GL_LIGHT0); //enable LIGHT0, our diffuse light
	glShadeModel(GL_SMOOTH); //set the shader to smooth shader
}

void camera(void)
{
	glRotatef(xrot, 1.0, 0.0, 0.0); // rotate camera on the x-axis(left to right)
	glRotatef(yrot, 0.0, 1.0, 0.0); // rotate camera on the y-axis(up and down)
	glTranslated(-xpos, -ypos, -zpos); //translate the screen to the position of our camera
}

void display(void)
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear the color buffer and depth buffer
	glLoadIdentity();
	camera();
	enable();
	cube(); //call the cube drawing function
	glutSwapBuffers(); //swap the buffers
	angle++;
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h); //set the viewport to the current window specifications
	glMatrixMode(GL_PROJECTION); //set the matrix to projection
	glLoadIdentity();
	gluPerspective(60, (GLfloat)w / (GLfloat)h, 1.0, 1000.0); //set the perspective(angle of sight, width, height, depth)
	glMatrixMode(GL_MODELVIEW);//set the matrix back to model
}

void keyboard(unsigned char key, int x, int y)
{
	if(key == 'q')
	{
		xrot += 1;
		if(xrot > 360) xrot -= 360; 
	}
	if(key == 'z')
	{
		xrot -= 1;
		if(xrot < -360) xrot += 360; 
	}
	if(key == 'w')
	{
		float xrotrad, yrotrad;
		yrotrad = (yrot / 180 * PI);
		xrotrad = (xrot / 180 * PI);
		xpos += (float)sin(yrotrad);
		zpos -= (float)cos(yrotrad);
		ypos += (float)sin(xrotrad);
	}
	if(key == 's')
	{
		float xrotrad, yrotrad;
		yrotrad = (yrot / 180 * PI);
		xrotrad = (xrot / 180 * PI);
		xpos -= (float)sin(yrotrad);
		zpos += (float)cos(yrotrad);
		ypos += (float)sin(xrotrad);
	}
	if(key == 'd')
	{
		yrot += 1;
		if(yrot > 360) yrot -= 360; 
	}
	if(key == 'a')
	{
		yrot -= 1;
		if(yrot < -360) yrot += 360; 
	}

	if(key == 27)
	{
		exit(0); //quit the program
	}
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH); //set the display to double buffer,with depth
	glutInitWindowSize(500, 500); //set the window size
	glutInitWindowPosition(100, 100); //set the position of the window
	glutCreateWindow("A basic OpenGL Window");
    init(); //call the init function
    glutDisplayFunc(display); //use the display function to draw everything
    //update any variables in display,display can be changed to anyhing, as long as you move the variables to be updated, in this case, angle++;
    glutIdleFunc(display);
    glutReshapeFunc(reshape); //reshape the window accordingly

    glutKeyboardFunc(keyboard); //check the keyboard
    glutMainLoop(); //call the main loop
	return 0;
}

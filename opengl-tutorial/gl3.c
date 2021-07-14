#include <GL/glew.h>		// Include the GLEW header file
#include <GL/glut.h>		// Include the GLUT header file
#include <stdlib.h>

int keyStates[256] = {0};
int keySpecialStates[254] = {0};

void KeyOperations(void)
{
	if(keyStates['a'])
	{
		printf("key a press \n");
	}
}

void keySpecialOperations(void) 
{  
	if (keySpecialStates[GLUT_KEY_LEFT]) { // If the left arrow key has been pressed  
		// Perform left arrow key operations  
	}  
}

void display(void)
{
	KeyOperations();
	keySpecialOperations();
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);	// Clear the background of our window to red  
	glClear(GL_COLOR_BUFFER_BIT);	//Clear the colour buffer (more buffers later on)  
	glLoadIdentity();	// Load the Identity Matrix to reset our drawing locations  

	glFlush();		// Flush the OpenGL buffers to the window  
}

void reshape(int width, int height)
{
	/* Set our viewport to the size of our window  */
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	/* Switch to the projection matrix so that we can manipulate how our scene is viewed  */
	glMatrixMode(GL_PROJECTION);
	/*Reset the projection matrix to the identity matrix so that we don't get any artifacts (cleaning up)*/
	glLoadIdentity();
	/*Set the Field of view angle (in degrees), the aspect ratio of our window, and the new and far planes*/
	gluPerspective(60, (GLfloat)width / (GLfloat)height, 1.0, 100.0);
	/* Switch back to the model view matrix, so that we can start drawing shapes correctly */
	glMatrixMode(GL_MODELVIEW);
}

void keyPressd(unsigned char key, int x, int y)
{
	keyStates[key] = 1;
}

void keyUp(unsigned char key, int x, int y)
{
	keyStates[key] = 0;
}

void keySpecial(int key, int x, int y)
{
	keySpecialStates[GLUT_KEY_LEFT] = 1;
}

void keySpecialUp(int key, int x, int y)
{
	keySpecialStates[GLUT_KEY_LEFT] = 0;
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);	// Initialize GLUT  
	glutInitDisplayMode(GLUT_SINGLE);	// Set up a basic display buffer (only single buffered for now)  
	glutInitWindowSize(500, 500);	// Set the width and height of the window  
	glutInitWindowPosition(100, 100);	// Set the position of the window  
	glutCreateWindow("Your first OpenGL Window");	// Set the title for the window  

	glutDisplayFunc(display);	// Tell GLUT to use the method "display" for rendering  
	glutReshapeFunc(reshape);

	glutKeyboardFunc(keyPressd);
	glutKeyboardUpFunc(keyUp);

	glutSpecialFunc(keySpecial);
	glutSpecialUpFunc(keySpecialUp);

	glutMainLoop();		// Enter GLUT's main loop  
}

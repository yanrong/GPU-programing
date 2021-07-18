#include <GL/glew.h>		// Include the GLEW header file
#include <GL/glut.h>		// Include the GLUT header file
#include <stdio.h>

int keyStates[256] = {0};
int movingUp = 0;
float yLocation = 0.0f;
float yRotationAngle = 0.0f;

void KeyOperations(void)
{
	if(keyStates['a'])
	{
		printf("key a press \n");
	}
}

void display(void)
{
	KeyOperations();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);	// Clear the background of our window to red  
	glClear(GL_COLOR_BUFFER_BIT);	//Clear the colour buffer (more buffers later on)  

	glLoadIdentity();	// Load the Identity Matrix to reset our drawing locations  
	glTranslatef(0.0f, 0.0f, -5.0f); // Push eveything 5 units back into the scene, otherwise we won't see the primitive  
	
	glTranslatef(0.0f, yLocation, 0.0f);// Translate our object along the y axis  
	glRotatef(yRotationAngle, 0.0f, 1.0f, 0.0f);// Rotate our object around the y axis  
	
	glScalef(0.5f, 1.0f, 2.0f);
	glutWireCube(2.0f);
	//glutSolidCube(2.0f);
	glutSwapBuffers();		// Flush the OpenGL buffers to the window

	// rotate and translation
	if(movingUp) // if the item is in moving up
		yLocation -= 0.05f; // Move up along our yLocation
	else
		yLocation += 0.05f; // Move down alog our yLocation
	if (yLocation < -3.0f) // If we have gone up too far  
		movingUp = 0; // Reverse our direction so we are moving down  
	else if (yLocation > 3.0f) // Else if we have gone down too far  
		movingUp = 1; // Reverse our direction so we are moving up  
  
	yRotationAngle += 0.15f; // Increment our rotation value  
	if (yRotationAngle > 360.0f) // If we have rotated beyond 360 degrees (a full rotation)  
		yRotationAngle -= 360.0f; // Subtract 360 degrees off of our rotation  
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

void keyPressed(unsigned char key, int x, int y)
{
	keyStates[key] = 1;
}

void keyUp(unsigned char key, int x, int y)
{
	keyStates[key] = 0;
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);	// Initialize GLUT  
	glutInitDisplayMode(GLUT_DOUBLE);	// Set up a basic display buffer (only single buffered for now)  
	glutInitWindowSize(500, 500);	// Set the width and height of the window  
	glutInitWindowPosition(100, 100);	// Set the position of the window  
	glutCreateWindow("Your first OpenGL Window");	// Set the title for the window  

	glutDisplayFunc(display);	// Tell GLUT to use the method "display" for rendering  
	glutIdleFunc(display);
	glutReshapeFunc(reshape);

	glutKeyboardFunc(keyPressed); // Tell GLUT to use the method "keyPressed" for key presses  
	glutKeyboardUpFunc(keyUp); // Tell GLUT to use the method "keyUp" for key up events  
	
	glutMainLoop();		// Enter GLUT's main loop  
}

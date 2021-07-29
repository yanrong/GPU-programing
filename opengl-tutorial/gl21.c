#include <GL/gl.h>
#include <GL/glut.h>
#include <stdlib.h>

GLuint cubelist; //hold gen list here

//create the cube display list
void createcube(void)
{
	cubelist = glGenLists(1); //set the cube list to Generate a list
	glNewList(cubelist, GL_COMPILE); //compile the new list 
	glPushMatrix();
	glutSolidCube(2);
	glPopMatrix();
	glEndList();//end the list
}

void init(void)
{
	glEnable(GL_DEPTH_TEST); //enable the depth testing
	glEnable(GL_LIGHTING); //enable the lighting
	glEnable(GL_LIGHT0); //enable LIGHT0, our Diffuse light
	glShadeModel(GL_SMOOTH); //set the shader to smooth shader
	createcube(); //call the command to create the cube
}

void display(void) {
    glClearColor(0.0,0.0,0.0,1.0); //clear the screen to black
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//clear the color buffer and the depth buffer
    glLoadIdentity();  
    glTranslatef(0, 0, -5);
    glCallList(cubelist); //call the cube list
    glutSwapBuffers(); //swap the buffers
}

void reshape(int w, int h) {
    glViewport(0, 0,(GLsizei)w,(GLsizei)h); //set the viewport to the current window specifications
    glMatrixMode(GL_PROJECTION); //set the matrix to projection
    glLoadIdentity();
    gluPerspective(60,(GLfloat)w /(GLfloat)h, 1.0, 100.0); //set the perspective(angle of sight, width, height, , depth)
    glMatrixMode(GL_MODELVIEW); //set the matrix back to model
}

void keyboard(unsigned char key, int x, int y) {
    if(key==27)
    {
    	glutLeaveGameMode(); //set the resolution how it was
    	exit(0); //quit the program
    }
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH); //set the display to Double buffer, with depth
    glutGameModeString("1024x768:32@75"); //the settings for fullscreen mode
    glutEnterGameMode(); //set glut to fullscreen using the settings in the line above
    init(); //call the init function
    glutDisplayFunc(display); //use the display function to draw everything
    //update any variables in display, display can be changed to anyhing, as long as you move the variables to be updated, in this case, angle++;
    glutIdleFunc(display); 
    glutReshapeFunc(reshape); //reshape the window accordingly

    glutKeyboardFunc(keyboard); //check the keyboard
    glutMainLoop(); //call the main loop
    return 0;
}
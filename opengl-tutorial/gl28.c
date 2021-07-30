#include <GL/gl.h>
#include <GL/glut.h>
#include <stdio.h>

float angle = 0;
GLint stencilBits;
GLuint texture[40];

void freeTexture(GLuint texture)
{
	glDeleteTextures(1, &texture);
}

GLuint loadTexture(const char *filename, float width, float height)
{
	GLuint texture;
	unsigned char *data;
	FILE *file;
	file = fopen(filename, "rb");
	if(!file) return 0;
	data = (unsigned char *)malloc(width * height *3);
	fread(data, width * height * 3, 1, file);
	fclose(file);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	gluBuild2DMipmaps(GL_TEXTURE_2D, 3, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
	free(data);
	data = NULL;
	return texture;
}

void square(void)
{
	glPushMatrix();
	glBindTexture(GL_TEXTURE_2D, texture[0]);
	glTranslatef(0, 2.5, 0);
	glScalef(2, 2, 2);
	glBegin(GL_QUADS);
	glTexCoord2f(1, 0); //generate texture coodrination
	glVertex3f(-1, -1, 0);
	glTexCoord2f(1, 1);
	glVertex3f(-1, 1, 0);
	glTexCoord2f(0, 1);
	glVertex3f(1, 1, 0);
	glTexCoord2f(0, 0);
	glVertex3f(1, -1, 0);
	glEnd();
	glPopMatrix();
}

void bench(void)
{
	glPushMatrix();
	glColor4f(1, 1, 1, 0.7);
	glBindTexture(GL_TEXTURE_2D, texture[1]);
	glTranslatef(0, -2.5, 0);
	glScalef(4, 2, 4);
	glBegin(GL_QUADS);
	glTexCoord2f(1, 0); //generate texture coodrination
	glVertex3f(-1, -1, 1);
	glTexCoord2f(1, 1);
	glVertex3f(-1, 1, -0.5);
	glTexCoord2f(0, 1);
	glVertex3f(1, 1, -0.5);
	glTexCoord2f(0, 0);
	glVertex3f(1, -1, 1);
	glEnd();
	glPopMatrix();
}

void display(void)
{
	glClearStencil(0); //clear the stencil buffer
	glClearDepth(1.0f);
	glClearColor(0.0, 0.0, 0.0, 1);
	// clear the buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(0 ,0, -10);

	//start to draw shadow
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); //disable color mask
	glDepthMask(GL_FALSE); //disalbe the depth mask
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_ALWAYS, 1, 0xFFFFFFFF);
	glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE); // set th stencil buffer to keep our next lot of data
	bench();
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); //enable the color mask
	glDepthMask(GL_TRUE); //enable the depth mask
	glStencilFunc(GL_EQUAL, 1, 0xFFFFFFFF);
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP); // set the stencil buffer to keep our next lot of data 
	
	glDisable(GL_DEPTH_TEST); //disable the testing of shw shadow

	glPushMatrix();
	glScalef(1.0f, -1.0f, 1.0f); //filp the shadow vertically
	glTranslatef(0, 2, 0);//translate the shadow onto our drawing plane
	glRotatef(angle, 0, 1, 0);//rotate the shadow accordingly
	square();
	glPopMatrix();

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_STENCIL_TEST); //disable the stencil testing
	//end the shadow draw
	glEnable(GL_BLEND); //enable alpha blending
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // set the alpha blending
	bench(); //draw our bench
	glDisable(GL_BLEND); //disable blend
	glRotatef(angle, 0, 1, 0); //rotate the square
	square();
	glutSwapBuffers();
	angle++;
}

void init(void)
{
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_TEXTURE_2D);
	texture[0] = loadTexture("texture.raw", 256, 256);
	texture[0] = loadTexture("water.raw", 256, 256);
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat) w/ (GLfloat)h , 1.0, 1000.0);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char *argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_STENCIL); //add stencil buffer to window
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("A basic OpenGL window");
	init();
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
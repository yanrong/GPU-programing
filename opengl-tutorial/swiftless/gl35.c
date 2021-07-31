#include <GL/gl.h>
#include <GL/glut.h>
#include <stdio.h>

GLuint texture; //the array for our texture
GLuint texture2; //the array for our second texture

int cMap[10][10] = { //our map
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 0, 1, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 1, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 1, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 1, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
};

//function to load the RAW file
GLuint LoadTexture(const char * filename, int width, int height)
{
    GLuint texture;
    unsigned char *data;
    FILE *file;

    //The following code will read in our RAW file
    file = fopen(filename, "rb");
    if(file == NULL) return 0;
    data =(unsigned char *)malloc(width * height * 3);
    fread(data, width * height * 3, 1, file);
    fclose(file);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); //set texture environment parameters

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);

    //Here we are setting the parameter to repeat the texture instead of clamping the texture
    //to the edge of our shape. 
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    //Generate the texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    free(data); //free the texture
    return texture; //return whether it was successfull
}

void FreeTexture(GLuint texture)
{
    glDeleteTextures(1, &texture);
}

void drawTiles(void)
{ 
    //our function to draw the tiles
    for(int i = 0; i < 10; i++) //loop through the height of the map
    {
        for(int j = 0; j < 10; j++) //loop through the width of the map
        {
            if(cMap[i][j] == 0) //if the map at this position contains a 0
            {
                glBindTexture(GL_TEXTURE_2D, texture); //bind our grass texture to our shape
            }
            else //otherwise
            {
                glBindTexture(GL_TEXTURE_2D, texture2); //bind our dirt texture to our shape
            }
            
            glPushMatrix(); //push the matrix so that our translations only affect this tile
            glTranslatef(j, -i, 0); //translate the tile to where it should belong

            glBegin(GL_QUADS); //begin drawing our quads
            glTexCoord2d(0.0, 0.0);
            glVertex3f(0.0, 0.0, 0.0); //with our vertices we have to assign a texcoord
            glTexCoord2d(1.0, 0.0);
            glVertex3f(1.0, 0.0, 0.0); //so that our texture has some points to draw to
            glTexCoord2d(1.0, 1.0);
            glVertex3f(1.0, 1.0, 0.0);
            glTexCoord2d(0.0, 1.0);
            glVertex3f(0.0, 1.0, 0.0);
            glEnd();

            glPopMatrix(); //pop the matrix
        } //end first loop
    } //end second loop
}

void display(void)
{
    glClearColor(0.0,0.0,0.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glTranslatef(-5, 4, -20); //translate back a bit to view the map correctly
    drawTiles(); //draw our tiles
    glutSwapBuffers();
}

void reshape(int w, int h)
{
    glViewport(0, 0,(GLsizei)w,(GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60,(GLfloat)w /(GLfloat)h, 1.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("A basic OpenGL Window");
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutReshapeFunc(reshape);

    //Load our texture
    texture = LoadTexture("texture.raw", 256, 256);
    texture2 = LoadTexture("texture2.raw", 256, 256);

    glutMainLoop();

    //Free our texture
    FreeTexture(texture);
    FreeTexture(texture2);

    return 0;
}